"""CPU preflight: every selected row must produce real model inputs."""

import argparse
import gc
import os
import sys
import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.repo / "verl"))
    from verl.utils.dataset.rl_dataset import RLHFDataset

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    config = OmegaConf.create({
        "prompt_key": "problem", "image_key": "images", "video_key": "videos",
        "audio_key": "audios", "modalities": "audio,videos",
        "max_prompt_length": 896, "filter_overlong_prompts": False,
        "truncation": "error", "return_multi_modal_inputs": True,
        "format_prompt": str(args.repo / "training/rl/format_prompt/default.jinja"),
    })
    audit = {"max_indices": {}, "max_tokens": {}}
    for name, expected in (("train", 304), ("validation", 64)):
        dataset = RLHFDataset([str(args.dataset / f"{name}.parquet")], processor.tokenizer,
                              config, processor=processor)
        if len(dataset) != expected:
            raise AssertionError(f"{name}: expected {expected} rows, got {len(dataset)}")
        signatures = Counter()
        largest = 0
        for index in range(len(dataset)):
            row = dataset.dataframe[index]
            signature = row["modality_signature"]
            item = dataset[index]
            inputs = item.get("multi_modal_inputs", {})
            keys = set(inputs)
            if signature != "text" and not inputs:
                raise AssertionError(f"{name}[{index}]: missing media")
            if "audio" in signature and "input_features" not in keys:
                raise AssertionError(f"{name}[{index}]: missing audio features")
            if "video" in signature and "pixel_values_videos" not in keys:
                raise AssertionError(f"{name}[{index}]: missing video pixels")
            tokens = int(item["attention_mask"].sum())
            if tokens > 896:
                raise AssertionError(f"{name}[{index}]: prompt length {tokens}")
            largest = max(largest, tokens)
            if name == "train" and tokens > audit["max_tokens"].get(signature, -1):
                audit["max_tokens"][signature] = tokens
                audit["max_indices"][signature] = index
            signatures[signature] += 1
            del item
            if index % 25 == 0:
                gc.collect()
                print(name, index, "of", expected, flush=True)
        print(name, "PASS", dict(signatures), "max_prompt_tokens", largest, flush=True)
        if name == "train":
            import torch
            from torch.utils.data import RandomSampler, BatchSampler
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl import DataProto
            batches = list(BatchSampler(RandomSampler(dataset, generator=torch.Generator().manual_seed(45)), 8, True))
            flat = [i for b in batches for i in b]
            assert len(batches) == 38 and sorted(flat) == list(range(304))
            multi = [b for b in batches if len({dataset.dataframe[i]['task'] for i in b}) > 1]
            assert multi, 'Change 3 would be inactive in every batch'
            sample = DataProto.from_single_dict(collate_fn([dataset[i] for i in multi[0]]))
            repeated = sample.repeat(repeat_times=4, interleave=True)
            assert len(repeated) == 32 and len(repeated.non_tensor_batch['multi_modal_inputs']) == 32
            assert sample.batch['position_ids'].shape == (8,3,896)
            audit['mixed_task_batches'] = len(multi)
            print('MIXED_BATCH_PASS',len(multi),'of',len(batches),flush=True)
    (Path(__file__).parent / 'cpu_audit_result.json').write_text(json.dumps(audit, indent=2))
    print("ALL_MEDIA_PASS")


if __name__ == "__main__":
    main()
