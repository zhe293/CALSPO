"""CPU-only validation that each study modality reaches the model processor."""

import argparse
import os
import sys
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
    dataset = RLHFDataset([str(args.dataset / "train.parquet")], processor.tokenizer, config, processor=processor)
    for signature in ("text", "text_audio", "text_video_audio"):
        i = next(i for i, row in enumerate(dataset.dataframe) if row["modality_signature"] == signature)
        item = dataset[i]
        processed = item.get("multi_modal_inputs", {})
        if signature != "text" and not processed:
            raise AssertionError(f"No processed media inputs for {signature}")
        if item["data_source"] != item["task"]:
            raise AssertionError("Reward task metadata lost")
        print(signature, "index", i, "tokens", int(item["attention_mask"].sum()),
              "processed_keys", sorted(processed), flush=True)
    print("CPU_MEDIA_CHECK_PASS")


if __name__ == "__main__":
    main()
