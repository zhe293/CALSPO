"""CPU regression for dataset -> trainer split -> HF multimodal generation."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.repo / "experiments/p40_text_trend"), str(args.repo / "verl")]
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    import verl.workers.rollout.hf_rollout as hf

    hf.get_device_name = lambda: "cpu"
    hf.get_torch_device = lambda: SimpleNamespace(empty_cache=lambda: None)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    config = OmegaConf.create({
        "prompt_key": "problem", "image_key": "images", "video_key": "videos",
        "audio_key": "audios", "modalities": "audio,videos",
        "max_prompt_length": 896, "filter_overlong_prompts": False,
        "truncation": "error", "return_multi_modal_inputs": True,
        "format_prompt": str(args.repo / "training/rl/format_prompt/default.jinja"),
    })
    dataset = RLHFDataset([str(args.dataset / "train.parquet")], processor.tokenizer,
                          config, processor=processor)

    class Fake(torch.nn.Module):
        def generate(self, input_ids, **kwargs):
            self.forwarded_keys = set(kwargs)
            response = torch.tensor([[5, 2]], device=input_ids.device)
            return SimpleNamespace(sequences=torch.cat([input_ids, response], dim=1))

    for signature, required in (
        ("text_audio", {"input_features"}),
        ("text_video_audio", {"input_features", "pixel_values_videos", "video_grid_thw"}),
    ):
        index = next(i for i, row in enumerate(dataset.dataframe)
                     if row["modality_signature"] == signature)
        batch = DataProto.from_single_dict(collate_fn([dataset[index]]))
        pop_keys = ["raw_prompt_ids", "task", "class_label", "multi_modal_data",
                    "multi_modal_inputs", "modality_signature"]
        generated = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[key for key in pop_keys if key in batch.non_tensor_batch],
        )
        for key in ("task", "class_label", "multi_modal_inputs", "modality_signature"):
            batch.non_tensor_batch[key] = generated.non_tensor_batch[key].copy()
        generated.meta_info = {"eos_token_id": 2, "pad_token_id": 0}
        model = Fake()
        rollout = hf.HFRollout(model, OmegaConf.create({
            "do_sample": True, "temperature": 1.0, "response_length": 4,
            "top_p": 1.0, "top_k": 0, "micro_batch_size": 1,
        }))
        output = rollout.generate_sequences(generated)
        if len(output) != 1 or not required.issubset(model.forwarded_keys):
            raise AssertionError(f"Media was lost during {signature} rollout")
        if output.batch["position_ids"].shape[-1] != 900:
            raise AssertionError("Multimodal response position IDs were not extended")
        if not required.issubset(batch.non_tensor_batch["multi_modal_inputs"][0]):
            raise AssertionError("Actor side lost media inputs")
        print("MEDIA_PATH_PASS", signature, sorted(required), flush=True)


if __name__ == "__main__":
    main()
