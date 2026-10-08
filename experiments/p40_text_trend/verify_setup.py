"""CPU-only preflight for the P40 text trend experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--reward-model", type=Path, required=True)
    args = parser.parse_args()

    require(args.repo.is_dir(), f"missing repo: {args.repo}")
    require(args.model.is_dir(), f"missing model: {args.model}")
    require(args.train.is_file(), f"missing train subset: {args.train}")
    require(args.validation.is_file(), f"missing validation subset: {args.validation}")
    require(args.reward_model.is_dir(), f"missing reward model: {args.reward_model}")

    index_path = args.model / "model.safetensors.index.json"
    require(index_path.is_file(), f"missing model index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = sorted({name for name in index["weight_map"].values() if not (args.model / name).is_file()})
    require(not missing, f"missing model shards: {missing}")

    import pyarrow.parquet as pq
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5OmniThinkerConfig

    for split in (args.train, args.validation):
        schema = pq.read_schema(split)
        for column in ("problem", "answer", "task", "modality_signature"):
            require(column in schema.names, f"{split} lacks column {column}")
        require(pq.ParquetFile(split).metadata.num_rows > 0, f"empty subset: {split}")

    config = Qwen2_5OmniThinkerConfig.from_pretrained(args.model, local_files_only=True)
    AutoProcessor.from_pretrained(args.model, local_files_only=True)

    sys.path.insert(0, str(args.repo / "verl"))
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: F401

    require("grpo" in ADV_ESTIMATOR_REGISTRY, "GRPO is not registered")
    require("harpo" in ADV_ESTIMATOR_REGISTRY, "HARPO is not registered; setup_verl patch is missing")
    fsdp_worker = (args.repo / "verl" / "verl" / "workers" / "fsdp_workers.py").read_text(encoding="utf-8")
    require("HBA_EXPERIMENT_SEED" in fsdp_worker, "experiment seed patch is missing; rerun apply_p40_compat.py")

    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"transformers={transformers.__version__}")
    print(f"model_type={config.model_type}")
    print("grpo_registered=True harpo_registered=True experiment_seed_patch=True")
    print("preflight=PASS (CPU only; GPU execution remains untested)")


if __name__ == "__main__":
    main()
