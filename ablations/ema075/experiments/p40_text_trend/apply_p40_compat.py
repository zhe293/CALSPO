"""Apply the two minimal compatibility changes required by Tesla P40.

This edits only the caller's personal verl checkout. It preserves the upstream
FlashAttention default while allowing VERL_ATTENTION_BACKEND=eager, and changes
HF rollout autocast to FP16 on GPUs without BF16 support.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_checked(path: Path, old: str, new: str, expected: int) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already_applied={path}")
        return
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} matches, found {count}; refusing to edit")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched={path} replacements={count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    verl = args.repo / "verl" / "verl"

    replace_checked(
        verl / "workers" / "fsdp_workers.py",
        'attn_implementation="flash_attention_2"',
        'attn_implementation=os.environ.get("VERL_ATTENTION_BACKEND", "flash_attention_2")',
        # The HBA patch adds critic/reward-model loader sites, giving four
        # occurrences in the patched checkout.
        expected=4,
    )
    replace_checked(
        verl / "workers" / "rollout" / "hf_rollout.py",
        "        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):\n",
        "        autocast_dtype = torch.bfloat16\n"
        "        if get_device_name() == \"cuda\" and not torch.cuda.is_bf16_supported():\n"
        "            autocast_dtype = torch.float16\n"
        "        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=autocast_dtype):\n",
        expected=1,
    )
    replace_checked(
        verl / "workers" / "fsdp_workers.py",
        "import os\nimport warnings\n",
        "import os\nimport random\nimport warnings\n",
        expected=1,
    )
    replace_checked(
        verl / "workers" / "fsdp_workers.py",
        "    def init_model(self):\n        from verl.workers.actor import DataParallelPPOActor\n",
        "    def init_model(self):\n"
        "        from verl.workers.actor import DataParallelPPOActor\n\n"
        "        # Reproducible per-rank RNG stream for the controlled P40 experiment.\n"
        "        experiment_seed = int(os.environ.get(\"HBA_EXPERIMENT_SEED\", \"45\"))\n"
        "        rank_seed = experiment_seed  # identical initial LoRA on every rank\n"
        "        random.seed(rank_seed)\n"
        "        np.random.seed(rank_seed)\n"
        "        torch.manual_seed(rank_seed)\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.manual_seed_all(rank_seed)\n",
        expected=1,
    )


if __name__ == "__main__":
    main()
