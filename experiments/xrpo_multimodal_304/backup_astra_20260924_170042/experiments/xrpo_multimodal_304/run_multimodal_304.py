"""Controlled, opt-in 304-row multimodal HARPO/XRPO study launcher.

Default is a dry run. Every arm shares reward B, data, model and optimizer.
The original_harpo arm uses the fork's original HARPO implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


SEED = 45
RUNS = ("original_harpo", "paper_harpo", "xrpo_23")


def setting(key: str, value, *, add=False) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"{'+' if add else ''}{key}={value}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reward-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=38)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--registration-confirmed", action="store_true")
    args = parser.parse_args()
    if args.steps != 38:
        parser.error("304 rows at batch size 8 require exactly 38 steps")
    for directory in (args.repo, args.model, args.reward_model, args.dataset):
        if not directory.is_dir():
            parser.error(f"Missing directory: {directory}")
    train = args.dataset / "train.parquet"
    validation = args.dataset / "validation.parquet"
    if pq.ParquetFile(train).metadata.num_rows != 304:
        parser.error("Training parquet must have 304 rows")
    if pq.ParquetFile(validation).metadata.num_rows != 64:
        parser.error("Validation parquet must have 64 rows")
    experiment = Path(__file__).resolve().parent
    output = args.output_root / args.run
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        setting("ray_init.num_cpus", 4),
        setting("algorithm.adv_estimator", "harpo"),
        setting("data.train_files", train),
        setting("data.val_files", validation),
        setting("data.train_batch_size", 8),
        setting("data.val_batch_size", 8),
        setting("data.max_prompt_length", 896),
        setting("data.max_response_length", 256),
        setting("data.prompt_key", "problem"),
        setting("data.image_key", "images"),
        setting("data.video_key", "videos"),
        setting("data.audio_key", "audios"),
        "data.modalities='audio,videos'",
        setting("data.return_multi_modal_inputs", True),
        setting("data.train_modality_batching.enabled", True),
        setting("data.train_modality_batching.drop_last", True),
        setting("data.val_modality_batching.enabled", True),
        setting("data.val_modality_batching.drop_last", False),
        setting("data.filter_overlong_prompts", False),
        setting("data.truncation", "error"),
        setting("data.shuffle", False),
        setting("data.seed", SEED, add=True),
        setting("data.dataloader_num_workers", 0),
        setting("data.format_prompt", args.repo / "training/rl/format_prompt/default.jinja"),
        setting("actor_rollout_ref.model.path", args.model),
        setting("actor_rollout_ref.model.lora_rank", 16),
        setting("actor_rollout_ref.model.lora_alpha", 32),
        "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
        setting("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        setting("actor_rollout_ref.model.use_liger", False),
        setting("actor_rollout_ref.model.use_remove_padding", False),
        setting("actor_rollout_ref.actor.optim.lr", "1e-6"),
        setting("actor_rollout_ref.actor.ppo_mini_batch_size", 8),
        setting("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1),
        setting("actor_rollout_ref.actor.use_dynamic_bsz", False),
        setting("actor_rollout_ref.actor.fsdp_config.model_dtype", "float16", add=True),
        setting("actor_rollout_ref.actor.fsdp_config.use_orig_params", False, add=True),
        setting("actor_rollout_ref.actor.use_torch_compile", False),
        setting("actor_rollout_ref.rollout.name", "hf"),
        setting("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        setting("actor_rollout_ref.rollout.dtype", "float16"),
        setting("actor_rollout_ref.rollout.top_k", 0),
        setting("actor_rollout_ref.rollout.n", 4),
        setting("actor_rollout_ref.rollout.val_kwargs.n", 1),
        setting("actor_rollout_ref.rollout.val_kwargs.do_sample", False),
        setting("actor_rollout_ref.rollout.val_kwargs.temperature", 0),
        setting("actor_rollout_ref.rollout.micro_batch_size", 1, add=True),
        setting("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1),
        setting("algorithm.use_kl_in_reward", False),
        setting("actor_rollout_ref.actor.use_kl_loss", False),
        setting("custom_reward_function.path", experiment / "reward_b_local.py"),
        setting("custom_reward_function.name", "human_behaviour_compute_score_batch"),
        setting("reward_model.reward_manager", "batch"),
        setting("trainer.logger", "[console]"),
        setting("trainer.project_name", "p40-xrpo-multimodal-304"),
        setting("trainer.experiment_name", args.run),
        setting("trainer.n_gpus_per_node", 1),
        setting("trainer.nnodes", 1),
        setting("trainer.val_before_train", False),
        setting("trainer.save_freq", 999999),
        setting("trainer.test_freq", 999999),
        setting("trainer.total_epochs", 38),
        setting("trainer.total_training_steps", 38),
        setting("trainer.default_local_dir", output),
        setting("trainer.validation_data_dir", output / "final_test"),
        setting("trainer.resume_mode", "disable"),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HBA_EXPERIMENT_SEED": str(SEED),
        "HBA_RUN_DIR": str(output),
        "HBA_REWARD_EMBEDDING_MODEL": str(args.reward_model),
        "PYTHONPATH": os.pathsep.join((str(experiment), str(args.repo / "experiments/p40_text_trend"),
                                         str(args.repo / "verl"), env.get("PYTHONPATH", ""))),
        "PYTHONHASHSEED": str(SEED),
        "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled",
        "RAY_ADDRESS": "local", "RAY_TMPDIR": "/tmp/anonymous",
        "VERL_ATTENTION_BACKEND": "eager", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    if args.run == "original_harpo":
        env.pop("HBA_XRPO_MODE", None)
    else:
        env["HBA_XRPO_MODE"] = args.run
    print("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"])
    print("mode=" + args.run)
    print(shlex.join(str(part) for part in command))
    if args.validate_config:
        checked = subprocess.run([*map(str, command), "--cfg", "job"], cwd=args.repo,
                                 env=env, text=True, capture_output=True)
        if checked.returncode:
            raise RuntimeError("Hydra config validation failed:\n" + checked.stderr[-6000:])
        print("HYDRA_CONFIG_VALID")
    if not args.execute:
        print("dry_run=True")
        return
    if not args.registration_confirmed:
        parser.error("GPU work requires a current matching resource registration")
    from resource_monitor import assert_free_gpus
    assert_free_gpus({args.gpu})
    output.mkdir(parents=True, exist_ok=False)
    source_files = [
        train, validation, args.dataset / "manifest.json", experiment / "reward_b_local.py",
        experiment / "original_reward.py", experiment / "multimodal_evaluation.py",
        args.repo / "verl/verl/trainer/ppo/xrpo_advantage.py",
        args.repo / "verl/verl/trainer/ppo/ray_trainer.py",
        args.repo / "verl/verl/workers/rollout/hf_rollout.py",
        args.repo / "verl/verl/workers/reward_manager/batch.py",
        args.repo / "verl/verl/utils/dataset/rl_dataset.py",
        args.repo / "verl/verl/utils/dataset/audio_utils.py",
        args.repo / "verl/verl/utils/dataset/vision_utils.py",
    ]
    manifest = {
        "run": args.run, "algorithm": args.run, "reward_function": "reward_b_local.py",
        "seed": SEED, "gpus": [str(args.gpu)], "total_steps": 38,
        "train": str(train), "validation": str(validation), "validation_rows": 64,
        "model": str(args.model), "reward_model": str(args.reward_model),
        "command": [str(part) for part in command],
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}), encoding="utf-8")
    monitor = [sys.executable, str(args.repo / "experiments/p40_text_trend/resource_monitor.py"),
               "--gpus", str(args.gpu), "--output", str(output / "resource_usage.csv"),
               "--log", str(output / "training.log"), "--", *map(str, command)]
    try:
        subprocess.run(monitor, cwd=args.repo, env=env, check=True)
        subprocess.run([sys.executable, str(experiment / "verify_multimodal_304.py"),
                        "--run-dir", str(output)], cwd=args.repo, env=env, check=True)
    except BaseException as exc:
        (output / "status.json").write_text(json.dumps({"state": "failed", "error": str(exc)}), encoding="utf-8")
        raise
    (output / "status.json").write_text(json.dumps({"state": "complete", "steps": 38}), encoding="utf-8")


if __name__ == "__main__":
    main()
