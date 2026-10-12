"""Build or execute one controlled GRPO/HARPO text trend run.

The default is a dry run. Actual execution requires both --execute and
--registration-confirmed so an unregistered GPU job cannot start accidentally.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path


RUNS = {
    "grpo_reward_a": ("grpo", "reward_a_local.py"),
    "grpo_reward_b": ("grpo", "reward_b_local.py"),
    "harpo_reward_a": ("harpo", "reward_a_local.py"),
    "harpo_reward_b": ("harpo", "reward_b_local.py"),
}

EXPERIMENT_SEED = 45


def override(key: str, value: object, add: bool = False) -> str:
    prefix = "+" if add else ""
    if isinstance(value, bool):
        value = str(value).lower()
    return f"{prefix}{key}={value}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reward-model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Explicit IDs, for example 0,1")
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=[EXPERIMENT_SEED], default=EXPERIMENT_SEED)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Start a detached session; requires --execute")
    parser.add_argument("--registration-confirmed", action="store_true")
    args = parser.parse_args()
    if args.total_steps < 1:
        parser.error('--total-steps must be positive')
    if args.detach and not args.execute:
        parser.error('--detach requires --execute')

    algorithm, reward_name = RUNS[args.run]
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) > 2 or any(not item.isdigit() for item in gpu_ids):
        parser.error("--gpus must contain one or two numeric GPU IDs")
    if len(set(gpu_ids)) != len(gpu_ids):
        parser.error('GPU IDs must be distinct')
    for path in (args.repo, args.model, args.reward_model):
        if not path.is_dir():
            parser.error(f"missing directory: {path}")
    for path in (args.train, args.validation):
        if not path.is_file():
            parser.error(f"missing file: {path}")

    experiment_dir = Path(__file__).resolve().parent
    reward_path = experiment_dir / reward_name
    output = args.output_root / args.run
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        override("ray_init.num_cpus", 4),
        override("algorithm.adv_estimator", algorithm),
        override("data.train_files", args.train),
        override("data.val_files", args.validation),
        override("data.train_batch_size", 8),
        override("data.val_batch_size", 8),
        override("data.max_prompt_length", 512),
        override("data.max_response_length", 256),
        override("data.prompt_key", "problem"),
        override("data.custom_cls.path", experiment_dir / 'text_dataset.py'),
        override("data.custom_cls.name", 'TextTrendDataset'),
        override("data.truncation", 'middle'),
        override("data.shuffle", False),
        override("data.seed", args.seed, add=True),
        override("data.dataloader_num_workers", 0),
        override("data.modalities", ""),
        override("data.return_multi_modal_inputs", False),
        override("data.train_modality_batching.enabled", False),
        override("data.val_modality_batching.enabled", False),
        override("data.format_prompt", experiment_dir / "text_prompt.jinja"),
        override("actor_rollout_ref.model.path", args.model),
        override("actor_rollout_ref.model.lora_rank", 16),
        override("actor_rollout_ref.model.lora_alpha", 32),
        "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
        override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        override("actor_rollout_ref.model.use_liger", False),
        override("actor_rollout_ref.model.use_remove_padding", False),
        override("actor_rollout_ref.actor.optim.lr", "1e-6"),
        override("actor_rollout_ref.actor.ppo_mini_batch_size", 8),
        override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1),
        override("actor_rollout_ref.actor.use_dynamic_bsz", False),
        override("actor_rollout_ref.actor.fsdp_config.model_dtype", "float16", add=True),
        override("actor_rollout_ref.actor.fsdp_config.use_orig_params", False, add=True),
        override("actor_rollout_ref.actor.use_torch_compile", False),
        override("actor_rollout_ref.rollout.name", "hf"),
        override("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        override("actor_rollout_ref.rollout.dtype", "float16"),
        override("actor_rollout_ref.rollout.top_k", 0),
        override("actor_rollout_ref.rollout.n", 4),
        override("actor_rollout_ref.rollout.val_kwargs.n", 1),
        override("actor_rollout_ref.rollout.val_kwargs.do_sample", False),
        override("actor_rollout_ref.rollout.val_kwargs.temperature", 0),
        override("actor_rollout_ref.rollout.micro_batch_size", 1, add=True),
        override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1),
        override("algorithm.use_kl_in_reward", False),
        override("actor_rollout_ref.actor.use_kl_loss", False),
        override("custom_reward_function.path", reward_path),
        override("custom_reward_function.name", "human_behaviour_compute_score_batch"),
        override("reward_model.reward_manager", "batch"),
        override("trainer.logger", "[console]"),
        override("trainer.project_name", "p40-harpo-trend"),
        override("trainer.experiment_name", args.run),
        override("trainer.n_gpus_per_node", len(gpu_ids)),
        override("trainer.nnodes", 1),
        override("trainer.val_before_train", False),
        override("trainer.save_freq", 999999),
        override("trainer.test_freq", 999999),
        override("trainer.total_epochs", args.total_steps),
        override("trainer.total_training_steps", args.total_steps),
        override("trainer.default_local_dir", output),
        override("trainer.validation_data_dir", output / "final_test"),
        override("trainer.resume_mode", "disable"),
    ]

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "PYTHONPATH": os.pathsep.join([str(experiment_dir), str(args.repo / 'verl'), env.get('PYTHONPATH', '')]),
            "PYTHONHASHSEED": str(args.seed),
            "HBA_EXPERIMENT_SEED": str(args.seed),
            "HBA_RUN_DIR": str(output),
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RAY_ADDRESS": "local",
            "RAY_TMPDIR": "/tmp/anonymous",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VERL_ATTENTION_BACKEND": "eager",
            "HBA_REWARD_EMBEDDING_MODEL": str(args.reward_model),
        }
    )
    print("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"])
    print(shlex.join(str(item) for item in command))
    if not args.execute:
        print("dry_run=True")
        return
    if not args.registration_confirmed:
        parser.error("GPU execution requires --registration-confirmed after the resource table is updated")
    if args.detach:
        args.output_root.mkdir(parents=True, exist_ok=True)
        launcher = args.output_root / (args.run + '.launcher.log')
        with launcher.open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      *[a for a in sys.argv[1:] if a != '--detach']],
                                     cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.output_root / (args.run+'.pid')).write_text(str(child.pid),encoding='utf-8')
        print(f'detached_pid={child.pid} launcher_log={launcher}')
        return
    # Fail before loading any weights if a chosen card has a compute process.
    from resource_monitor import assert_free_gpus
    assert_free_gpus(set(map(int,gpu_ids)))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run": args.run,
        "algorithm": algorithm,
        "reward_function": reward_name,
        "seed": args.seed,
        "gpus": gpu_ids,
        "total_steps": args.total_steps,
        "model": str(args.model),
        "reward_model": str(args.reward_model),
        "train": str(args.train),
        "validation": str(args.validation),
        "command": [str(item) for item in command],
        "validation_rows": __import__('pyarrow.parquet', fromlist=['ParquetFile']).ParquetFile(args.validation).metadata.num_rows,
        "precision": "FP16 frozen base; FP32 trainable LoRA; nested FSDP FP16 autocast; sharded gradient scaling",
        "truncation": "middle, 512 prompt tokens, same for every condition",
    }
    hash_files = [args.train, args.validation, args.model / 'config.json', args.model / 'model.safetensors.index.json',
                  *experiment_dir.glob('*.py'), experiment_dir/'text_prompt.jinja',
                  * (args.repo/'training/rl/reward_function').glob('human_behaviour*.py')]
    hash_files += [args.repo/'verl/verl'/p for p in ('workers/fsdp_workers.py','workers/actor/dp_actor.py',
                  'workers/rollout/hf_rollout.py','trainer/ppo/ray_trainer.py','trainer/ppo/core_algos.py','trainer/main_ppo.py')]
    manifest['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in hash_files}
    manifest['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    monitor_command = [
        sys.executable,
        str(experiment_dir / "resource_monitor.py"),
        "--gpus",
        ",".join(gpu_ids),
        "--output",
        str(output / "resource_usage.csv"),
        "--log",
        str(output / "training.log"),
        "--",
        *[str(item) for item in command],
    ]
    status_path = output/'status.json'
    status_path.write_text(json.dumps({'state':'running','pid':os.getpid()}),encoding='utf-8')
    try:
        subprocess.run(monitor_command, cwd=args.repo, env=env, check=True)
        subprocess.run([sys.executable,str(experiment_dir/'verify_run_complete.py'),
                        '--run-dir',str(output),'--expected-steps',str(args.total_steps)],cwd=args.repo,env=env,check=True)
    except BaseException as exc:
        status_path.write_text(json.dumps({'state':'failed','error':str(exc)}),encoding='utf-8')
        raise
    status_path.write_text(json.dumps({'state':'complete','seed':args.seed,'steps':args.total_steps}),encoding='utf-8')


if __name__ == "__main__":
    main()
