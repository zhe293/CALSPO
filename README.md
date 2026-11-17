# CALSPO / EquiAffect-3B

**Class-Aware Supervised Policy Optimization (CALSPO)**

This repository provides the **CALSPO training code**, two AV + HARPO ablation implementations, and the **EquiAffect-3B LoRA adapter**. The code builds on Human Behavior Atlas / OmniSapiens and VERL. Implementations of AV advantage recovery, HARPO modulation, and class-weighted label supervision are located in `verl/verl/trainer/ppo/`.

## Three Configurations

| Configuration | Task-scale smoothing: previous / new | AV | Class-weighted label supervision |
|---|---|---|---|
| CALSPO | 75% / 25% | Enabled | Enabled, λ = 1 |
| ema075 | 75% / 25% | Enabled | Disabled |
| ema095 | 95% / 5% | Enabled | Disabled |

Each implementation comes from the code snapshot used in its corresponding experiment.  These smoothing ratios apply specifically to HARPO task-scale updates, rather than to all smoothing parameters. Complete historical run settings are preserved in each implementation's `provenance/source_run_manifest.json`. The portable runner uses the same random seed by default to facilitate subsequent controlled reruns.

## Directory Structure

```text
training/                       Training code, configurations, prompts, and rewards
verl/                           Modified veRL training framework and license
experiments/                    Experiment entry points and training utilities
ablations/ema075/                AV + HARPO implementation, 75% / 25% task-scale smoothing
ablations/ema095/                AV + HARPO implementation, 95% / 5% task-scale smoothing
provenance/                     Training run settings and environment snapshot
scripts/                        Training, model download, and verification tools
evaluation/                     Evaluation utilities
models/EquiAffect-3B/            Trained LoRA adapter and loading instructions
models/base_model_metadata/     Base-model metadata, shard index, and license
```

## External Inputs

CALSPO was trained on Qwen2.5-Omni-3B. The recorded base-model revision is
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`.
The experiments used **2,160 records** in total. Obtain the experiment data separately.

To rerun the HBA experiment, supply the external data directory with samples.parquet, manifest.json, and the referenced HBA media under media/. The launcher reconstructs the original run inputs using the archived row assignments.

## Model Weights

Install Git LFS and run `git lfs pull` after cloning to retrieve the trained adapter. Run
`python scripts/verify_equiaffect.py` to verify the adapter. See the
[model README](models/EquiAffect-3B/README.md) for loading instructions.

The base-model download script obtains the three official Qwen2.5-Omni-3B shards
at the recorded revision and verifies their hashes. It can also retrieve the reward embedding model.

```bash
python -m pip install huggingface_hub
python scripts/download_base_model.py --output models/Qwen2.5-Omni-3B --embedding-output models/all-MiniLM-L6-v2
```

## Environment And Execution

Training runs in a Linux / CUDA environment. On Windows, the code can be inspected and the launcher can print the resolved training arguments. The original experimental environment used Python 3.10, PyTorch 2.5.1+cu118, Transformers 4.52.2, Ray 2.44.1, and PEFT 0.15.2. The complete recorded package versions are available in `provenance/environment_snapshot.json`. This file records the original environment and does not establish that installation has succeeded on a new machine.

Installation files are provided in `verl/requirements.txt`, `verl/setup.py`, and `scripts/setup_verl.sh`. The archived source includes the modifications required by the experiments. Install this repository's version rather than overwriting it with the upstream version.

```bash
bash scripts/setup_verl.sh
# First install any remaining dependencies in your Linux environment to match the environment snapshot.
python scripts/run_experiment.py --variant calspo --gpu 0 --data-dir /path/to/external-data --base-model models/Qwen2.5-Omni-3B --embedding-model models/all-MiniLM-L6-v2 --output outputs/calspo
```

By default, the runner only prints the resolved arguments. Add `--execute` to start training. Set `--variant` to `ema075` or `ema095` to select the corresponding complete implementation. Each configuration requires its own empty output directory, and automatic resumption is disabled. Settings inherited from the original run manifests include a batch size of 8, four sampled responses per question, a learning rate of 1e-6, LoRA rank 16 / alpha 32, and 38 training steps.

Execution with `--execute` requires `--data-dir` and verifies the external inputs before training. Use `bash scripts/setup_verl.sh --check-only` to inspect the veRL installation source.

Before execution, check GPU availability and register usage according to the rules of your server. Preparing this archive involved only CPU-based file, data, and argument checks. GPU training and evaluation were not rerun. Historical helper scripts may retain paths from the original server. Use `scripts/run_experiment.py` as the recommended entry point.

## Sources And Licenses

Original method project: [OmniSapiens / Human Behavior Atlas](https://github.com/MIT-MI/human_behavior_atlas). VERL's Apache 2.0 license is preserved in `verl/LICENSE` within each code snapshot. Model licenses apply separately. Original datasets and project code remain subject to their respective source terms. This repository does not grant a new blanket license for redistribution or reuse.
