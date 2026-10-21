# CALSPO / EquiAffect-3B

**Class-Aware Supervised Policy Optimization (CALSPO)**

This repository provides both the **CALSPO training algorithm** and the **EquiAffect-3B model weights** trained with it. It also archives the training and evaluation data, two AV + HARPO ablation implementations, and the manuscript draft. The code builds on Human Behavior Atlas / OmniSapiens and VERL. Implementations of AV advantage recovery, HARPO modulation, and class-weighted label supervision are located in `verl/verl/trainer/ppo/`. This is a private archive for a personal research project.

## Three Configurations

| Configuration | Task-scale smoothing: previous / new | AV | Class-weighted label supervision |
|---|---|---|---|
| CALSPO | 75% / 25% | Enabled | Enabled, λ = 1 |
| ema075 | 75% / 25% | Enabled | Disabled |
| ema095 | 95% / 5% | Enabled | Disabled |

Each implementation comes from the code snapshot used in its corresponding experiment. The SHA-256 hashes of key files match those recorded in the original run manifests. These smoothing ratios apply specifically to HARPO task-scale updates, rather than to all smoothing parameters. Complete historical run settings are preserved in each implementation's `provenance/source_run_manifest.json`. The portable runner uses the same random seed by default to facilitate subsequent controlled reruns. Archiving these implementations does not imply that all ablation results have been completed.

## Repository Structure

```text
verl/                           Complete VERL implementation of CALSPO
training/                       Training templates and original project training code
evaluation/                     Original project evaluation code
ablations/ema075/               Complete code snapshot without supervision, 75% / 25% smoothing
ablations/ema095/               Complete code snapshot without supervision, 95% / 5% smoothing
datasets/train.parquet          304-item training set with portable media paths
datasets/validation.parquet     216-item validation set used in the original experiments
datasets/eval328.parquet        Frozen 328-item evaluation set with portable media paths
datasets/media/                 887 required audio and video files
datasets/original/              Original Parquet files, preserved byte for byte
datasets/provenance/eval328/     Evaluation-set construction, filtering, and verification records
models/EquiAffect-3B/            Trained LoRA weights, configuration, verification, and loading instructions
paper/                          Synchronized manuscript drafts in Markdown, LaTeX, and PDF
models/base_model_metadata/     Base-model configuration, tokenizer, license, and related files
models/base_model_manifest.json Pinned revision and weight verification information
provenance/                     Original run, environment, and collection records
scripts/                        Portable runners, model downloaders, and asset verification tools
```

Dataset migration changes only media paths. Questions, labels, ordering, and table schemas remain unchanged. The mapping between original and portable paths is recorded in `datasets/media_mapping.json`. Media files use short filenames for Windows compatibility. The 328-item set retains the existing frozen data and its original construction records. The repository preserves the data provenance and selection process. The training set contains 224 emotion-classification items and 80 depression-classification items. The corresponding evaluation counts are 287 and 41.

## Trained EquiAffect-3B Model

EquiAffect-3B uses Qwen2.5-Omni-3B as its base model and is jointly trained on emotion and depression classification with CALSPO. The complete trained low-rank adaptation (LoRA) weights from the step-38 checkpoint are stored in [models/EquiAffect-3B](models/EquiAffect-3B/README.md). Load them together with the pinned base-model revision. The weights are stored using Git LFS. After cloning, run `git lfs pull` to download the actual weight files. Then run `python scripts/verify_equiaffect.py` to verify file sizes, SHA-256 hashes, and target-layer coverage.

All LoRA parameters were extracted from the complete FSDP checkpoint, correcting the omission of language decoder layers in the original export. Parameter coverage, finite values, and consistency with the parameters in the original export have been verified. GPU training and inference were not rerun during this extraction and verification. Loading instructions and provenance records are provided in the model directory.

Manuscript files are available as a [Markdown draft](paper/CALSPO论文中文初稿.md), [LaTeX source](paper/latex/main.tex), and [PDF preview](paper/latex/main.pdf). The manuscript presents CALSPO and EquiAffect-3B as the algorithm and model outputs, respectively. It also cites Wang et al. (Interspeech 2023) as background on overfitting to the majority neutral class.

## Base Model

The experiments use **Qwen/Qwen2.5-Omni-3B** as the base model. This GitHub repository includes its configuration, tokenizer, model-file manifest, and download script. Model weights are retrieved from a pinned revision of the official Hugging Face repository.

- Official repository: https://huggingface.co/Qwen/Qwen2.5-Omni-3B
- Pinned revision: `f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- License: Qwen Research License. See `models/base_model_metadata/LICENSE` and `NOTICE`.

```bash
pip install huggingface_hub
python scripts/download_base_model.py --output models/Qwen2.5-Omni-3B --embedding-output models/all-MiniLM-L6-v2
```

The sizes and SHA-256 hashes of all three weight shards are verified automatically after download. The reward implementation also uses a local copy of `all-MiniLM-L6-v2`. This auxiliary model can be downloaded using the optional argument shown above.

## Environment And Execution

Training runs in a Linux / CUDA environment. On Windows, the repository can be inspected, assets verified, and run arguments printed. The original experimental environment used Python 3.10, PyTorch 2.5.1+cu118, Transformers 4.52.2, Ray 2.44.1, and PEFT 0.15.2. The complete recorded package versions are available in `provenance/environment_snapshot.json`. This file records the original environment and does not establish that installation has succeeded on a new machine.

Installation files are provided in `verl/requirements.txt`, `verl/setup.py`, and `scripts/setup_verl.sh`. The archived source includes the modifications required by the experiments. Install this repository's version rather than overwriting it with the upstream version.

```bash
bash scripts/setup_verl.sh
# First install any remaining dependencies in your Linux environment to match the environment snapshot.
python scripts/verify_assets.py
python scripts/run_experiment.py --variant calspo --gpu 0 --base-model models/Qwen2.5-Omni-3B --embedding-model models/all-MiniLM-L6-v2 --output outputs/calspo
```

By default, the runner only prints the resolved arguments. Add `--execute` to start training. Set `--variant` to `ema075` or `ema095` to select the corresponding complete implementation. Each configuration requires its own empty output directory, and automatic resumption is disabled. Settings inherited from the original run manifests include a batch size of 8, four sampled responses per question, a learning rate of 1e-6, LoRA rank 16 / alpha 32, and 38 training steps.

All media paths are relative to the `datasets/` directory containing the Parquet files and follow the format `media/filename`. The verification script follows the actual loader's path-resolution rules to check all 983 media references. These comprise 296 training references, 216 validation references, and 471 references in the 328-item evaluation set. It also verifies the 887 media files and three core code snapshots.

Training and evaluation runs with `--execute` perform this verification before starting. Missing files, empty files, or invalid paths cause execution to stop. The installation script directly installs the modified source included in this repository. It does not initialize submodules or apply external patches. Use `bash scripts/setup_verl.sh --check-only` to check the installation source without installing it.

## Reproducing The Paper Results

[paper/README.md](paper/README.md) describes the complete item-level responses, original metrics, figure source data, and metric recomputation procedure. Each model has 328 complete outputs, aligned by sample identity.

```bash
python -m pip install -r paper/requirements.txt
python paper/recompute_metrics.py
python paper/plot_figures.py
```

Both metric recomputation and plotting can run on a CPU. The figure-checking utilities are included in the repository and do not require a local research-skills directory. CALSPO in the figures corresponds to EquiAffect-3B.

To evaluate a trained checkpoint on the 328-item dataset, run the following command.

```bash
python scripts/run_experiment.py --variant calspo --mode evaluate --gpu 0 --base-model models/Qwen2.5-Omni-3B --embedding-model models/all-MiniLM-L6-v2 --checkpoint outputs/calspo/global_step_38 --output outputs/calspo_eval328 --execute
```

Before execution, check GPU availability and register usage according to the rules of your server. Preparing this archive involved only CPU-based file, data, and argument checks. GPU training and evaluation were not rerun. Historical helper scripts may retain paths from the original server. Use `scripts/run_experiment.py` as the recommended entry point.

## Current Manuscript Evaluation And Statistical Scope

In the current manuscript, EquiAffect denotes the CALSPO model trained on Human Behavior Atlas. Historical directory names containing EquiAffect-3B are retained for compatibility. The new main comparisons comprise four methods in three settings: models trained on EAV and assessed on EAV, and EquiAffect with the corresponding baseline models assessed directly on EAV and AvaMERG without retraining. Retraining examines CALSPO with a new training dataset; direct evaluation examines the existing models on different datasets. Each setting contains 328 paired evaluation records, for 984 records across the three settings. The original Human Behavior Atlas evaluation is excluded from this pooled analysis.

The two EAV assessment collections are different and share 302 source clips. Their archived manifests mark them as `outcome_selected_diagnostic` and `independent_test=false`; the EAV-trained model assessment also records `selection_uses_model_outcomes=true`. Grouping repeated sources does not adjust for outcome-dependent selection. The two EAV collections therefore cannot be compared as though retraining were their only difference. AvaMERG uses records 328–655 of the saved 656-record assessment. Its native ground-truth labels contain no neutral class, so false-neutral rate is inapplicable. Its accuracy results concern a different label space and do not directly establish a reduction in neutral-class bias.

Shared EAV source clips are grouped across both evaluations, producing 682 source groups including the 328 AvaMERG dialogues. A second sensitivity analysis groups EAV records by participant, yielding 349 groups across the three settings. All predictions are retained; repeated sources are handled jointly rather than deleting one evaluation.

For each baseline, a two-sided paired source-group label-swap test exchanges all correctness differences within a group together. An integer dynamic program enumerates the null distribution. Holm correction is applied across the three baseline comparisons. The manuscript table reports these Holm-adjusted p-values: 0.000001011212895 for HARPO, 0.000328016343 for GRPO, and 0.000718480508 for GRPO + AVSPO. These are conditional sensitivity calculations on the saved assessment records. Holm correction addresses the three comparisons, not outcome-dependent sample selection, unidentified dependencies, or training-run uncertainty; these results are not selection-adjusted tests of independent generalization.

The 95% accuracy-gain intervals use 50,000 paired source-group bootstrap replicates. AvaMERG and EAV are separate resampling strata; the two EAV settings share source resampling weights. Each replicate recomputes the equal-weight mean accuracy across the three settings. The intervals are pointwise and not multiplicity-adjusted. Macro-F1 is computed over each setting's native ground-truth classes rather than pooling different label spaces. Invalid outputs remain errors, including in EAV false-neutral denominators when the ground truth is non-neutral.

The three-setting mean accuracy is 28.15% for CALSPO, compared with 23.98% for HARPO, 24.70% for GRPO, and 25.20% for GRPO + AVSPO. These comparisons supplement the task-specific metrics. They do not imply that every task improves: depression classification does not improve over HARPO, and the Human Behavior Atlas text-and-audio subset has a higher invalid-output rate despite improved accuracy.

## Sources And Licenses

Original method project: [OmniSapiens / Human Behavior Atlas](https://github.com/MIT-MI/human_behavior_atlas). VERL's Apache 2.0 license is preserved in `verl/LICENSE` within each code snapshot. Model licenses apply separately. Original datasets and project code remain subject to their respective source terms. This repository does not grant a new blanket license for redistribution or reuse.
