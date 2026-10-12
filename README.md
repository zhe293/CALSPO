# CALSPO / EquiAffect-3B — Anonymous Review Materials

CALSPO combines class-frequency-weighted label supervision, virtual-reward advantage recovery, and HARPO modulation. EquiAffect-3B is the trained LoRA adapter for Qwen2.5-Omni-3B.

## Contents

- `verl/`: CALSPO training implementation.
- `ablations/ema075/` and `ablations/ema095/`: no-supervision configurations with task EMA coefficients 0.75 and 0.95.
- `datasets/`: portable training, validation, evaluation data and media.
- `models/EquiAffect-3B/`: LoRA adapter, configuration and loading instructions.
- `paper/results/`: archived original results; `paper/source_data/` contains the six-configuration prediction/metric snapshot used in the current paper.
- `paper/figures/`: manuscript figures.
- `provenance/`: run and dataset provenance with personal path components redacted.

## Reproduction

Install dependencies using `scripts/setup_verl.sh`. Run `python scripts/verify_assets.py` to verify media references and source snapshots; run `python scripts/verify_equiaffect.py` to check adapter tensors. Base-model revision and loading instructions are in `models/`.

The adapter uses Git LFS. Obtain actual weight contents rather than the text pointer. The anonymous mirror's weight-download support must be checked before assuming weights can be downloaded there.

Original dataset and upstream software licenses remain applicable. Personal path redaction changes file digests; manifests describe this anonymized copy. Dataset labels and portable media references are preserved.
