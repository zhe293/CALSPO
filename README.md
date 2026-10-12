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

The source adapter uses Git LFS, and the anonymous mirror cannot serve files larger than 8 MB. Download the review ZIP and run `python scripts/restore_review_assets.py` to reconstruct the actual adapter from the 7 MB binary parts in `review_downloads/`; the same command restores the tokenizer and two large figure assets. Then run `python scripts/verify_equiaffect.py`. See `review_downloads/README.md` for details.

Original dataset and upstream software licenses remain applicable. Personal path redaction changes file digests; manifests describe this anonymized copy. Dataset labels and portable media references are preserved.

