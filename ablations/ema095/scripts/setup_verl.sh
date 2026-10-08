#!/usr/bin/env bash
# Install the bundled, already-patched source snapshot. No submodule or patches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for file in verl/setup.py verl/verl/trainer/ppo/core_algos.py verl/verl/utils/dataset/hba_sft_bam_dataset.py; do
    [[ -f "$ROOT/$file" ]] || { echo "Missing bundled source: $file" >&2; exit 1; }
done
if [[ "${1:-}" == "--check-only" ]]; then
    echo "PASS: bundled VERL source is present; no submodules or patches required."
    exit 0
fi
[[ $# -eq 0 ]] || { echo "Usage: bash scripts/setup_verl.sh [--check-only]" >&2; exit 2; }
"${PYTHON:-python}" -m pip install --no-deps -e "$ROOT/verl"
echo "Installed bundled VERL. Use the archived environment snapshot to prepare Linux/CUDA dependencies."
