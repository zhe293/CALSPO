"""Validate the experiment archive without loading a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from experiment_inputs import MEDIA_COLUMNS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, required=True, help="External archive data directory")
    args = parser.parse_args()
    root = args.root.resolve()
    data = args.data_dir.resolve()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = (data / relative).resolve()
        if not path.is_relative_to(data) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected archive file: {relative}")
    rows = pq.read_table(data / "samples.parquet").to_pylist()
    counts = Counter(row["archive_source"] for row in rows)
    if counts != {"HBA": 848, "EAV": 656, "AvaMERG": 656}:
        raise ValueError(f"Unexpected source counts: {counts}")
    for row in rows:
        if row["archive_source"] != "HBA":
            continue
        for column in MEDIA_COLUMNS:
            for ref in row.get(column) or []:
                path = (data / ref).resolve()
                if not path.is_relative_to(data / "media") or not path.is_file():
                    raise ValueError(f"Missing or nonportable HBA media: {ref}")
    for variant in ("calspo", "ema075", "ema095"):
        code = root if variant == "calspo" else root / "ablations" / variant
        run = json.loads((code / "provenance" / "source_run_manifest.json").read_text(encoding="utf-8"))
        for relative, expected in run["code_sha256"].items():
            if hashlib.sha256((code / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Unexpected code fingerprint: {variant}/{relative}")
    print(json.dumps({"sources": counts, "included_media": manifest["included_media"],
                      "external_eav_media": manifest["external_eav_media"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
