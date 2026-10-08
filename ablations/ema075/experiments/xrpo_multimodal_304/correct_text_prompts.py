"""Clone the selected dataset and fix text-only depression prompts only."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


OLD_SUFFIX = (
    "The above is an audio recording of a participant speaking in a clinical setting. "
    "Based on the speech content, tone, and prosody, what is the speaker's likely\n\n"
    "Based on the speech content, tone, and prosody, classify the speaker’s depression "
    "severity using the PHQ-9 categories: depression, no depression.\n\n?"
)
NEW_SUFFIX = (
    "For this dataset, classify the participant's written transcript using only the "
    "words shown above. Answer with exactly one label: depression or no depression."
)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_dir() or args.output.exists():
        parser.error("Source must exist and output must not exist")
    if args.source.stat().st_uid != os.getuid() or args.output.parent.stat().st_uid != os.getuid():
        parser.error("Source and destination parent must belong to this account")

    # Media are immutable during this experiment, so hard links avoid a second
    # copy. Every changed parquet/manifest is written to a new inode below.
    shutil.copytree(args.source, args.output, copy_function=os.link)
    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    manifest["prompt_correction"] = {
        "source_dataset": str(args.source),
        "scope": "text-only depression_cls rows",
        "old_suffix_sha256": digest(OLD_SUFFIX),
        "new_suffix_sha256": digest(NEW_SUFFIX),
        "counts": {},
    }
    hashes = {}
    for split, expected in (("train", 80), ("validation", 16)):
        source_file = args.source / f"{split}.parquet"
        output_file = args.output / f"{split}.parquet"
        source_table = pq.read_table(source_file)
        rows = source_table.to_pylist()
        changed = 0
        for row, record in zip(rows, manifest["samples"][split]):
            if row["task"] != "depression_cls":
                continue
            if row["modality_signature"] != "text" or any(row[kind] for kind in ("audios", "videos", "images")):
                raise ValueError(f"Depression row unexpectedly includes media: {split}")
            prompt = row["problem"]
            if not prompt.endswith(OLD_SUFFIX):
                raise ValueError(f"Unexpected depression prompt: {split}")
            corrected = prompt[: -len(OLD_SUFFIX)].rstrip() + "\n\n" + NEW_SUFFIX
            row["problem"] = corrected
            record["original_problem_sha256"] = record["problem_sha256"]
            record["problem_sha256"] = digest(corrected)
            changed += 1
        if changed != expected:
            raise AssertionError(f"Expected {expected} corrected rows in {split}, got {changed}")
        # Preserve column types and ordering, but replace the hard-linked
        # parquet path atomically so the source dataset remains byte-for-byte.
        new_table = pa.Table.from_pylist(rows, schema=source_table.schema)
        temporary = args.output / f".{split}.corrected.parquet"
        pq.write_table(new_table, temporary, compression="zstd")
        os.replace(temporary, output_file)
        hashes[split] = {digest(row["problem"]) for row in rows}
        manifest["prompt_correction"]["counts"][split] = changed
        print(split, "rows", len(rows), "corrected", changed)
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Corrected train/validation prompts overlap")
    temporary = args.output / ".manifest.corrected.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output / "manifest.json")
    print("PROMPT_CORRECTION_PASS")


if __name__ == "__main__":
    main()
