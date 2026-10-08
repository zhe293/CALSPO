"""Inspect HBA metadata and build deterministic, text-only Parquet subsets.

Only lightweight metadata columns are read from the shared source. The output
contains empty string-list media columns, so the HBA loader will not read the
embedded audio/video/image payloads during the text-only experiment.
"""

from __future__ import annotations

import argparse
import collections
import glob
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


LIGHT_COLUMNS = [
    "problem",
    "answer",
    "dataset",
    "modality_signature",
    "task",
    "class_label",
    "texts",
    "ext_video_feats",
    "ext_audio_feats",
]


def iter_light_rows(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files match: {pattern}")
    for path in files:
        schema = pq.read_schema(path)
        columns = [name for name in LIGHT_COLUMNS if name in schema.names]
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            yield row


def signature_counts(pattern: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files match: {pattern}")
    for path in files:
        schema = pq.read_schema(path)
        if "modality_signature" not in schema.names:
            counts["<missing>"] += pq.ParquetFile(path).metadata.num_rows
            continue
        column = pq.read_table(path, columns=["modality_signature"]).column("modality_signature")
        counts.update(str(value.as_py() or "<missing>") for value in column)
    return counts


def normalized_row(row: dict) -> dict:
    return {
        "problem": str(row.get("problem") or ""),
        "answer": str(row.get("answer") or ""),
        "images": [],
        "videos": [],
        "audios": [],
        "dataset": str(row.get("dataset") or ""),
        "task": str(row.get("task") or ""),
        "class_label": str(row.get("class_label") or ""),
        "texts": [str(x) for x in (row.get("texts") or [])],
        "modality_signature": str(row.get("modality_signature") or ""),
        "ext_video_feats": [],
        "ext_audio_feats": [],
    }


def balanced_reservoir(
    pattern: str,
    allowed_signatures: set[str],
    per_task: int,
    seed: int,
) -> tuple[list[dict], collections.Counter[str]]:
    rng = random.Random(seed)
    seen: collections.Counter[str] = collections.Counter()
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files match: {pattern}")
    for path in files:
        schema = pq.read_schema(path)
        if "modality_signature" not in schema.names:
            continue
        signatures = pq.read_table(path, columns=["modality_signature"]).column("modality_signature")
        selected_indices = [
            index
            for index, value in enumerate(signatures)
            if str(value.as_py() or "") in allowed_signatures
        ]
        if not selected_indices:
            continue
        columns = [name for name in LIGHT_COLUMNS if name in schema.names]
        table = pq.read_table(path, columns=columns)
        for index in selected_indices:
            row = {name: table.column(name)[index].as_py() for name in table.column_names}
            task = str(row.get("task") or row.get("dataset") or "unknown")
            seen[task] += 1
            item = normalized_row(row)
            bucket = buckets[task]
            if len(bucket) < per_task:
                bucket.append(item)
            else:
                replacement = rng.randrange(seen[task])
                if replacement < per_task:
                    bucket[replacement] = item
    rows = [item for task in sorted(buckets) for item in buckets[task]]
    rng.shuffle(rows)
    return rows, seen


def write_rows(rows: list[dict], output: Path) -> None:
    if not rows:
        raise RuntimeError("No rows matched the requested signatures")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Glob such as /share/.../train-*.parquet")
    parser.add_argument("--inspect", action="store_true", help="Print modality counts and write nothing")
    parser.add_argument("--allowed-signatures", help="Exact comma-separated signatures selected after inspection")
    parser.add_argument("--per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.inspect:
        for signature, count in signature_counts(args.source).most_common():
            print(f"{count}\t{signature}")
        return

    if not args.allowed_signatures or args.output is None:
        parser.error("extraction requires --allowed-signatures and --output")
    allowed = {value.strip() for value in args.allowed_signatures.split(",") if value.strip()}
    rows, seen = balanced_reservoir(args.source, allowed, args.per_task, args.seed)
    write_rows(rows, args.output)
    print(f"wrote_rows={len(rows)} output={args.output}")
    for task in sorted(seen):
        print(f"task={task} eligible={seen[task]} selected={min(seen[task], args.per_task)}")


if __name__ == "__main__":
    main()
