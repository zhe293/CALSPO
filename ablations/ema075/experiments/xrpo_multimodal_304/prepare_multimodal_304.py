"""Create a fixed 304-row HBA study subset without changing the shared dataset.

The input HBA parquet stores embedded media bytes. VERL's HBA reader expects
relative file paths, so this script extracts *selected* media into an owned
directory and writes path-based parquet. It refuses to overwrite an output.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


TRAIN_QUOTAS = {
    ("depression_cls", "text"): 80,
    ("emotion_cls", "text_audio"): 152,
    ("emotion_cls", "text_video_audio"): 72,
}
VALIDATION_QUOTAS = {
    ("depression_cls", "text"): 16,
    ("emotion_cls", "text_audio"): 32,
    ("emotion_cls", "text_video_audio"): 16,
}
MEDIA_COLUMNS = ("images", "videos", "audios")
READ_COLUMNS = (
    "problem", "answer", "images", "videos", "audios", "dataset", "task",
    "class_label", "modality_signature",
)


def _extension(kind: str, content: bytes) -> str:
    if kind == "audios" and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return ".wav"
    if kind == "videos" and content[4:8] == b"ftyp":
        return ".mp4"
    if kind == "images" and content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if kind == "images" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    raise ValueError(f"Unrecognized embedded {kind} format")


def _select(root: Path, split: str, quotas: dict, seed: int, forbidden: set[str]) -> dict[Path, list[int]]:
    counts_by_file = {}
    for filename in sorted(glob.glob(str(root / f"{split}-*.parquet"))):
        path = Path(filename)
        table = pq.read_table(path, columns=["task", "modality_signature", "problem"])
        pairs = list(zip(table.column("task").to_pylist(), table.column("modality_signature").to_pylist()))
        hashes = [hashlib.sha256(str(value).encode()).hexdigest() for value in table.column("problem").to_pylist()]
        counts_by_file[path] = (pairs, hashes)
    if not counts_by_file:
        raise FileNotFoundError(f"No {split} shards under {root}")

    selected = defaultdict(list)
    for key, count in quotas.items():
        eligible = []
        for path, (pairs, hashes) in counts_by_file.items():
            possible = {digest for pair, digest in zip(pairs, hashes) if pair == key and digest not in forbidden}
            if len(possible) >= count:
                eligible.append((len(possible), path))
        if not eligible:
            raise ValueError(f"No single {split} shard can supply {count} rows for {key}")
        _, path = sorted(eligible, key=lambda entry: (-entry[0], str(entry[1])))[0]
        pairs, hashes = counts_by_file[path]
        indices = [i for i, pair in enumerate(pairs) if pair == key and hashes[i] not in forbidden]
        rng = random.Random(f"{seed}:{split}:{key[0]}:{key[1]}")
        rng.shuffle(indices)
        unique = []
        seen = set(forbidden)
        for i in indices:
            if hashes[i] not in seen:
                unique.append(i)
                seen.add(hashes[i])
            if len(unique) == count:
                break
        if len(unique) != count:
            raise ValueError(f"Could not obtain {count} distinct {split} prompts for {key}")
        selected[path].extend(unique)
    return {path: sorted(set(indices)) for path, indices in selected.items()}


def _write_split(root: Path, output: Path, split: str, quotas: dict, seed: int, forbidden: set[str]):
    chosen = _select(root, split, quotas, seed, forbidden)
    rows, records = [], []
    for source, indices in sorted(chosen.items()):
        # Only a few selected shards are read with their embedded media columns.
        table = pq.read_table(source, columns=list(READ_COLUMNS))
        for index in indices:
            original = table.slice(index, 1).to_pylist()[0]
            row = {
                "problem": str(original["problem"]),
                "answer": str(original["answer"]),
                "dataset": str(original["dataset"]),
                "task": str(original["task"]),
                "class_label": str(original["class_label"] or original["answer"]),
                "modality_signature": str(original["modality_signature"]),
                "texts": [],
                "ext_video_feats": [],
                "ext_audio_feats": [],
            }
            media_record = []
            for kind in MEDIA_COLUMNS:
                row[kind] = []
                for ordinal, content in enumerate(original[kind] or []):
                    if not isinstance(content, bytes) or not content:
                        raise ValueError(f"Empty/nonbinary media: {source}:{index}:{kind}")
                    digest = hashlib.sha256(content).hexdigest()
                    suffix = _extension(kind, content)
                    relative = Path("media") / split / kind / f"{digest}{suffix}"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(content)
                    elif target.stat().st_size != len(content):
                        raise ValueError(f"Media hash collision or corrupt prior file: {target}")
                    row[kind].append(relative.as_posix())
                    media_record.append({"kind": kind, "sha256": digest, "bytes": len(content)})
            signature = row["modality_signature"]
            expected_audio = signature in ("text_audio", "text_video_audio")
            expected_video = signature == "text_video_audio"
            if bool(row["audios"]) != expected_audio or bool(row["videos"]) != expected_video:
                raise ValueError(f"Media/signature mismatch: {source}:{index}:{signature}")
            rows.append(row)
            records.append({
                "source": str(source), "row": index, "task": row["task"],
                "modality_signature": signature,
                "problem_sha256": hashlib.sha256(row["problem"].encode()).hexdigest(),
                "media": media_record,
            })
    actual = Counter((row["task"], row["modality_signature"]) for row in rows)
    if actual != quotas:
        raise AssertionError(f"Wrong {split} quotas: {actual} != {quotas}")
    pq.write_table(pa.Table.from_pylist(rows), output / f"{split}.parquet", compression="zstd")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("source directory does not exist")
    if args.output.exists():
        parser.error("output directory already exists; refusing to overwrite")
    args.output.mkdir(parents=True)
    manifest = {"seed": args.seed, "source": str(args.source), "quotas": {}, "samples": {}}
    forbidden = set()
    for split, quotas in (("train", TRAIN_QUOTAS), ("validation", VALIDATION_QUOTAS)):
        records = _write_split(args.source, args.output, split, quotas, args.seed, forbidden)
        manifest["samples"][split] = records
        manifest["quotas"][split] = {f"{task}/{sig}": count for (task, sig), count in quotas.items()}
        forbidden.update(record["problem_sha256"] for record in records)
        print(split, len(records), Counter((r["task"], r["modality_signature"]) for r in records), flush=True)
    train_hashes = {r["problem_sha256"] for r in manifest["samples"]["train"]}
    validation_hashes = {r["problem_sha256"] for r in manifest["samples"]["validation"]}
    overlap = train_hashes & validation_hashes
    if overlap:
        raise ValueError(f"Train/validation prompt overlap: {len(overlap)}")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
