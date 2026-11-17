"""Read the archived sample table and reconstruct run inputs in row order."""
import json
from pathlib import Path

MEDIA_COLUMNS = ("images", "videos", "audios", "ext_video_feats", "ext_audio_feats")

def materialize(root, data_dir, destination, mode):
    import pyarrow as pa
    import pyarrow.parquet as pq

    plan = json.loads((root / "provenance" / "input_rows.json").read_text(encoding="utf-8"))
    all_rows = pq.read_table(data_dir / "samples.parquet")
    schema = all_rows.schema.remove(all_rows.schema.get_field_index("archive_source"))
    schema = schema.remove(schema.get_field_index("archive_row"))
    roles = ("train", "evaluation" if mode == "evaluate" else "validation")
    for role, name in zip(roles, ("train.parquet", "assessment.parquet")):
        start, count = plan[role]
        records = all_rows.slice(start, count)
        if records.num_rows != count or set(records["archive_source"].to_pylist()) != {"HBA"}:
            raise ValueError(f"Unexpected input rows for {role}")
        rows = records.to_pylist()
        for row in rows:
            row.pop("archive_source")
            row.pop("archive_row")
            for column in MEDIA_COLUMNS:
                if row.get(column):
                    row[column] = [str((data_dir / ref).resolve()) for ref in row[column]]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination / name)
