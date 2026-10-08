"""Check completion, held-out coverage, and per-step resource records."""

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "run_manifest.json").read_text())
    summary = json.loads((root / "resource_usage.summary.json").read_text())
    require(manifest["seed"] in (46, 47) and manifest["total_steps"] == 38, "Wrong seed or steps")
    require(summary["return_code"] == 0, "Training process failed")
    step = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    require(step == 38, "Final checkpoint is not step 38")
    require((root / "global_step_38/actor/model_world_size_1_rank_0.pt").is_file(), "Missing actor checkpoint")
    reload_marker = json.loads((root / "checkpoint_reloaded.json").read_text())
    require(reload_marker["step"] == 38, "Final weights were not reloaded")
    final_test = root / "final_test"
    predictions = json.loads((final_test / "predictions_step_38.json").read_text())
    metrics = json.loads((final_test / "metrics_step_38.json").read_text())
    truth = pq.read_table(manifest["validation"], columns=["task", "answer"]).to_pylist()
    require(len(predictions) == len(truth) == metrics["count"] == 64, "Incomplete validation")
    require(
        sorted((p["task"], p["label"]) for p in predictions)
        == sorted((r["task"], r["answer"].strip().lower()) for r in truth),
        "Evaluation rows do not match the fixed validation set",
    )
    for task, result in metrics["per_task"].items():
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "invalid_rate"):
            value = result[key]
            require(math.isfinite(value) and 0 <= value <= 1, f"Invalid {task}/{key}")
    steps = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines() if line]
    require([row["step"] for row in steps] == list(range(1, 39)), "Missing or repeated step")
    require(all(row["responses"] == 32 and row["response_tokens"] > 0 for row in steps), "Wrong rollout counts")
    if manifest["run"] != "original_harpo":
        require(all("xrpo/seen_tasks" in row["metrics"] for row in steps), "Missing algorithm diagnostics")
    require(summary.get("monitor_error") is None, "Resource monitor failed")
    print("run_complete=PASS")


if __name__ == "__main__":
    main()
