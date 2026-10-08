"""Identical held-out classification evaluation for both 304-row study arms."""

import json
import re
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, recall_score


KNOWN_LABELS = {
    "depression_cls": ["no depression", "depression"],
    "emotion_cls": ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
}


def extract(text):
    for pattern in (r"\\boxed\{([^}]*)\}", r"\[(.*?)\]", r"<answer>(.*?)</answer>"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip().lower()
    return str(text).strip().lower()


def evaluate(predictions, ground_truths, tasks, save_path, global_steps):
    if not predictions or len(predictions) != len(ground_truths) or len(tasks) != len(predictions):
        raise ValueError("Evaluation lengths mismatch or empty")
    rows = [
        {"index": i, "task": str(task), "output": str(pred),
         "prediction": extract(pred), "label": str(label).strip().lower()}
        for i, (pred, label, task) in enumerate(zip(predictions, ground_truths, tasks))
    ]
    result = {"step": int(global_steps), "count": len(rows), "per_task": {}}
    flat = {}
    for task in sorted({row["task"] for row in rows}):
        labels = KNOWN_LABELS.get(task)
        if labels is None:
            raise ValueError(f"Task has no fixed label vocabulary: {task}")
        subset = [row for row in rows if row["task"] == task]
        truth = [row["label"] for row in subset]
        if not set(truth).issubset(labels):
            raise ValueError(f"Unknown ground-truth labels for {task}: {sorted(set(truth) - set(labels))}")
        prediction = [row["prediction"] if row["prediction"] in labels else "<invalid>" for row in subset]
        per_label_recall = recall_score(truth, prediction, labels=labels, average=None, zero_division=0)
        values = {
            "accuracy": sum(a == b for a, b in zip(truth, prediction)) / len(subset),
            "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(sum(per_label_recall) / len(labels)),
            "invalid_rate": prediction.count("<invalid>") / len(subset),
        }
        if task == "depression_cls":
            values["positive_recall"] = float(per_label_recall[1])
        result["per_task"][task] = {
            **values, "count": len(subset), "labels": labels + ["<invalid>"],
            "support": {label: truth.count(label) for label in labels},
            "confusion_matrix": confusion_matrix(truth, prediction, labels=labels + ["<invalid>"]).tolist(),
        }
        flat.update({f"common/{task}/{key}": value for key, value in values.items()})
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"predictions_step_{global_steps}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / f"metrics_step_{global_steps}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return flat
