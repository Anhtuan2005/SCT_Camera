"""Train a lightweight behavior risk classifier from logged JSONL events.

Examples:
  .venv\\Scripts\\python.exe scripts\\train_behavior_classifier.py
  .venv\\Scripts\\python.exe scripts\\train_behavior_classifier.py --labels data\\behavior_labels.csv

Label CSV format:
  event_id,label,notes
  8f3...,false_alarm,person walked past gate
  2a1...,theft_attempt,staged bicycle theft
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analytics.behavior_learning import FEATURE_NAMES

DEFAULT_POSITIVE_LABELS = {
    "positive",
    "true_positive",
    "suspicious",
    "theft",
    "theft_attempt",
    "asset_removed",
    "asset_missing",
    "loitering_bad",
}
DEFAULT_NEGATIVE_LABELS = {
    "negative",
    "false_positive",
    "false_alarm",
    "normal",
    "benign",
    "ignore",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SCT behavior classifier")
    parser.add_argument("--events", default="data/behavior_events.jsonl", help="Logged behavior events JSONL")
    parser.add_argument("--labels", default="", help="Optional CSV/JSONL labels keyed by event_id")
    parser.add_argument("--out", default="models/behavior_classifier.npz", help="Output model path")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()

    events_path = Path(args.events)
    labels_path = Path(args.labels) if args.labels else None
    records = load_events(events_path)
    labels = load_external_labels(labels_path) if labels_path else {}
    dataset = build_dataset(records, labels)
    if len(dataset) < 4:
        raise SystemExit("Need at least 4 labeled events. Add labels to JSONL or pass --labels CSV.")

    x = np.array([[row["features"].get(name, 0.0) for name in FEATURE_NAMES] for row in dataset], dtype=float)
    y = np.array([row["target"] for row in dataset], dtype=float)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need both positive and negative labels before training.")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_norm = (x - mean) / scale

    weights, bias, metrics = train_logistic_regression(
        x_norm,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        feature_names=np.array(FEATURE_NAMES),
        weights=weights,
        bias=np.array(bias),
        mean=mean,
        scale=scale,
        metrics=json.dumps(metrics, sort_keys=True),
    )
    print(f"trained={len(dataset)} positives={int(y.sum())} negatives={int(len(y) - y.sum())}")
    print(f"accuracy={metrics['accuracy']:.3f} loss={metrics['loss']:.4f}")
    print(f"saved={out_path}")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Events file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def load_external_labels(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Labels file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        labels: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                labels[str(item["event_id"])] = {
                    "label": str(item.get("label", "")),
                    "notes": str(item.get("notes", item.get("label_notes", ""))),
                }
        return labels
    labels = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = str(row.get("event_id", "")).strip()
            if not event_id:
                continue
            labels[event_id] = {
                "label": str(row.get("label", "")).strip(),
                "notes": str(row.get("notes", row.get("label_notes", ""))).strip(),
            }
    return labels


def build_dataset(
    records: list[dict[str, Any]],
    external_labels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for record in records:
        event_id = str(record.get("event_id", ""))
        label = str(record.get("label") or "").strip()
        if event_id in external_labels:
            label = external_labels[event_id]["label"]
        target = label_to_target(label)
        if target is None:
            continue
        features = record.get("features")
        if not isinstance(features, dict):
            continue
        dataset.append({"event_id": event_id, "features": features, "target": target, "label": label})
    return dataset


def label_to_target(label: str) -> int | None:
    normalized = label.strip().lower()
    if normalized in DEFAULT_POSITIVE_LABELS:
        return 1
    if normalized in DEFAULT_NEGATIVE_LABELS:
        return 0
    return None


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    for _ in range(max(1, epochs)):
        logits = x @ weights + bias
        predictions = sigmoid(logits)
        error = predictions - y
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    predictions = sigmoid(x @ weights + bias)
    loss = float(
        -np.mean(y * np.log(predictions + 1e-9) + (1.0 - y) * np.log(1.0 - predictions + 1e-9))
        + 0.5 * l2 * np.sum(weights * weights)
    )
    accuracy = float(np.mean((predictions >= 0.5) == y))
    return weights, bias, {"loss": loss, "accuracy": accuracy}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.array([_sigmoid(float(value)) for value in values], dtype=float)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


if __name__ == "__main__":
    main()
