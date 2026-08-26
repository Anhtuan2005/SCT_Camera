"""Evaluate the behavior-risk logistic model on a deterministic train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_learning import FEATURE_NAMES
from scripts.train_behavior_classifier import (
    build_dataset,
    load_events,
    load_external_labels,
    train_logistic_regression,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data/behavior_events.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/behavior_labels.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser.parse_args()


def stratified_split(
    targets: np.ndarray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than 1")
    rng = np.random.default_rng(seed)
    buckets = {"train": [], "validation": [], "test": []}
    for target in (0, 1):
        indices = np.flatnonzero(targets == target)
        rng.shuffle(indices)
        if len(indices) < 5:
            raise ValueError(f"Need at least 5 samples for class {target}; found {len(indices)}")
        test_count = max(1, int(round(len(indices) * test_fraction)))
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        if test_count + validation_count >= len(indices):
            raise ValueError(f"Not enough class {target} samples for three non-empty splits")
        buckets["test"].extend(indices[:test_count])
        buckets["validation"].extend(indices[test_count:test_count + validation_count])
        buckets["train"].extend(indices[test_count + validation_count:])
    return {name: np.asarray(sorted(values), dtype=int) for name, values in buckets.items()}


def stratified_group_split(
    targets: np.ndarray,
    group_ids: list[str],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Keep every source/scene/person group in exactly one partition."""
    if any(not group_id for group_id in group_ids):
        raise ValueError("Every labeled event needs group_id before leakage-safe splitting")
    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    if len(grouped) < 3:
        raise ValueError(f"Need at least 3 independent group_id values; found {len(grouped)}")

    fractions = {
        "train": 1.0 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    target_counts = {
        split: np.asarray([len(targets), np.sum(targets == 0), np.sum(targets == 1)], dtype=float) * fraction
        for split, fraction in fractions.items()
    }
    group_rows = [
        (group_id, indices, np.asarray([
            len(indices),
            np.sum(targets[indices] == 0),
            np.sum(targets[indices] == 1),
        ], dtype=float))
        for group_id, indices in grouped.items()
    ]
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, list[int]]] | None = None
    for _ in range(2000):
        rng.shuffle(group_rows)
        assignments = {split: [] for split in fractions}
        counts = {split: np.zeros(3, dtype=float) for split in fractions}
        for _, indices, group_count in group_rows:
            split = min(
                fractions,
                key=lambda name: float(np.sum(((counts[name] + group_count - target_counts[name]) / np.maximum(target_counts[name], 1.0)) ** 2)),
            )
            assignments[split].extend(indices)
            counts[split] += group_count
        if any(not values for values in assignments.values()):
            continue
        if any(set(targets[values]) != {0.0, 1.0} for values in assignments.values()):
            continue
        score = sum(
            float(np.sum(((counts[name] - target_counts[name]) / np.maximum(target_counts[name], 1.0)) ** 2))
            for name in fractions
        )
        if best is None or score < best[0]:
            best = score, assignments
    if best is None:
        raise ValueError("Could not create three group-disjoint splits containing both classes")
    return {name: np.asarray(sorted(values), dtype=int) for name, values in best[1].items()}


def binary_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    actual = targets.astype(bool)
    tp = int(np.sum(predictions & actual))
    fp = int(np.sum(predictions & ~actual))
    tn = int(np.sum(~predictions & ~actual))
    fn = int(np.sum(~predictions & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round((tp + tn) / max(len(targets), 1), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": round(fp / (fp + tn), 6) if fp + tn else 0.0,
    }


def curve_points(targets: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    thresholds = [float("inf"), *sorted(set(float(value) for value in probabilities), reverse=True), float("-inf")]
    points: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = binary_metrics(targets, probabilities, threshold)
        points.append(
            {
                "threshold": threshold,
                "tpr": float(metrics["recall"]),
                "fpr": float(metrics["fpr"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
            }
        )
    return points


def _auc(points: list[dict[str, float]]) -> float:
    ordered = sorted(points, key=lambda item: (item["fpr"], item["tpr"]))
    return round(float(np.trapz([item["tpr"] for item in ordered], [item["fpr"] for item in ordered])), 6)


def _select_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict[str, float | int]]]:
    candidates = sorted({0.0, 0.5, 1.0, *[float(value) for value in probabilities]})
    sensitivity = [binary_metrics(targets, probabilities, threshold) for threshold in candidates]
    best = max(sensitivity, key=lambda item: (float(item["f1"]), float(item["recall"]), -abs(float(item["threshold"]) - 0.5)))
    return float(best["threshold"]), sensitivity


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_svg(path: Path, metrics: dict[str, Any]) -> None:
    cells = [
        ("TN", int(metrics["tn"]), 120, 110),
        ("FP", int(metrics["fp"]), 320, 110),
        ("FN", int(metrics["fn"]), 120, 310),
        ("TP", int(metrics["tp"]), 320, 310),
    ]
    maximum = max((value for _, value, _, _ in cells), default=1) or 1
    rects = []
    for label, value, x, y in cells:
        shade = 245 - round(150 * value / maximum)
        rects.append(
            f'<rect x="{x}" y="{y}" width="190" height="190" fill="rgb({shade},{shade},255)" stroke="#333"/>'
            f'<text x="{x + 95}" y="{y + 85}" text-anchor="middle" font-size="24">{label}</text>'
            f'<text x="{x + 95}" y="{y + 125}" text-anchor="middle" font-size="30" font-weight="bold">{value}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="560">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="320" y="45" text-anchor="middle" font-size="28" font-weight="bold">Risk confusion matrix (test)</text>'
        '<text x="310" y="535" text-anchor="middle" font-size="20">Predicted class</text>'
        '<text x="25" y="300" text-anchor="middle" font-size="20" transform="rotate(-90 25 300)">Actual class</text>'
        + ''.join(rects) + '</svg>',
        encoding="utf-8",
    )


def _write_roc_svg(path: Path, points: list[dict[str, float]], auc: float) -> None:
    ordered = sorted(points, key=lambda item: (item["fpr"], item["tpr"]))
    polyline = ' '.join(f'{70 + item["fpr"] * 500:.1f},{540 - item["tpr"] * 460:.1f}' for item in ordered)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="620">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="320" y="40" text-anchor="middle" font-size="28" font-weight="bold">Risk ROC curve (AUC={auc:.3f})</text>'
        '<line x1="70" y1="540" x2="570" y2="540" stroke="#222"/><line x1="70" y1="540" x2="70" y2="80" stroke="#222"/>'
        '<line x1="70" y1="540" x2="570" y2="80" stroke="#999" stroke-dasharray="8 8"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#1565c0" stroke-width="4"/>'
        '<text x="320" y="590" text-anchor="middle" font-size="20">False positive rate</text>'
        '<text x="22" y="310" text-anchor="middle" font-size="20" transform="rotate(-90 22 310)">True positive rate</text>'
        '</svg>',
        encoding="utf-8",
    )


def _blocked(out_dir: Path, dataset: list[dict[str, Any]], reason: str) -> int:
    positives = sum(int(row["target"]) for row in dataset)
    result = {
        "status": "blocked",
        "reason": reason,
        "labeled_samples": len(dataset),
        "positive_samples": positives,
        "negative_samples": len(dataset) - positives,
        "required_minimum": "at least 5 positive and 5 negative events for a three-way split",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_rows(out_dir / "risk_metrics.csv", [{
        "status": "NEEDS MANUAL INPUT",
        "labeled_samples": len(dataset),
        "positive_samples": positives,
        "negative_samples": len(dataset) - positives,
        "accuracy": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "roc_auc": "",
        "threshold": "",
        "blocker": reason,
    }])
    print(json.dumps(result))
    return 2


def main() -> int:
    args = _args()
    records = load_events(args.events)
    labels = load_external_labels(args.labels)
    dataset = build_dataset(records, labels)
    positives = sum(int(row["target"]) for row in dataset)
    negatives = len(dataset) - positives
    if positives < 5 or negatives < 5:
        return _blocked(args.out_dir, dataset, "Insufficient labeled events; metrics would not be scientifically valid")
    event_ids = [str(row["event_id"]) for row in dataset]
    if len(event_ids) != len(set(event_ids)):
        return _blocked(args.out_dir, dataset, "Duplicate event_id values would leak across splits")

    x = np.asarray([[row["features"].get(name, 0.0) for name in FEATURE_NAMES] for row in dataset], dtype=float)
    y = np.asarray([row["target"] for row in dataset], dtype=float)
    try:
        split = stratified_group_split(
            y,
            [str(row.get("group_id", "")) for row in dataset],
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    except ValueError as exc:
        return _blocked(args.out_dir, dataset, str(exc))
    train_x = x[split["train"]]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    weights, bias, training_metrics = train_logistic_regression(
        normalized[split["train"]],
        y[split["train"]],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    probabilities = 1.0 / (1.0 + np.exp(-(normalized @ weights + bias)))
    threshold, sensitivity = _select_threshold(y[split["validation"]], probabilities[split["validation"]])
    test_y = y[split["test"]]
    test_probabilities = probabilities[split["test"]]
    test_metrics = binary_metrics(test_y, test_probabilities, threshold)
    roc = curve_points(test_y, test_probabilities)
    test_metrics["roc_auc"] = _auc(roc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    split_by_index = {int(index): name for name, indices in split.items() for index in indices}
    for index, row in enumerate(dataset):
        predictions.append(
            {
                "event_id": row["event_id"],
                "split": split_by_index[index],
                "target": row["target"],
                "probability": round(float(probabilities[index]), 8),
                "prediction": int(probabilities[index] >= threshold),
            }
        )
    _write_rows(args.out_dir / "predictions.csv", predictions)
    _write_rows(args.out_dir / "threshold_sensitivity.csv", sensitivity)
    _write_rows(args.out_dir / "roc_curve.csv", roc)
    _write_rows(args.out_dir / "confusion_matrix.csv", [
        {"actual": 0, "predicted_0": test_metrics["tn"], "predicted_1": test_metrics["fp"]},
        {"actual": 1, "predicted_0": test_metrics["fn"], "predicted_1": test_metrics["tp"]},
    ])
    _write_confusion_svg(args.out_dir / "confusion_matrix.svg", test_metrics)
    _write_roc_svg(args.out_dir / "roc_curve.svg", roc, float(test_metrics["roc_auc"]))
    np.savez(
        args.out_dir / "behavior_classifier.npz",
        feature_names=np.asarray(FEATURE_NAMES),
        weights=weights,
        bias=np.asarray(bias),
        mean=mean,
        scale=scale,
        metrics=json.dumps(test_metrics, sort_keys=True),
    )
    result = {
        "status": "completed",
        "protocol": {
            "seed": args.seed,
            "split_unit": "group_id (source clip/scene/participant)",
            "train_samples": len(split["train"]),
            "validation_samples": len(split["validation"]),
            "test_samples": len(split["test"]),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "threshold_selection": "maximum validation F1, then recall, then nearest 0.5",
            "leakage_check": "duplicate event_id rejected and group_id kept wholly within one split",
        },
        "class_distribution": {"positive": positives, "negative": negatives},
        "training": training_metrics,
        "test": test_metrics,
    }
    (args.out_dir / "evaluation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
