"""Match time-aligned detections to continuous-video ground truth and compute behavior metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


BEHAVIORS = ("intrusion", "loitering", "suspicious_behavior", "theft", "line_crossing")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tolerance-seconds", type=float, default=1.0)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(gt_rows: list[dict[str, str]], detection_rows: list[dict[str, str]], split: str, tolerance: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    selected = [
        row
        for row in gt_rows
        if str(row.get("split", "")).strip().lower() == split.lower()
        and str(row.get("annotation_status", "VERIFIED")).strip().upper() == "VERIFIED"
    ]
    gt_versions = {
        str(row.get("experiment_version", "")).strip()
        for row in selected
        if str(row.get("experiment_version", "")).strip()
    }
    detection_versions = {
        str(row.get("experiment_version", "")).strip()
        for row in detection_rows
        if str(row.get("experiment_version", "")).strip()
    }
    if len(gt_versions) > 1 or len(detection_versions) > 1:
        raise ValueError("Input rows mix multiple experiment versions")
    if gt_versions and detection_versions and gt_versions != detection_versions:
        raise ValueError(
            f"Ground truth version {sorted(gt_versions)} does not match "
            f"detections {sorted(detection_versions)}"
        )
    experiment_version = next(iter(gt_versions or detection_versions), "UNVERSIONED")
    for behavior in BEHAVIORS:
        behavior_gt = [row for row in selected if row.get("behavior", "").strip() == behavior]
        positives = [row for row in behavior_gt if row.get("label", "").strip().lower() == "positive"]
        negative_clips = {row["clip_id"] for row in behavior_gt if row.get("label", "").strip().lower() == "negative"}
        evaluated_clips = {row["clip_id"] for row in behavior_gt}
        detections = sorted(
            [row for row in detection_rows if row.get("behavior", "").strip() == behavior and row.get("clip_id") in evaluated_clips],
            key=lambda row: (row["clip_id"], float(row["video_time_seconds"])),
        )
        used: set[int] = set()
        latencies: list[float] = []
        tp = 0
        for event in positives:
            start = float(event["start_seconds"])
            eligible = float(event.get("eligible_time_seconds") or start)
            end = float(event["end_seconds"])
            expected_direction = str(event.get("expected_direction", "")).strip().upper()
            candidate = next((
                (index, row) for index, row in enumerate(detections)
                if index not in used
                and row["clip_id"] == event["clip_id"]
                and (
                    expected_direction not in {"IN", "OUT"}
                    or str(row.get("direction", "")).strip().upper() == expected_direction
                )
                and eligible <= float(row["video_time_seconds"]) <= end + tolerance
            ), None)
            if candidate is None:
                match_rows.append({"experiment_version": experiment_version, "behavior": behavior, "clip_id": event["clip_id"], "event_id": event.get("event_id", ""), "behavior_event_id": "", "instrumentation_event_id": "", "expected_direction": expected_direction, "detected_direction": "", "status": "FN", "detection_time_seconds": "", "latency_seconds": ""})
                continue
            index, detection = candidate
            used.add(index)
            latency = max(0.0, float(detection["video_time_seconds"]) - eligible)
            latencies.append(latency)
            tp += 1
            match_rows.append({"experiment_version": experiment_version, "behavior": behavior, "clip_id": event["clip_id"], "event_id": event.get("event_id", ""), "behavior_event_id": detection.get("behavior_event_id", ""), "instrumentation_event_id": detection.get("instrumentation_event_id", ""), "expected_direction": expected_direction, "detected_direction": detection.get("direction", ""), "status": "TP", "detection_time_seconds": detection["video_time_seconds"], "latency_seconds": round(latency, 3)})
        fp_indices = [index for index in range(len(detections)) if index not in used]
        for index in fp_indices:
            detection = detections[index]
            match_rows.append({"experiment_version": experiment_version, "behavior": behavior, "clip_id": detection["clip_id"], "event_id": "", "behavior_event_id": detection.get("behavior_event_id", ""), "instrumentation_event_id": detection.get("instrumentation_event_id", ""), "expected_direction": "", "detected_direction": detection.get("direction", ""), "status": "FP", "detection_time_seconds": detection["video_time_seconds"], "latency_seconds": ""})
        fp = len(fp_indices)
        fn = len(positives) - tp
        negative_alarm_clips = {detections[index]["clip_id"] for index in fp_indices if detections[index]["clip_id"] in negative_clips}
        tn = len(negative_clips - negative_alarm_clips)
        ready = bool(positives and negative_clips)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics_rows.append({
            "experiment_version": experiment_version,
            "behavior": behavior,
            "status": "completed" if ready else "NEEDS MANUAL INPUT",
            "positive_events": len(positives),
            "negative_clips": len(negative_clips),
            "tp": tp if ready else "",
            "fp": fp if ready else "",
            "tn": tn if ready else "",
            "fn": fn if ready else "",
            "detection_rate": round(recall, 6) if ready else "",
            "false_alarm_rate_negative_clip": round(len(negative_alarm_clips) / len(negative_clips), 6) if ready else "",
            "precision": round(precision, 6) if ready else "",
            "recall": round(recall, 6) if ready else "",
            "f1": round(f1, 6) if ready else "",
            "latency_median_seconds": round(median(latencies), 3) if ready and latencies else "",
            "latency_p95_seconds": round(float(_percentile(latencies, 0.95)), 3) if ready and latencies else "",
        })
    return metrics_rows, match_rows


def main() -> int:
    args = _args()
    metrics, matches = evaluate(_rows(args.ground_truth), _rows(args.detections), args.split, args.tolerance_seconds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "behavior_metrics.csv", metrics)
    _write(args.out_dir / "behavior_event_matches.csv", matches)
    (args.out_dir / "behavior_evaluation.json").write_text(json.dumps({"split": args.split, "tolerance_seconds": args.tolerance_seconds, "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"wrote={args.out_dir}")
    return 0 if all(row["status"] == "completed" for row in metrics) else 2


if __name__ == "__main__":
    raise SystemExit(main())
