"""Plot paper-ready behavior outcomes without mixing event and clip units."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--caption",
        default=(
            "Single-participant pilot; one positive and one negative frozen-test clip "
            "per behavior. No aggregate score."
        ),
    )
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: str | None) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


def build_outcome_rows(
    metrics_rows: list[dict[str, str]],
    match_rows: list[dict[str, str]],
    ground_truth_rows: list[dict[str, str]],
    split: str,
) -> list[dict[str, Any]]:
    selected_gt = [
        row
        for row in ground_truth_rows
        if str(row.get("split", "")).strip().lower() == split.lower()
        and str(row.get("annotation_status", "VERIFIED")).strip().upper() == "VERIFIED"
    ]
    outcomes: list[dict[str, Any]] = []
    for metric in metrics_rows:
        behavior = str(metric["behavior"]).strip()
        behavior_gt = [row for row in selected_gt if str(row.get("behavior", "")).strip() == behavior]
        positive_clips = {
            row["clip_id"] for row in behavior_gt if str(row.get("label", "")).strip().lower() == "positive"
        }
        negative_clips = {
            row["clip_id"] for row in behavior_gt if str(row.get("label", "")).strip().lower() == "negative"
        }
        behavior_matches = [row for row in match_rows if str(row.get("behavior", "")).strip() == behavior]
        positive_fp = sum(
            row.get("status") == "FP" and row.get("clip_id") in positive_clips
            for row in behavior_matches
        )
        negative_fp_rows = [
            row
            for row in behavior_matches
            if row.get("status") == "FP" and row.get("clip_id") in negative_clips
        ]
        negative_alarm_clips = {row["clip_id"] for row in negative_fp_rows}
        accounted_fp = positive_fp + len(negative_fp_rows)
        if accounted_fp != int(metric["fp"]):
            raise ValueError(
                f"{behavior}: metrics fp={metric['fp']} but match rows account for {accounted_fp}"
            )
        outcomes.append({
            "behavior": behavior,
            "matched_tp_events": int(metric["tp"]),
            "unmatched_positive_clip_fp_events": positive_fp,
            "negative_clip_fp_events": len(negative_fp_rows),
            "fn_events": int(metric["fn"]),
            "negative_clips": len(negative_clips),
            "negative_clips_with_false_alert": len(negative_alarm_clips),
            "negative_clip_far": float(metric["false_alarm_rate_negative_clip"]),
            "precision": float(metric["precision"]),
            "recall": float(metric["recall"]),
            "f1": float(metric["f1"]),
            "latency_median_seconds": _float(metric.get("latency_median_seconds")),
        })
    return outcomes


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]], caption: str) -> None:
    labels = [row["behavior"].replace("suspicious_behavior", "suspicious\nbehavior").replace("line_crossing", "line\ncrossing") for row in rows]
    x = np.arange(len(rows))
    width = 0.19
    series = [
        ("Matched TP events", "matched_tp_events", "#2F6B9A"),
        ("Unmatched events on positive clip", "unmatched_positive_clip_fp_events", "#D98C10"),
        ("False-alert events on negative clip", "negative_clip_fp_events", "#B33A3A"),
        ("FN events", "fn_events", "#6F2A3A"),
    ]

    figure, (event_axis, negative_axis) = plt.subplots(
        1,
        2,
        figsize=(8.2, 4.0),
        gridspec_kw={"width_ratios": [4.8, 1.15]},
    )
    for index, (legend, key, color) in enumerate(series):
        values = [int(row[key]) for row in rows]
        bars = event_axis.bar(x + (index - 1.5) * width, values, width, label=legend, color=color)
        event_axis.bar_label(bars, labels=[str(value) for value in values], padding=2, fontsize=8)

    event_axis.set_xticks(x, labels)
    event_axis.set_ylabel("Number of events")
    event_axis.set_ylim(0, max(6, max(int(row[key]) for row in rows for _, key, _ in series) + 1))
    event_axis.set_yticks(range(0, int(event_axis.get_ylim()[1]) + 1))
    event_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    event_axis.set_axisbelow(True)
    event_axis.spines[["top", "right"]].set_visible(False)
    event_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False, fontsize=8)

    alarm_values = np.array([[int(row["negative_clips_with_false_alert"] > 0)] for row in rows])
    negative_axis.imshow(alarm_values, cmap=ListedColormap(["#DCE9E2", "#F3D6D4"]), vmin=0, vmax=1, aspect="auto")
    negative_axis.set_xticks([0], ["Negative clip"])
    negative_axis.set_yticks(np.arange(len(rows)), labels)
    negative_axis.tick_params(axis="y", length=0)
    for index, row in enumerate(rows):
        alarm = int(row["negative_clips_with_false_alert"])
        label = "FALSE ALERT" if alarm else "CLEAN"
        negative_axis.text(0, index, f"{label}\nFAR={row['negative_clip_far']:.1f}", ha="center", va="center", fontsize=8, fontweight="bold")
    for spine in negative_axis.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.7)

    figure.text(
        0.5,
        0.01,
        caption,
        ha="center",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.31, wspace=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _args()
    rows = build_outcome_rows(
        _read(args.metrics),
        _read(args.matches),
        _read(args.ground_truth),
        args.split,
    )
    if not rows:
        raise ValueError("No behavior outcome rows")
    _write_table(args.table_out, rows)
    _plot(args.output, rows, args.caption)
    print(f"wrote={args.output} table={args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
