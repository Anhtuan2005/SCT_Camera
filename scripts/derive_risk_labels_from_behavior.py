"""Derive risk positive/negative labels from matched behavior ground truth."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.detections.open("r", encoding="utf-8-sig", newline="") as handle:
        detections = {
            row.get("behavior_event_id", ""): row
            for row in csv.DictReader(handle)
            if row.get("behavior_event_id")
        }
    with args.matches.open("r", encoding="utf-8-sig", newline="") as handle:
        statuses = {
            row.get("behavior_event_id", ""): row.get("status", "")
            for row in csv.DictReader(handle)
            if row.get("behavior_event_id") and row.get("status") in {"TP", "FP"}
        }
    rows = [
        {
            "event_id": event_id,
            "label": "positive" if statuses[event_id] == "TP" else "negative",
            "group_id": detection.get("group_id", ""),
            "notes": f"derived from continuous-video ground truth: {statuses[event_id]}",
        }
        for event_id, detection in detections.items()
        if event_id in statuses
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "label", "group_id", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"labels={len(rows)} wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
