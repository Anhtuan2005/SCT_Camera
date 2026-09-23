"""Generate the manual recording plan and blank ground-truth template."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCENARIOS = {
    "intrusion": {
        "positive": (25, "A person crosses the configured line in the IN direction once."),
        "negative": (25, "A person approaches the line, stops/turns before it, or crosses only OUT."),
    },
    "loitering": {
        "positive": (35, "A person stays inside the loitering ROI for at least 25 s (configured threshold 20 s)."),
        "negative": (25, "A person passes through the ROI and leaves before 15 s."),
    },
    "suspicious_behavior": {
        "positive": (200, "A confirmed stranger stands nearly still or paces inside stranger_watch for at least 190 s."),
        "negative": (200, "A stranger moves purposefully through the watched area without standing still or pacing."),
    },
    "theft": {
        "positive": (35, "A confirmed stranger stays near a watched vehicle for at least 10 s and then moves/pushes it."),
        "negative": (35, "A stranger stays near the vehicle but does not move it, follow its motion, or touch/push it."),
    },
    "line_crossing": {
        "positive": (25, "A person crosses the configured counting line exactly once in the expected direction."),
        "negative": (25, "A person walks parallel to, touches, or turns before the counting line without crossing it."),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for behavior, scenarios in SCENARIOS.items():
        for label, (duration, action) in scenarios.items():
            for participant_number in range(1, 6):
                participant = f"P{participant_number:02d}"
                split = "train" if participant_number <= 3 else "validation" if participant_number == 4 else "test"
                for lighting in ("daylight", "low_light"):
                    for viewpoint in ("frontal_oblique", "side_oblique"):
                        clip_id = f"{behavior}_{label}_{participant}_{lighting}_{viewpoint}"
                        rows.append({
                            "clip_id": clip_id,
                            "behavior": behavior,
                            "label": label,
                            "participant_id": participant,
                            "lighting": lighting,
                            "viewpoint": viewpoint,
                            "split": split,
                            "group_id": participant,
                            "target_duration_seconds": duration,
                            "action_script": action,
                            "camera_setup": "fixed 2.5-3 m high; 20-35 degree downward angle; full body/ROI visible; 1280x720 or higher; 25-30 FPS",
                            "video_path": f"experiments/priority_19_08/manual/videos/{behavior}/{split}/{clip_id}.mp4",
                            "camera_config_path": f"experiments/priority_19_08/manual/configs/{clip_id}.yaml",
                            "status": "TO_RECORD",
                        })
    fields = list(rows[0])
    with (args.out_dir / "recording_plan.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "behavior_ground_truth_template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        gt_fields = ["clip_id", "behavior", "event_id", "start_seconds", "eligible_time_seconds", "end_seconds", "label", "split", "group_id", "annotator", "reviewer", "annotation_status", "notes"]
        writer = csv.DictWriter(handle, fieldnames=gt_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "clip_id": row["clip_id"],
                "behavior": row["behavior"],
                "event_id": f"gt_{row['clip_id']}_01" if row["label"] == "positive" else "",
                "label": row["label"],
                "split": row["split"],
                "group_id": row["group_id"],
                "annotation_status": "NEEDS_MANUAL_INPUT",
            })
    print(f"clips={len(rows)} wrote={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
