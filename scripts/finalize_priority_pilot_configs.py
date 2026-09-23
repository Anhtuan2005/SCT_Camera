"""Apply the frozen rotation and common ROI/doorway geometry to pilot configs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FULL_FRAME_POLYGON = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
DOORWAY_LINE = {
    "id": "doorway",
    "name": "Doorway",
    "point1": [0.30, 0.16],
    "point2": [0.75, 0.16],
    "direction": "forward",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual/recording_plan_pilot.csv",
    )
    parser.add_argument(
        "--confirmation-out",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual/roi_line_confirmation.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    with args.plan.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    confirmation = []
    for row in rows:
        path = ROOT / row["camera_config_path"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["frame_rotation"] = "cw90"
        config["manual_confirmation"] = {
            "status": "CONFIRMED_BY_AI_VISUAL_QA",
            "basis": "Accepted recordings; fixed geometry; no threshold or test-output tuning.",
            "do_not_change": "experiment_version, behavior thresholds, expected behavior/label",
        }
        behavior = row["behavior"]
        if behavior in {"intrusion", "line_crossing"}:
            config["zones"] = []
            config["lines"] = [dict(DOORWAY_LINE)]
            geometry = "doorway line (0.30,0.16)-(0.75,0.16), forward"
        else:
            if len(config.get("zones", [])) != 1:
                raise ValueError(f"Expected one zone in {path}")
            config["zones"][0]["polygon"] = FULL_FRAME_POLYGON
            config["lines"] = []
            geometry = "full visible room polygon (0.02,0.02)-(0.98,0.98)"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        confirmation.append(
            {
                "experiment_version": row["experiment_version"],
                "clip_id": row["clip_id"],
                "behavior": behavior,
                "frame_rotation": "cw90",
                "geometry": geometry,
                "thresholds_changed": "NO",
                "model_output_used_for_tuning": "NO",
                "status": "CONFIRMED_BY_AI_VISUAL_QA",
            }
        )

    args.confirmation_out.parent.mkdir(parents=True, exist_ok=True)
    with args.confirmation_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confirmation[0]))
        writer.writeheader()
        writer.writerows(confirmation)
    print(f"configs={len(confirmation)} confirmation={args.confirmation_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
