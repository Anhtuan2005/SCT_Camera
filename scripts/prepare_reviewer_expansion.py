from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("E:/")
BASE = ROOT / "experiments" / "reviewer_expansion_20260908"
VIDEO_ROOT = BASE / "videos_p02"
CONFIG_ROOT = BASE / "configs_p02"
P01_MANIFEST = ROOT / "experiments" / "priority_19_08" / "manual" / "pilot_manifest.csv"
P01_CONFIG_ROOT = ROOT / "experiments" / "priority_19_08" / "manual" / "configs_pilot"
VERSION = "priority-pilot-v3-two-participant-2026-09-08"


# One P02 recording for every scenario in the frozen P01 protocol.  Entries are
# (zip/file path, member name or None).  The rejected takes are intentionally
# absent from this mapping.
SOURCES: dict[str, tuple[str, str | None]] = {
    "P02_intrusion_positive_r1_daylight_frontal_oblique": (
        "Intrusion_positive.zip", "38956703589233412001.mp4"
    ),
    "P02_intrusion_negative_r1_low_light_side_oblique": (
        "Intrusion_lowlight.zip",
        "20260908164028412_F6B80AMPBVCDD6C_L_0_L0120908164028_1.mp4",
    ),
    "P02_loitering_positive_r1_daylight_side_oblique": (
        "Loitering_light.zip",
        "Loitering_light/20260908150028628_F6B80AMPBVCDD6C_L_0_L0120908150028_1.mp4",
    ),
    "P02_loitering_negative_r1_low_light_frontal_oblique": (
        "Loitering_lowlight.zip",
        "Loitering_lowlight/20260908161130716_F6B80AMPBVCDD6C_L_0_L0120908161130_1.mp4",
    ),
    "P02_suspicious_behavior_positive_r1_daylight_frontal_oblique": (
        "Suspicious_light.zip",
        "Suspicious_light/2aOboQxAyfcL5KDKwqc66UyRQpCp6gCnfKxqJUmW (1).mp4",
    ),
    "P02_suspicious_behavior_negative_r1_low_light_side_oblique": (
        "Suspicious&Thefnegative.zip",
        "Suspicious&Thefnegative/20260908163124828_F6B80AMPBVCDD6C_L_0_L0120908163124_1.mp4",
    ),
    "P02_theft_positive_r1_daylight_side_oblique": (
        "Thef_light.zip", "361702087876935531011.mp4"
    ),
    "P02_theft_negative_r1_low_light_frontal_oblique": (
        "Suspicious&Thefnegative.zip",
        "Suspicious&Thefnegative/20260908162953096_F6B80AMPBVCDD6C_L_0_L0120908162953_1.mp4",
    ),
    "P02_line_crossing_positive_r1_daylight_frontal_oblique": (
        "Line_Crossing_light.zip",
        "Line_Crossing_light/20260908150756785_F6B80AMPBVCDD6C_L_0_L0120908150756_1.mp4",
    ),
    "P02_line_crossing_negative_r1_low_light_side_oblique": (
        "3197319469530502097.mp4", None
    ),
    "P02_intrusion_positive_r2_low_light_side_oblique": (
        "Intrusion_lowlight.zip",
        "20260908163924795_F6B80AMPBVCDD6C_L_0_L0120908163924_1.mp4",
    ),
    "P02_intrusion_negative_r2_daylight_frontal_oblique": (
        "Intrusion_positive.zip", "49393196503606430892.mp4"
    ),
    "P02_loitering_positive_r2_low_light_frontal_oblique": (
        "Loitering_lowlight.zip",
        "Loitering_lowlight/20260908160937665_F6B80AMPBVCDD6C_L_0_L0120908160937_1.mp4",
    ),
    "P02_loitering_negative_r2_daylight_side_oblique": (
        "Loitering_light.zip",
        "Loitering_light/20260908150326134_F6B80AMPBVCDD6C_L_0_L0120908150326_1.mp4",
    ),
    "P02_suspicious_behavior_positive_r2_low_light_side_oblique": (
        "Suspicious&Thefnegative.zip",
        "Suspicious&Thefnegative/20260908162050217_F6B80AMPBVCDD6C_L_0_L0120908162050_1.mp4",
    ),
    "P02_suspicious_behavior_negative_r2_daylight_frontal_oblique": (
        "Suspicious_light.zip",
        "Suspicious_light/2aOboQxAzshFaXQ8HxmRbennGTf7vffcXKIdXaqm.mp4",
    ),
    "P02_theft_positive_r2_low_light_frontal_oblique": (
        "Thef_lowlight.zip",
        "Thef_lowlight/20260908161419097_F6B80AMPBVCDD6C_L_0_L0120908161419_1.mp4",
    ),
    "P02_theft_negative_r2_daylight_side_oblique": (
        "Thef_light.zip", "513291836995261253512.mp4"
    ),
    "P02_line_crossing_positive_r2_low_light_side_oblique": (
        "LineCrossingLow.zip",
        "20260908164836302_F6B80AMPBVCDD6C_L_0_L0120908164836_1.mp4",
    ),
    "P02_line_crossing_negative_r2_daylight_frontal_oblique": (
        "Line_Crossing_light.zip",
        "Line_Crossing_light/20260908150943220_F6B80AMPBVCDD6C_L_0_L0120908150943_1.mp4",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: Path, member: str | None, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if member is None:
        shutil.copy2(source, destination)
    else:
        with zipfile.ZipFile(source) as archive, archive.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return True


def video_metadata(path: Path) -> dict[str, str | int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "frames": frames,
        "fps": round(fps, 6),
        "duration_seconds": round(frames / fps, 3) if fps else 0.0,
        "width": width,
        "height": height,
    }


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    with P01_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        p01_rows = list(csv.DictReader(stream))

    output_rows: list[dict[str, object]] = []
    for p01_row in p01_rows:
        p01_id = p01_row["clip_id"]
        p02_id = p01_id.replace("P01_", "P02_", 1)
        source_name, member = SOURCES[p02_id]
        source = SOURCE_ROOT / source_name
        session_dir = p01_row["group_id"]
        group_id = f"P02_{session_dir}"
        destination = VIDEO_ROOT / p01_row["split"] / session_dir / f"{p02_id}.mp4"
        present = materialize(source, member, destination)

        config_path = CONFIG_ROOT / f"{p02_id}.yaml"
        with (P01_CONFIG_ROOT / f"{p01_id}.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["experiment_version"] = VERSION
        config["clip_id"] = p02_id
        config["participant_id"] = "P02"
        config["session_id"] = group_id
        config["camera_id"] = f"pilot_{p02_id}"
        config["name"] = f"Pilot {p02_id}"
        config["source"] = destination.relative_to(ROOT).as_posix()
        config["frame_rotation"] = "cw90" if "daylight" in p02_id else "none"
        config["manual_confirmation"] = {
            "status": "CONFIRMED_BY_AI_VISUAL_QA",
            "basis": "Second-participant recordings accepted before inference; fixed protocol thresholds.",
            "do_not_change": "experiment_version, behavior thresholds, expected behavior/label",
        }
        if "low_light" in p02_id and config.get("lines"):
            config["lines"][0]["point1"] = [0.36, 0.74]
            config["lines"][0]["point2"] = [0.43, 0.31]
            config["lines"][0]["direction"] = "forward"
        with config_path.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

        row: dict[str, object] = {
            "experiment_version": VERSION,
            "clip_id": p02_id,
            "group_id": group_id,
            "split": p01_row["split"],
            "video_path": destination.relative_to(ROOT).as_posix(),
            "camera_config_path": config_path.relative_to(ROOT).as_posix(),
            "status": "READY" if present else "MISSING_SOURCE",
            "source_path": f"{source}!{member}" if member else str(source),
            "source_sha256": "",
            "video_sha256": "",
            "frames": "",
            "fps": "",
            "duration_seconds": "",
            "width": "",
            "height": "",
        }
        if present:
            digest = sha256(destination)
            row.update(video_metadata(destination))
            row["source_sha256"] = digest
            row["video_sha256"] = digest
        output_rows.append(row)

    manifest = BASE / "p02_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    ready = sum(row["status"] == "READY" for row in output_rows)
    print(f"Prepared {ready}/{len(output_rows)} clips")
    for row in output_rows:
        if row["status"] != "READY":
            print(f"MISSING: {row['clip_id']} <- {row['source_path']}")
    print(manifest)


if __name__ == "__main__":
    main()
