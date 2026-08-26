"""Copy QA-accepted pilot recordings into canonical paths and build a runner manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual/recording_plan_pilot.csv",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual/accepted_clip_sources.csv",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "experiments/priority_19_08/manual/pilot_manifest.csv",
    )
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, str | int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open copied video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {
        "frames": frames,
        "fps": round(fps, 6),
        "duration_seconds": round(frames / fps, 3),
        "width": width,
        "height": height,
    }


def main() -> int:
    args = _args()
    plans = {row["clip_id"]: row for row in _rows(args.plan)}
    sources = {row["clip_id"]: row for row in _rows(args.sources)}
    if set(plans) != set(sources):
        missing = sorted(set(plans) - set(sources))
        extra = sorted(set(sources) - set(plans))
        raise ValueError(f"Source map mismatch: missing={missing}, extra={extra}")

    manifest: list[dict[str, object]] = []
    ready = 0
    for clip_id, plan in plans.items():
        source = Path(sources[clip_id]["source_path"])
        destination = ROOT / plan["video_path"]
        config = ROOT / plan["camera_config_path"]
        row: dict[str, object] = {
            "experiment_version": plan["experiment_version"],
            "clip_id": clip_id,
            "group_id": plan["group_id"],
            "split": plan["split"],
            "video_path": plan["video_path"],
            "camera_config_path": plan["camera_config_path"],
            "status": "MISSING_SOURCE",
            "source_path": str(source),
            "source_sha256": "",
            "video_sha256": "",
            "frames": "",
            "fps": "",
            "duration_seconds": "",
            "width": "",
            "height": "",
        }
        if source.is_file():
            source_hash = _sha256(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and _sha256(destination) != source_hash:
                raise RuntimeError(f"Refusing to overwrite different file: {destination}")
            if not destination.is_file():
                shutil.copy2(source, destination)
            if not config.is_file():
                raise RuntimeError(f"Missing camera config: {config}")
            destination_hash = _sha256(destination)
            if destination_hash != source_hash:
                raise RuntimeError(f"Copy hash mismatch: {destination}")
            row.update(
                status="READY",
                source_sha256=source_hash,
                video_sha256=destination_hash,
                **_metadata(destination),
            )
            ready += 1
        manifest.append(row)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"ready={ready} missing={len(manifest) - ready} manifest={args.manifest_out}")
    return 0 if ready == len(manifest) else 2


if __name__ == "__main__":
    raise SystemExit(main())
