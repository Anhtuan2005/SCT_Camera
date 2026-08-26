"""Evaluate SCT Camera person tracking against CVAT MOT ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detector import YOLOv11Detector
from core.tracker import ByteTrackTracker


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    video: Path
    ground_truth_zip: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        action="append",
        nargs=3,
        metavar=("NAME", "VIDEO", "GT_ZIP"),
        required=True,
        help="Sequence name, source video, and CVAT MOT ZIP; repeat per video",
    )
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--rotation", choices=("none", "cw90", "ccw90", "180"), default="cw90")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mot_evaluation")
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "tmp/TrackEval")
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse existing MOT prediction files and only rerun TrackEval",
    )
    return parser.parse_args()


def _load_settings(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _rotate_and_resize(
    frame: np.ndarray,
    rotation: str,
    max_height: int,
) -> tuple[np.ndarray, float]:
    if rotation == "cw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "ccw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    height, width = frame.shape[:2]
    if max_height <= 0 or height <= max_height:
        return frame, 1.0
    scale = max_height / float(height)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _processed_bbox_to_source(
    bbox_xyxy: tuple[float, float, float, float],
    rotation: str,
    scale: float,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    xr1, yr1, xr2, yr2 = (value / scale for value in bbox_xyxy)

    if rotation == "cw90":
        x1, y1, x2, y2 = yr1, source_height - xr2, yr2, source_height - xr1
    elif rotation == "ccw90":
        x1, y1, x2, y2 = source_width - yr2, xr1, source_width - yr1, xr2
    elif rotation == "180":
        x1, y1 = source_width - xr2, source_height - yr2
        x2, y2 = source_width - xr1, source_height - yr1
    else:
        x1, y1, x2, y2 = xr1, yr1, xr2, yr2

    x1 = min(max(x1, 0.0), float(source_width))
    x2 = min(max(x2, 0.0), float(source_width))
    y1 = min(max(y1, 0.0), float(source_height))
    y2 = min(max(y2, 0.0), float(source_height))
    return x1, y1, x2, y2


def _read_ground_truth(gt_zip: Path, frame_count: int) -> str:
    with zipfile.ZipFile(gt_zip) as archive:
        labels = archive.read("gt/labels.txt").decode("utf-8-sig").strip().lower()
        text = archive.read("gt/gt.txt").decode("utf-8-sig").strip()
    if labels != "person":
        raise ValueError(f"Expected only the person label in {gt_zip}, got {labels!r}")

    lines = text.splitlines()
    for line_number, row in enumerate(csv.reader(lines), start=1):
        if len(row) < 9:
            raise ValueError(f"Invalid MOT row {line_number} in {gt_zip}: {row}")
        frame, _track_id = int(row[0]), int(row[1])
        x, y, width, height = map(float, row[2:6])
        if not 1 <= frame <= frame_count:
            raise ValueError(f"Frame {frame} outside 1..{frame_count} in {gt_zip}")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid box on row {line_number} in {gt_zip}")
    return "\n".join(lines)


def _write_ground_truth_layout(
    spec: SequenceSpec,
    gt_root: Path,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    sequence_dir = gt_root / spec.name
    (sequence_dir / "gt").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "gt/gt.txt").write_text(
        _read_ground_truth(spec.ground_truth_zip, frame_count),
        encoding="utf-8",
    )
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={spec.name}",
                "imDir=img1",
                f"frameRate={fps:.6f}",
                f"seqLength={frame_count}",
                f"imWidth={width}",
                f"imHeight={height}",
                "imExt=.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _export_predictions(
    spec: SequenceSpec,
    prediction_path: Path,
    gt_root: Path,
    detector: YOLOv11Detector,
    settings: dict[str, Any],
    rotation: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(spec.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {spec.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    _write_ground_truth_layout(spec, gt_root, frame_count, fps, width, height)

    tracker = ByteTrackTracker(detector, settings)
    max_height = int(settings.get("pipeline", {}).get("processing_max_height", 0))
    rows: list[str] = []
    track_ids: set[int] = set()
    processed_frames = 0
    started_at = time.perf_counter()
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok or source_frame is None:
                break
            processed_frames += 1
            frame, scale = _rotate_and_resize(source_frame, rotation, max_height)
            for obj in tracker.track(frame):
                if obj.class_id != 0 and obj.class_name != "person":
                    continue
                x1, y1, x2, y2 = _processed_bbox_to_source(
                    obj.bbox_xyxy,
                    rotation,
                    scale,
                    width,
                    height,
                )
                box_width, box_height = x2 - x1, y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue
                track_ids.add(obj.track_id)
                rows.append(
                    f"{processed_frames},{obj.track_id},{x1:.3f},{y1:.3f},"
                    f"{box_width:.3f},{box_height:.3f},{obj.confidence:.6f},-1,-1,-1"
                )
            if processed_frames % 100 == 0:
                print(f"{spec.name}: {processed_frames}/{frame_count} frames", flush=True)
    finally:
        capture.release()

    if processed_frames != frame_count:
        raise RuntimeError(
            f"Decoded {processed_frames} frames but metadata reports {frame_count}: {spec.video}"
        )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("\n".join(rows), encoding="utf-8")
    elapsed = time.perf_counter() - started_at
    return {
        "name": spec.name,
        "video": str(spec.video),
        "ground_truth_zip": str(spec.ground_truth_zip),
        "resolution": [width, height],
        "frames": processed_frames,
        "fps": fps,
        "prediction_rows": len(rows),
        "prediction_track_ids": len(track_ids),
        "inference_seconds": round(elapsed, 3),
        "inference_fps": round(processed_frames / elapsed, 3) if elapsed else 0.0,
    }


def _reuse_predictions(
    spec: SequenceSpec,
    prediction_path: Path,
    gt_root: Path,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(spec.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {spec.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    _write_ground_truth_layout(spec, gt_root, frame_count, fps, width, height)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    text = prediction_path.read_text(encoding="utf-8").strip()
    prediction_path.write_text(text, encoding="utf-8")
    rows = list(csv.reader(text.splitlines())) if text else []
    return {
        "name": spec.name,
        "video": str(spec.video),
        "ground_truth_zip": str(spec.ground_truth_zip),
        "resolution": [width, height],
        "frames": frame_count,
        "fps": fps,
        "prediction_rows": len(rows),
        "prediction_track_ids": len({int(row[1]) for row in rows}),
        "inference_seconds": None,
        "inference_fps": None,
        "reused_predictions": True,
    }


def _percent(value: Any, *, mean: bool = False) -> float:
    array = np.asarray(value, dtype=float)
    scalar = float(array.mean()) if mean else float(array)
    return round(100.0 * scalar, 3)


def _integer(value: Any) -> int:
    return int(np.asarray(value).item())


def _metric_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    pedestrian = result["pedestrian"]
    hota = pedestrian["HOTA"]
    clear = pedestrian["CLEAR"]
    identity = pedestrian["Identity"]
    count = pedestrian["Count"]
    return {
        "sequence": name,
        "HOTA": _percent(hota["HOTA"], mean=True),
        "DetA": _percent(hota["DetA"], mean=True),
        "AssA": _percent(hota["AssA"], mean=True),
        "MOTA": _percent(clear["MOTA"]),
        "MOTP": _percent(clear["MOTP"]),
        "IDF1": _percent(identity["IDF1"]),
        "IDP": _percent(identity["IDP"]),
        "IDR": _percent(identity["IDR"]),
        "TP": _integer(clear["CLR_TP"]),
        "FP": _integer(clear["CLR_FP"]),
        "FN": _integer(clear["CLR_FN"]),
        "IDSW": _integer(clear["IDSW"]),
        "GT_Dets": _integer(count["GT_Dets"]),
        "Pred_Dets": _integer(count["Dets"]),
    }


def _run_trackeval(
    trackeval_root: Path,
    output_dir: Path,
    sequence_lengths: dict[str, int],
) -> list[dict[str, Any]]:
    if not (trackeval_root / "trackeval").is_dir():
        raise FileNotFoundError(f"TrackEval not found: {trackeval_root}")
    # TrackEval currently references NumPy aliases removed in NumPy 1.24.
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    np.bool = bool  # type: ignore[attr-defined]
    sys.path.insert(0, str(trackeval_root))
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(output_dir / "trackeval/gt"),
            "TRACKERS_FOLDER": str(output_dir / "trackeval/trackers"),
            "OUTPUT_FOLDER": str(output_dir / "trackeval/results"),
            "TRACKERS_TO_EVAL": ["sct"],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "SCTPilot",
            "SPLIT_TO_EVAL": "train",
            "DO_PREPROC": False,
            "SEQ_INFO": sequence_lengths,
            "PRINT_CONFIG": False,
        }
    )
    metric_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]
    output, messages = evaluator.evaluate([dataset], metrics)
    if messages["MotChallenge2DBox"]["sct"] != "Success":
        raise RuntimeError(messages["MotChallenge2DBox"]["sct"])
    result = output["MotChallenge2DBox"]["sct"]
    rows = [_metric_row(name, result[name]) for name in sequence_lengths]
    rows.append(_metric_row("COMBINED", result["COMBINED_SEQ"]))
    return rows


def _write_results(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    detector: YOLOv11Detector,
    settings: dict[str, Any],
    rotation: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0])
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "single-person internal pilot",
        "model": detector.model_path,
        "device": detector.device,
        "rotation": rotation,
        "processing_max_height": int(settings.get("pipeline", {}).get("processing_max_height", 0)),
        "person_confidence": detector.class_confidences.get("person", detector.confidence),
        "iou_threshold": 0.5,
        "runs": runs,
        "metrics_percent": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    specs = [
        SequenceSpec(name, Path(video).resolve(), Path(gt_zip).resolve())
        for name, video, gt_zip in args.sequence
    ]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Sequence names must be unique")
    for spec in specs:
        if not spec.video.is_file():
            raise FileNotFoundError(spec.video)
        if not spec.ground_truth_zip.is_file():
            raise FileNotFoundError(spec.ground_truth_zip)

    settings = _load_settings(args.settings.resolve())
    output_dir = args.output_dir.resolve()
    gt_root = output_dir / "trackeval/gt/SCTPilot-train"
    prediction_root = output_dir / "trackeval/trackers/SCTPilot-train/sct/data"
    detector = YOLOv11Detector(settings)
    if args.reuse_predictions:
        runs = [
            _reuse_predictions(spec, prediction_root / f"{spec.name}.txt", gt_root)
            for spec in specs
        ]
    else:
        runs = [
            _export_predictions(
                spec,
                prediction_root / f"{spec.name}.txt",
                gt_root,
                detector,
                settings,
                args.rotation,
            )
            for spec in specs
        ]
    sequence_lengths = {run["name"]: run["frames"] for run in runs}
    metrics = _run_trackeval(args.trackeval_root.resolve(), output_dir, sequence_lengths)
    _write_results(output_dir, metrics, runs, detector, settings, args.rotation)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_dir / 'metrics.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
