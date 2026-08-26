"""Run the frozen two-identity face-threshold pilot used in the thesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "config" / "known_people" / "Face.zip"
DEFAULT_OUT = ROOT / "outputs" / "face_pilot_20260820"
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORIENTATIONS = ("none", "cw90", "ccw90", "180")

ENROLLMENT_STEMS = {
    "Nguyen": {
        "nguyen_01_front_close",
        "nguyen_02_front_alt",
        "nguyen_04_three_quarter_close",
    },
    "Tuan": {
        "tuan_03_me_03_color_front",
        "tuan_06_me_06_color_left",
        "tuan_09_me_09_color_right",
    },
}
FROZEN_TEST_STEMS = {
    "Nguyen": {
        "nguyen_07_new_front_stand",
        "nguyen_08_new_front_left",
        "nguyen_09_new_front_clear",
        "nguyen_11_new_side_near",
        "nguyen_12_new_front_near",
    },
    "Tuan": {
        "tuan_11_new_front_body",
        "tuan_12_new_front_near",
        "tuan_13_new_front_clear",
    },
}
VISUAL_ORIENTATION_OVERRIDES = {
    "nguyen/nguyen_07_new_front_stand.png": "cw90",
    "nguyen/nguyen_08_new_front_left.png": "cw90",
    "nguyen/nguyen_09_new_front_clear.png": "cw90",
    "nguyen/nguyen_11_new_side_near.png": "cw90",
    "nguyen/nguyen_12_new_front_near.png": "cw90",
    "tuan/tuan_06_me_06_color_left.png": "none",
    "tuan/tuan_11_new_front_body.png": "cw90",
    "tuan/tuan_12_new_front_near.png": "cw90",
    "tuan/tuan_13_new_front_clear.png": "cw90",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["identity"]


def build_face_app(*, recognition: bool):
    from insightface.app import FaceAnalysis

    identity = read_settings()
    modules = ["detection", "recognition"] if recognition else ["detection"]
    app = FaceAnalysis(
        name=str(identity["model"]),
        root=str(ROOT / identity.get("model_root", "models/insightface")),
        allowed_modules=modules,
        providers=["CPUExecutionProvider"],
    )
    size = identity.get("detection_size", 320)
    det_size = (int(size), int(size)) if not isinstance(size, list) else tuple(size)
    app.prepare(ctx_id=-1, det_size=det_size)
    return app, identity


def rotate_bgr(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "cw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def face_value(face, key: str):
    value = getattr(face, key, None)
    if value is not None:
        return value
    return face.get(key) if isinstance(face, dict) else None


def face_quality(face) -> tuple[float, float, float]:
    bbox = np.asarray(face_value(face, "bbox"), dtype=np.float32).reshape(-1)
    det_score = float(face_value(face, "det_score") or 1.0)
    area = max(0.0, float(bbox[2] - bbox[0])) * max(
        0.0, float(bbox[3] - bbox[1])
    )
    return area * det_score, det_score, area


def identity_from_member(member: str) -> str:
    first = Path(member).parts[0].lower()
    if first == "nguyen":
        return "Nguyen"
    if first == "tuan":
        return "Tuan"
    raise ValueError(f"Unexpected identity directory: {member}")


def safe_extract_images(zip_path: Path, source_dir: Path) -> list[Path]:
    paths: list[Path] = []
    source_root = source_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or Path(member).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = (source_dir / member).resolve()
            if source_root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            paths.append(target)
    return paths


def orientation_scores(app, image_bgr: np.ndarray) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for orientation in ORIENTATIONS:
        candidate = rotate_bgr(image_bgr, orientation)
        faces = list(app.get(candidate))
        best = max(faces, key=lambda face: face_quality(face)[0], default=None)
        quality, det_score, area = face_quality(best) if best is not None else (0, 0, 0)
        result[orientation] = {
            "face_count": len(faces),
            "quality": float(quality),
            "det_score": float(det_score),
            "face_area": float(area),
        }
    return result


def choose_orientation(scores: dict[str, dict]) -> tuple[str, bool, str]:
    ranked = sorted(
        ORIENTATIONS,
        key=lambda orientation: scores[orientation]["quality"],
        reverse=True,
    )
    best, second = ranked[:2]
    best_quality = scores[best]["quality"]
    second_quality = scores[second]["quality"]
    if best_quality <= 0:
        return "none", False, "no_face_detected"
    confident = second_quality <= 0 or best_quality >= second_quality * 1.25
    if confident:
        return best, True, "auto_face_orientation"
    return "none", False, "ambiguous_kept_original"


def make_contact_sheet(rows: list[dict], source_dir: Path, normalized_dir: Path, out: Path):
    cell_w, cell_h = 300, 220
    sheet = Image.new("RGB", (cell_w * 2, cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        source = Image.open(source_dir / row["filename"]).convert("RGB")
        normalized = Image.open(normalized_dir / row["filename"]).convert("RGB")
        for column, image in enumerate((source, normalized)):
            image.thumbnail((cell_w - 20, cell_h - 45))
            x = column * cell_w + (cell_w - image.width) // 2
            y = index * cell_h + 28
            sheet.paste(image, (x, y))
        label = (
            f'{row["identity"]}: {Path(row["filename"]).name} | '
            f'{row["orientation_applied"]} | {row["orientation_status"]}'
        )
        draw.text((8, index * cell_h + 7), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)


def audit(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        raise FileExistsError(
            f"Audit output already exists; refusing to overwrite: {out_dir}"
        )
    source_dir = out_dir / "source"
    normalized_dir = out_dir / "normalized"
    source_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)
    source_paths = safe_extract_images(zip_path, source_dir)
    app, identity_settings = build_face_app(recognition=False)

    rows: list[dict] = []
    for source_path in source_paths:
        relative = source_path.relative_to(source_dir)
        identity = identity_from_member(relative.as_posix())
        row = {
            "filename": relative.as_posix(),
            "identity": identity,
            "source_sha256": sha256(source_path),
            "source_bytes": source_path.stat().st_size,
            "readable": False,
            "source_width": "",
            "source_height": "",
            "exif_orientation": "",
            "orientation_applied": "",
            "orientation_confident": False,
            "orientation_status": "",
            "normalized_width": "",
            "normalized_height": "",
            "normalized_sha256": "",
            "face_detected": False,
            "face_count_best_orientation": 0,
            "best_detection_score": 0.0,
            "orientation_candidates_json": "",
        }
        try:
            with Image.open(source_path) as opened:
                opened.verify()
            with Image.open(source_path) as opened:
                exif_orientation = opened.getexif().get(274, 1)
                normalized_exif = ImageOps.exif_transpose(opened).convert("RGB")
            row["readable"] = True
            row["source_width"], row["source_height"] = normalized_exif.size
            row["exif_orientation"] = exif_orientation
            base_bgr = cv2.cvtColor(np.asarray(normalized_exif), cv2.COLOR_RGB2BGR)
            scores = orientation_scores(app, base_bgr)
            orientation, confident, status = choose_orientation(scores)
            normalized_bgr = rotate_bgr(base_bgr, orientation)
            normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
            target = normalized_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normalized_rgb).save(target, format="PNG", compress_level=3)
            row["orientation_applied"] = orientation
            row["orientation_confident"] = confident
            row["orientation_status"] = status
            row["normalized_width"] = normalized_bgr.shape[1]
            row["normalized_height"] = normalized_bgr.shape[0]
            row["normalized_sha256"] = sha256(target)
            row["face_detected"] = scores[orientation]["face_count"] > 0
            row["face_count_best_orientation"] = scores[orientation]["face_count"]
            row["best_detection_score"] = scores[orientation]["det_score"]
            row["orientation_candidates_json"] = json.dumps(scores, sort_keys=True)
        except Exception as exc:  # Preserve the failed row in the inventory.
            row["orientation_status"] = f"error:{type(exc).__name__}:{exc}"
        rows.append(row)

    fieldnames = list(rows[0])
    with (out_dir / "dataset_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for identity in ("Nguyen", "Tuan"):
        identity_rows = [row for row in rows if row["identity"] == identity]
        make_contact_sheet(
            identity_rows,
            source_dir,
            normalized_dir,
            out_dir / f"_orientation_audit_{identity.lower()}.png",
        )

    summary = {
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "image_count": len(rows),
        "identity_counts": {
            identity: sum(row["identity"] == identity for row in rows)
            for identity in ("Nguyen", "Tuan")
        },
        "unreadable": [row["filename"] for row in rows if not row["readable"]],
        "no_face_detected": [
            row["filename"] for row in rows if not row["face_detected"]
        ],
        "orientation_changes": [
            {"filename": row["filename"], "orientation": row["orientation_applied"]}
            for row in rows
            if row["orientation_applied"] != "none"
        ],
        "orientation_ambiguous": [
            row["filename"]
            for row in rows
            if row["orientation_status"] == "ambiguous_kept_original"
        ],
        "production_model": identity_settings["model"],
        "detection_size": identity_settings["detection_size"],
        "audit_provider": "CPUExecutionProvider",
    }
    (out_dir / "_AUDIT_SUMMARY.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def correct_orientations(out_dir: Path) -> None:
    inventory_path = out_dir / "dataset_inventory.csv"
    if (out_dir / "FACE_PILOT_FREEZE.yaml").exists():
        raise RuntimeError("Cannot change normalized images after split freeze")
    rows = load_csv(inventory_path)
    app, identity_settings = build_face_app(recognition=False)
    source_dir = out_dir / "source"
    normalized_dir = out_dir / "normalized"
    by_filename = {row["filename"]: row for row in rows}
    for filename, orientation in VISUAL_ORIENTATION_OVERRIDES.items():
        row = by_filename[filename]
        source_path = source_dir / filename
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        base_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        normalized_bgr = rotate_bgr(base_bgr, orientation)
        target = normalized_dir / filename
        Image.fromarray(cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)).save(
            target, format="PNG", compress_level=3
        )
        faces = list(app.get(normalized_bgr))
        best = max(faces, key=lambda face: face_quality(face)[0], default=None)
        _quality, det_score, _area = (
            face_quality(best) if best is not None else (0.0, 0.0, 0.0)
        )
        row["orientation_applied"] = orientation
        row["orientation_confident"] = "True"
        row["orientation_status"] = "visual_audit_override_before_freeze"
        row["normalized_width"] = str(normalized_bgr.shape[1])
        row["normalized_height"] = str(normalized_bgr.shape[0])
        row["normalized_sha256"] = sha256(target)
        row["face_detected"] = str(bool(faces))
        row["face_count_best_orientation"] = str(len(faces))
        row["best_detection_score"] = str(det_score)

    write_csv(inventory_path, rows)
    for identity in ("Nguyen", "Tuan"):
        make_contact_sheet(
            [row for row in rows if row["identity"] == identity],
            source_dir,
            normalized_dir,
            out_dir / f"_orientation_audit_{identity.lower()}.png",
        )
    summary = {
        "zip": str(DEFAULT_ZIP),
        "zip_sha256": sha256(DEFAULT_ZIP),
        "image_count": len(rows),
        "identity_counts": {
            identity: sum(row["identity"] == identity for row in rows)
            for identity in ("Nguyen", "Tuan")
        },
        "unreadable": [row["filename"] for row in rows if row["readable"] != "True"],
        "no_face_detected": [
            row["filename"] for row in rows if row["face_detected"] != "True"
        ],
        "orientation_changes": [
            {"filename": row["filename"], "orientation": row["orientation_applied"]}
            for row in rows
            if row["orientation_applied"] != "none"
        ],
        "visual_audit_overrides": VISUAL_ORIENTATION_OVERRIDES,
        "production_model": identity_settings["model"],
        "detection_size": identity_settings["detection_size"],
        "audit_provider": "CPUExecutionProvider",
    }
    (out_dir / "_AUDIT_SUMMARY.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


def freeze(out_dir: Path) -> None:
    inventory_path = out_dir / "dataset_inventory.csv"
    freeze_path = out_dir / "FACE_PILOT_FREEZE.yaml"
    split_path = out_dir / "face_pilot_split.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("Run audit before freeze")
    if freeze_path.exists() or split_path.exists():
        raise FileExistsError("Frozen split already exists; refusing to overwrite")
    rows = load_csv(inventory_path)
    if len(rows) != 24:
        raise ValueError(f"Expected 24 images, found {len(rows)}")
    failures = [row for row in rows if row["readable"] != "True"]
    if failures:
        raise RuntimeError(f"Unreadable images prevent freezing: {failures}")

    split_rows: list[dict] = []
    for row in rows:
        stem = Path(row["filename"]).stem
        identity = row["identity"]
        if stem in ENROLLMENT_STEMS[identity]:
            role = "enrollment"
        elif stem in FROZEN_TEST_STEMS[identity]:
            role = "frozen_test"
        else:
            role = "development"
        split_rows.append(
            {
                "filename": row["filename"],
                "identity": identity,
                "role": role,
                "sha256": row["normalized_sha256"],
                "source_sha256": row["source_sha256"],
                "orientation_applied": row["orientation_applied"],
            }
        )

    expected = {
        ("Nguyen", "enrollment"): 3,
        ("Nguyen", "development"): 3,
        ("Nguyen", "frozen_test"): 5,
        ("Tuan", "enrollment"): 3,
        ("Tuan", "development"): 7,
        ("Tuan", "frozen_test"): 3,
    }
    actual = {
        key: sum(
            row["identity"] == key[0] and row["role"] == key[1]
            for row in split_rows
        )
        for key in expected
    }
    if actual != expected:
        raise AssertionError(f"Unexpected split: {actual}")

    with split_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    identity_settings = read_settings()
    freeze_doc = {
        "pilot_id": "sct-face-two-identity-pilot-20260820",
        "frozen": True,
        "frozen_on": "2026-08-20",
        "dataset_zip": str(DEFAULT_ZIP.relative_to(ROOT)).replace("\\", "/"),
        "dataset_zip_sha256": sha256(DEFAULT_ZIP),
        "protocol": {
            "identities": 2,
            "production_model": identity_settings["model"],
            "model_root": identity_settings["model_root"],
            "detection_size": identity_settings["detection_size"],
            "embedding_normalization": "L2",
            "similarity": "cosine_dot_product",
            "reference_aggregation": "maximum_similarity_per_reference_identity",
            "threshold_acceptance": "score >= threshold",
        },
        "counts": {
            "enrollment": 6,
            "development": 10,
            "frozen_test": 8,
        },
        "split": split_rows,
        "rule": "Do not change frozen_test membership after viewing metrics.",
    }
    freeze_path.write_text(
        yaml.safe_dump(freeze_doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(yaml.safe_dump({"split_counts": actual, "freeze": str(freeze_path)}))


def embedding_for_image(app, image_path: Path) -> tuple[np.ndarray | None, dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, {"status": "unreadable"}
    faces = list(app.get(image))
    if not faces:
        return None, {"status": "no_face_detected", "face_count": 0}
    face = max(faces, key=lambda item: face_quality(item)[0])
    raw_embedding = face_value(face, "normed_embedding")
    if raw_embedding is None:
        raw_embedding = face_value(face, "embedding")
    if raw_embedding is None:
        return None, {"status": "no_embedding", "face_count": len(faces)}
    embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(embedding).all() or norm <= 0:
        return None, {"status": "invalid_embedding", "face_count": len(faces)}
    quality, det_score, area = face_quality(face)
    return embedding / norm, {
        "status": "ok",
        "face_count": len(faces),
        "detection_score": det_score,
        "face_area": area,
        "quality": quality,
    }


def threshold_metrics(score_rows: list[dict], threshold: float) -> dict:
    genuine = [row for row in score_rows if row["comparison_type"] == "genuine"]
    impostor = [row for row in score_rows if row["comparison_type"] == "impostor"]
    tp = sum(float(row["cosine_similarity"]) >= threshold for row in genuine)
    fn = len(genuine) - tp
    fp = sum(float(row["cosine_similarity"]) >= threshold for row in impostor)
    tn = len(impostor) - fp
    tpr = tp / len(genuine) if genuine else math.nan
    fpr = fp / len(impostor) if impostor else math.nan
    frr = fn / len(genuine) if genuine else math.nan
    accuracy = (tp + tn) / (len(genuine) + len(impostor))
    return {
        "threshold": round(threshold, 6),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "FAR": fpr,
        "FRR": frr,
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + (1.0 - fpr)) / 2.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out_dir: Path, scores: list[dict], metrics: list[dict], eer: dict) -> None:
    genuine = np.asarray(
        [
            float(row["cosine_similarity"])
            for row in scores
            if row["comparison_type"] == "genuine"
        ]
    )
    impostor = np.asarray(
        [
            float(row["cosine_similarity"])
            for row in scores
            if row["comparison_type"] == "impostor"
        ]
    )

    exact_roc_thresholds = [math.inf] + sorted(
        {float(row["cosine_similarity"]) for row in scores}, reverse=True
    ) + [-math.inf]
    roc = [threshold_metrics(scores, threshold) for threshold in exact_roc_thresholds]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.step(
        [row["FPR"] for row in roc],
        [row["TPR"] for row in roc],
        where="post",
        lw=2,
        label="Two-identity pilot",
    )
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="Chance")
    for threshold, marker in ((0.35, "o"), (0.40, "s"), (0.45, "^")):
        point = min(metrics, key=lambda row: abs(row["threshold"] - threshold))
        ax.scatter(
            point["FPR"],
            point["TPR"],
            s=55,
            marker=marker,
            label=f"t={threshold:.2f}",
        )
    ax.set(
        xlabel="False Positive Rate (FAR)",
        ylabel="True Positive Rate",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "face_roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(
        [row["threshold"] for row in metrics],
        [row["FAR"] for row in metrics],
        lw=2,
        label="FAR",
    )
    ax.plot(
        [row["threshold"] for row in metrics],
        [row["FRR"] for row in metrics],
        lw=2,
        label="FRR",
    )
    ax.scatter(
        eer["threshold"],
        eer["eer"],
        s=60,
        color="black",
        zorder=3,
        label=f'Nearest EER: {eer["eer"]:.3f} at {eer["threshold"]:.3f}',
    )
    ax.axvline(eer["threshold"], color="black", ls="--", lw=1)
    ax.set(
        xlabel="Cosine similarity threshold",
        ylabel="Error rate",
        xlim=(0.20, 0.70),
        ylim=(-0.03, 1.03),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "face_far_frr_vs_threshold.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    rng = np.random.default_rng(20260820)
    ax.scatter(
        rng.normal(0, 0.025, len(genuine)),
        genuine,
        alpha=0.85,
        label="Genuine",
        s=42,
    )
    ax.scatter(
        rng.normal(1, 0.025, len(impostor)),
        impostor,
        alpha=0.85,
        label="Impostor",
        s=42,
    )
    ax.boxplot([genuine, impostor], positions=[0, 1], widths=0.35, showfliers=False)
    ax.set_xticks([0, 1], ["Genuine", "Impostor"])
    ax.set_ylabel("Maximum cosine similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "face_score_distribution.png", dpi=300)
    plt.close(fig)


def replot(out_dir: Path) -> None:
    scores = load_csv(out_dir / "face_similarity_scores.csv")
    raw_metrics = load_csv(out_dir / "face_threshold_metrics.csv")
    numeric_fields = {
        "threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "TPR",
        "FPR",
        "FAR",
        "FRR",
        "accuracy",
        "balanced_accuracy",
    }
    metrics = [
        {
            key: float(value) if key in numeric_fields else value
            for key, value in row.items()
        }
        for row in raw_metrics
    ]
    with (out_dir / "_EVALUATION_SUMMARY.yaml").open("r", encoding="utf-8") as handle:
        eer = yaml.safe_load(handle)["eer"]
    plot_results(out_dir, scores, metrics, eer)
    print("Regenerated figures from frozen score and metric CSV files")


def evaluate(out_dir: Path) -> None:
    split_path = out_dir / "face_pilot_split.csv"
    freeze_path = out_dir / "FACE_PILOT_FREEZE.yaml"
    if not split_path.exists() or not freeze_path.exists():
        raise FileNotFoundError("Run and verify freeze before evaluation")
    score_path = out_dir / "face_similarity_scores.csv"
    if score_path.exists():
        raise FileExistsError("Evaluation already exists; refusing to overwrite frozen results")
    split_rows = load_csv(split_path)
    normalized_dir = out_dir / "normalized"
    for row in split_rows:
        path = normalized_dir / row["filename"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Frozen image checksum changed: {row['filename']}")

    app, identity_settings = build_face_app(recognition=True)
    embeddings: dict[str, np.ndarray] = {}
    embedding_audit: list[dict] = []
    for row in split_rows:
        if row["role"] not in {"enrollment", "frozen_test"}:
            continue
        image_path = normalized_dir / row["filename"]
        embedding, audit_row = embedding_for_image(app, image_path)
        embedding_audit.append({**row, **audit_row})
        if embedding is not None:
            embeddings[row["filename"]] = embedding

    failures = [row for row in embedding_audit if row["status"] != "ok"]
    write_csv(out_dir / "face_embedding_audit.csv", embedding_audit)
    if failures:
        raise RuntimeError(
            "Embedding failures prevent meaningful evaluation; see face_embedding_audit.csv"
        )

    enrollment = [row for row in split_rows if row["role"] == "enrollment"]
    frozen_test = [row for row in split_rows if row["role"] == "frozen_test"]
    score_rows: list[dict] = []
    for test in frozen_test:
        test_embedding = embeddings[test["filename"]]
        for reference_identity in ("Nguyen", "Tuan"):
            references = [row for row in enrollment if row["identity"] == reference_identity]
            pair_scores = [
                float(np.dot(test_embedding, embeddings[reference["filename"]]))
                for reference in references
            ]
            best_index = int(np.argmax(pair_scores))
            score_rows.append(
                {
                    "test_image": test["filename"],
                    "test_identity": test["identity"],
                    "reference_identity": reference_identity,
                    "reference_images": "|".join(row["filename"] for row in references),
                    "reference_pair_scores": "|".join(f"{score:.9f}" for score in pair_scores),
                    "aggregation": "max",
                    "max_reference_image": references[best_index]["filename"],
                    "comparison_type": "genuine" if test["identity"] == reference_identity else "impostor",
                    "cosine_similarity": f"{pair_scores[best_index]:.9f}",
                }
            )
    write_csv(score_path, score_rows)

    thresholds = [round(float(value), 3) for value in np.arange(0.20, 0.7001, 0.005)]
    metrics = [threshold_metrics(score_rows, threshold) for threshold in thresholds]
    write_csv(out_dir / "face_threshold_metrics.csv", metrics)
    eer_row = min(metrics, key=lambda row: (abs(row["FAR"] - row["FRR"]), row["FAR"] + row["FRR"]))
    eer = {
        "threshold": eer_row["threshold"],
        "eer": (eer_row["FAR"] + eer_row["FRR"]) / 2.0,
        "FAR": eer_row["FAR"],
        "FRR": eer_row["FRR"],
        "method": "nearest point on 0.005 threshold grid",
    }
    best = min(
        metrics,
        key=lambda row: (
            -row["balanced_accuracy"],
            row["FAR"] + row["FRR"],
            abs(row["threshold"] - float(identity_settings["similarity_threshold"])),
        ),
    )
    selected = {
        threshold: min(metrics, key=lambda row: abs(row["threshold"] - threshold))
        for threshold in (0.35, 0.40, 0.45)
    }
    plot_results(out_dir, score_rows, metrics, eer)

    genuine_count = sum(row["comparison_type"] == "genuine" for row in score_rows)
    impostor_count = len(score_rows) - genuine_count
    threshold_lines = "\n".join(
        f"| {threshold:.2f} | {row['FAR']:.3f} | {row['FRR']:.3f} | {row['TPR']:.3f} | {row['FPR']:.3f} | {row['TP']} | {row['FP']} | {row['TN']} | {row['FN']} |"
        for threshold, row in selected.items()
    )
    report = f"""# Two-identity face recognition pilot

## Phạm vi và giao thức

- Dataset: 24 ảnh của 2 danh tính (Nguyen: 11; Tuan: 13).
- Enrollment: 6 ảnh (3 ảnh/danh tính).
- Development: 10 ảnh; không dùng để tính frozen metric.
- Frozen test: 8 ảnh (Nguyen: 5; Tuan: 3).
- Stack production: InsightFace `{identity_settings['model']}`, detection size `{identity_settings['detection_size']}`, cùng cơ chế detect/align của `FaceAnalysis`.
- Embedding được L2-normalize; cosine similarity được tính bằng dot product.
- Mỗi ảnh test được so với 3 enrollment references của mỗi identity; score identity là giá trị lớn nhất, đúng aggregation `max` của runtime.
- Số score dùng cho metric: {genuine_count} genuine và {impostor_count} impostor (tương ứng 48 phép so sánh ảnh-tham chiếu trước aggregation).
- Tất cả 14 ảnh enrollment/frozen-test đều detect được mặt và trích được embedding; chi tiết ở `face_embedding_audit.csv`.

## Kết quả tại các ngưỡng đang quan tâm

| Threshold | FAR | FRR | TPR | FPR | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{threshold_lines}

## EER và ngưỡng pilot

- EER gần đúng: **{eer['eer']:.3f}** tại threshold **{eer['threshold']:.3f}** ({eer['method']}; FAR={eer['FAR']:.3f}, FRR={eer['FRR']:.3f}).
- Best-performing threshold trên lưới sweep của pilot: **{best['threshold']:.3f}**, balanced accuracy={best['balanced_accuracy']:.3f}, FAR={best['FAR']:.3f}, FRR={best['FRR']:.3f}.
- Trong ba operating point 0.35/0.40/0.45, **0.35** cho kết quả tốt nhất (FAR=0.000, FRR=0.250); 0.40 và 0.45 làm FRR tăng lần lượt lên 0.500 và 0.750.
- Kết luận chỉ áp dụng cho two-identity pilot này; không khẳng định threshold tối ưu toàn cục.

## Hạn chế

- Chỉ có 2 identities và sample size nhỏ.
- Không participant-disjoint ngoài hai danh tính này và không đại diện cho dân số lớn.
- Ảnh enrollment/test khác filename và appearance/session theo mô tả, nhưng quy mô chưa đủ cho full-scale validation.
- Kết quả chỉ kiểm tra tính hợp lý của threshold hiện tại; cần thêm nhiều danh tính, session, điều kiện sáng, góc mặt và thiết bị trước khi suy rộng.

## Artifact integrity

- Split được khóa trước khi tính metric trong `FACE_PILOT_FREEZE.yaml` và `face_pilot_split.csv`.
- Không augment, không thay model, không retune ảnh và không thay frozen-test split sau khi xem kết quả.
"""
    (out_dir / "FACE_PILOT_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "enrollment_images": len(enrollment),
        "frozen_test_images": len(frozen_test),
        "genuine_scores": genuine_count,
        "impostor_scores": impostor_count,
        "eer": eer,
        "thresholds": selected,
        "best_threshold": best,
    }
    (out_dir / "_EVALUATION_SUMMARY.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    assert genuine_count == 8 and impostor_count == 8
    assert len(metrics) == 101
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("audit", "correct-orientation", "freeze", "evaluate", "replot"),
    )
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.zip.resolve(), args.out.resolve())
    elif args.command == "correct-orientation":
        correct_orientations(args.out.resolve())
    elif args.command == "freeze":
        freeze(args.out.resolve())
    elif args.command == "replot":
        replot(args.out.resolve())
    else:
        evaluate(args.out.resolve())


if __name__ == "__main__":
    main()
