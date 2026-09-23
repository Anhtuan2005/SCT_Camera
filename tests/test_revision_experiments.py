from __future__ import annotations

import numpy as np

from analytics.behavior_learning import FEATURE_NAMES
from scripts.evaluate_behavior_risk import (
    _write_confusion_svg,
    _write_roc_svg,
    binary_metrics,
    stratified_group_split,
    stratified_split,
)
from scripts.export_revision_tables import FEATURE_ROWS
from scripts.run_revision_benchmark import summarize
from scripts.train_behavior_classifier import load_external_labels


def test_feature_catalog_matches_runtime_order() -> None:
    assert [row[0] for row in FEATURE_ROWS] == FEATURE_NAMES


def test_stratified_split_keeps_both_classes_in_each_partition() -> None:
    targets = np.asarray([0] * 10 + [1] * 10, dtype=float)
    split = stratified_split(targets, validation_fraction=0.2, test_fraction=0.2, seed=7)
    for indices in split.values():
        assert set(targets[indices]) == {0.0, 1.0}


def test_group_split_never_leaks_a_group_across_partitions() -> None:
    targets = np.asarray(([0, 1] * 10), dtype=float)
    groups = [f"clip_{index // 2}" for index in range(20)]
    split = stratified_group_split(targets, groups, validation_fraction=0.2, test_fraction=0.2, seed=7)
    split_groups = [{groups[index] for index in indices} for indices in split.values()]
    assert not (split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2])


def test_binary_metrics_and_runtime_summary() -> None:
    metrics = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.8, 0.7, 0.9]), 0.5)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    runs = [
        {"camera_count": 2, "aggregate_fps": 20.0, "rss_peak_mb": 100.0, "latency": {"total": {"p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 30.0}}},
        {"camera_count": 2, "aggregate_fps": 24.0, "rss_peak_mb": 120.0, "latency": {"total": {"p50_ms": 12.0, "p95_ms": 22.0, "p99_ms": 32.0}}},
    ]
    summary = summarize(runs)[0]
    assert summary["aggregate_fps_mean"] == 22.0
    assert summary["fps_per_camera_mean"] == 11.0


def test_risk_plot_writers_create_svg(tmp_path) -> None:
    _write_confusion_svg(tmp_path / "cm.svg", {"tn": 3, "fp": 1, "fn": 2, "tp": 4})
    _write_roc_svg(tmp_path / "roc.svg", [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}], 0.5)
    assert (tmp_path / "cm.svg").read_text(encoding="utf-8").startswith("<svg")
    assert (tmp_path / "roc.svg").read_text(encoding="utf-8").startswith("<svg")


def test_external_label_loader_accepts_excel_utf8_bom(tmp_path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("\ufeffevent_id,label,notes\ne1,true_positive,checked\n", encoding="utf-8")
    assert load_external_labels(labels)["e1"]["label"] == "true_positive"
