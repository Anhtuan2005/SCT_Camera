import csv

import pytest
import yaml

from scripts.build_priority_pilot_plan import (
    BEHAVIORS,
    EXPERIMENT_VERSION,
    build_rows,
    write_plan,
)
from scripts.evaluate_behavior_experiment import evaluate
from scripts.plot_behavior_evaluation_outcomes import build_outcome_rows
from scripts.run_behavior_experiment import _counter_deltas


def test_behavior_evaluator_matches_once_and_counts_negative_clip_alarm() -> None:
    ground_truth = [
        {"clip_id": "positive", "behavior": "intrusion", "event_id": "e1", "start_seconds": "1", "eligible_time_seconds": "2", "end_seconds": "4", "label": "positive", "split": "test"},
        {"clip_id": "negative", "behavior": "intrusion", "event_id": "", "start_seconds": "", "end_seconds": "", "label": "negative", "split": "test"},
    ]
    detections = [
        {"clip_id": "positive", "behavior": "intrusion", "video_time_seconds": "3"},
        {"clip_id": "negative", "behavior": "intrusion", "video_time_seconds": "1"},
    ]
    metrics, matches = evaluate(ground_truth, detections, "test", 1.0)
    intrusion = metrics[0]
    assert (intrusion["tp"], intrusion["fp"], intrusion["fn"]) == (1, 1, 0)
    assert intrusion["false_alarm_rate_negative_clip"] == 1.0
    assert {row["status"] for row in matches} == {"TP", "FP"}


def test_outcome_figure_separates_positive_clip_event_fp_from_negative_clip_alarm() -> None:
    metrics = [{
        "behavior": "intrusion",
        "tp": "1",
        "fp": "5",
        "tn": "1",
        "fn": "0",
        "false_alarm_rate_negative_clip": "0.0",
        "precision": "0.166667",
        "recall": "1.0",
        "f1": "0.285714",
        "latency_median_seconds": "0.457",
    }]
    ground_truth = [
        {"behavior": "intrusion", "clip_id": "positive", "label": "positive", "split": "test", "annotation_status": "VERIFIED"},
        {"behavior": "intrusion", "clip_id": "negative", "label": "negative", "split": "test", "annotation_status": "VERIFIED"},
    ]
    matches = [
        {"behavior": "intrusion", "clip_id": "positive", "status": "TP"},
        *[
            {"behavior": "intrusion", "clip_id": "positive", "status": "FP"}
            for _ in range(5)
        ],
    ]

    outcomes = build_outcome_rows(metrics, matches, ground_truth, "test")

    assert outcomes == [{
        "behavior": "intrusion",
        "matched_tp_events": 1,
        "unmatched_positive_clip_fp_events": 5,
        "negative_clip_fp_events": 0,
        "fn_events": 0,
        "negative_clips": 1,
        "negative_clips_with_false_alert": 0,
        "negative_clip_far": 0.0,
        "precision": 0.166667,
        "recall": 1.0,
        "f1": 0.285714,
        "latency_median_seconds": 0.457,
    }]


def test_direction_specific_match_does_not_accept_wrong_crossing_direction() -> None:
    ground_truth = [
        {
            "experiment_version": EXPERIMENT_VERSION,
            "clip_id": "out_crossing",
            "behavior": "line_crossing",
            "event_id": "e1",
            "start_seconds": "1",
            "eligible_time_seconds": "1",
            "end_seconds": "5",
            "expected_direction": "OUT",
            "label": "positive",
            "split": "test",
            "annotation_status": "VERIFIED",
        },
        {
            "experiment_version": EXPERIMENT_VERSION,
            "clip_id": "negative",
            "behavior": "line_crossing",
            "label": "negative",
            "split": "test",
            "annotation_status": "VERIFIED",
        },
    ]
    detections = [
        {"experiment_version": EXPERIMENT_VERSION, "clip_id": "out_crossing", "behavior": "line_crossing", "video_time_seconds": "2", "direction": "IN"},
        {"experiment_version": EXPERIMENT_VERSION, "clip_id": "out_crossing", "behavior": "line_crossing", "video_time_seconds": "3", "direction": "OUT"},
    ]
    metrics, matches = evaluate(ground_truth, detections, "test", 1.0)
    line = next(row for row in metrics if row["behavior"] == "line_crossing")
    assert (line["tp"], line["fp"], line["fn"]) == (1, 1, 0)
    assert {row["status"] for row in matches} == {"TP", "FP"}


def test_counter_instrumentation_exports_each_direction_increment() -> None:
    before = {("door", "in"): 1, ("door", "out"): 0}
    after = {("door", "in"): 2, ("door", "out"): 2}
    assert _counter_deltas(before, after) == [
        ("door", "in"),
        ("door", "out"),
        ("door", "out"),
    ]


def test_evaluator_rejects_mixed_experiment_versions() -> None:
    ground_truth = [{
        "experiment_version": "pilot-v1",
        "clip_id": "clip",
        "behavior": "intrusion",
        "label": "negative",
        "split": "test",
        "annotation_status": "VERIFIED",
    }]
    detections = [{
        "experiment_version": "historical-v0",
        "clip_id": "clip",
        "behavior": "intrusion",
        "video_time_seconds": "1",
    }]
    with pytest.raises(ValueError, match="does not match"):
        evaluate(ground_truth, detections, "test", 1.0)


def test_single_participant_pilot_has_20_clips_and_session_disjoint_groups(tmp_path) -> None:
    rows = build_rows()
    assert len(rows) == 20
    assert sum(int(row["target_duration_seconds"]) for row in rows) == 1260
    assert {row["participant_id"] for row in rows} == {"P01"}
    assert sum(row["lighting"] == "daylight" for row in rows) == 10
    assert sum(row["viewpoint"] == "frontal_oblique" for row in rows) == 10
    assert {
        (row["lighting"], row["viewpoint"]): sum(
            candidate["lighting"] == row["lighting"]
            and candidate["viewpoint"] == row["viewpoint"]
            for candidate in rows
        )
        for row in rows
    } == {
        ("daylight", "frontal_oblique"): 6,
        ("daylight", "side_oblique"): 4,
        ("low_light", "frontal_oblique"): 4,
        ("low_light", "side_oblique"): 6,
    }
    for behavior in BEHAVIORS:
        for label in ("positive", "negative"):
            cases = [row for row in rows if row["behavior"] == behavior and row["label"] == label]
            assert {row["repetition"] for row in cases} == {1, 2}
            assert {row["split"] for row in cases} == {"development", "test"}
            assert {row["lighting"] for row in cases} == {"daylight", "low_light"}
            assert {row["viewpoint"] for row in cases} == {"frontal_oblique", "side_oblique"}
    development_groups = {row["group_id"] for row in rows if row["split"] == "development"}
    test_groups = {row["group_id"] for row in rows if row["split"] == "test"}
    assert len(development_groups) == len(test_groups) == 4
    assert development_groups.isdisjoint(test_groups)
    for split in ("development", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        assert len(split_rows) == 10
        assert sum(row["lighting"] == "daylight" for row in split_rows) == 5
        assert sum(row["viewpoint"] == "frontal_oblique" for row in split_rows) == 5

    (tmp_path / "configs_pilot").mkdir()
    (tmp_path / "configs_pilot" / "obsolete_v1.yaml").write_text("old", encoding="utf-8")
    written = write_plan(tmp_path)
    assert len(list((tmp_path / "configs_pilot").glob("*.yaml"))) == 20
    assert not (tmp_path / "configs_pilot" / "obsolete_v1.yaml").exists()
    for group_id in {row["group_id"] for row in rows}:
        split = next(row["split"] for row in rows if row["group_id"] == group_id)
        assert (tmp_path / "videos_pilot" / split / group_id).is_dir()
    line_row = next(row for row in written if row["behavior"] == "line_crossing" and row["label"] == "positive")
    config = yaml.safe_load((tmp_path / "configs_pilot" / f"{line_row['clip_id']}.yaml").read_text(encoding="utf-8"))
    assert config["expected_direction"] == "OUT"
    assert config["auto_global_zone"] is False
    with (tmp_path / "behavior_ground_truth_pilot.csv").open(encoding="utf-8-sig", newline="") as handle:
        gt_rows = list(csv.DictReader(handle))
    assert len(gt_rows) == 20
    assert {row["participant_id"] for row in gt_rows} == {"P01"}
    assert {row["split"] for row in gt_rows} == {"development", "test"}
    assert {row["annotation_status"] for row in gt_rows} == {"NEEDS_MANUAL_INPUT"}
