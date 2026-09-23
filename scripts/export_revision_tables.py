"""Export code-grounded feature, behavior, and historical failure tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.behavior_learning import FEATURE_NAMES


FEATURE_ROWS = [
    ("duration", "Alert duration", "float(alert['duration'] or 0)", "s; >=0", "z-score from training mean/std", "analytics.behavior_learning.extract_behavior_features"),
    ("threshold_seconds", "Rule duration threshold", "float(alert['threshold_seconds'] or 0)", "s; >=0", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("duration_ratio", "Duration-to-threshold ratio", "duration / max(threshold_seconds, 1)", ">=0", "derived ratio, then z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("near_seconds", "Person-asset proximity duration", "float(alert['near_seconds'] or 0)", "s; >=0", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("score", "Rule-based cue count", "float(alert['score'] or 0)", "count; current theft rule 0..5", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("score_ratio", "Cue-count-to-trigger ratio", "score / max(score_threshold, 1)", ">=0", "derived ratio, then z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("pacing_passes", "Pacing side changes", "float(alert['pacing_passes'] or 0)", "count; >=0", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("object_confidence", "Tracked-object confidence", "obj.confidence; identity_score fallback when no object matches", "typically [0,1]", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("bbox_area_ratio", "Bounding-box area ratio", "max(0,(x2-x1)*(y2-y1)) / frame_area", "ratio; normally [0,1]", "geometric ratio, then z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("path_length_ratio", "Track path length ratio", "sum Euclidean step lengths / frame diagonal", ">=0", "geometric ratio, then z-score", "analytics.behavior_learning._motion_metrics"),
    ("net_distance_ratio", "Net displacement ratio", "distance(first,last) / frame diagonal", ">=0", "geometric ratio, then z-score", "analytics.behavior_learning._motion_metrics"),
    ("displacement_ratio", "Maximum displacement ratio", "max distance(point,first) / frame diagonal", ">=0", "geometric ratio, then z-score", "analytics.behavior_learning._motion_metrics"),
    ("speed_ratio", "Mean normalized step length", "path_length_ratio / max(history_length-1,1)", ">=0 per history step", "derived ratio, then z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("has_vehicle_signal", "Any vehicle-related theft cue", "vehicle_started_moving OR moving_same_direction OR pose_push_contact", "{0,1}", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("vehicle_started_moving", "Vehicle displacement cue", "bool(alert['vehicle_started_moving'])", "{0,1}", "z-score", "analytics.theft_behavior.SuspiciousTheftDetector"),
    ("moving_same_direction", "Correlated-motion cue", "bool(alert['moving_same_direction'])", "{0,1}", "z-score", "analytics.theft_behavior.SuspiciousTheftDetector"),
    ("pose_push_contact", "Pose/contact cue", "bool(alert['pose_push_contact'])", "{0,1}", "z-score", "analytics.theft_behavior.SuspiciousTheftDetector"),
    ("object_count", "Objects in current frame", "len(objects)", "count; >=0", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("zone_configured", "Camera has configured zones", "bool(camera_config['zones'])", "{0,1}", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("is_person", "Person class indicator", "class_name == 'person'", "{0,1}", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("is_vehicle", "Vehicle class indicator", "class_name in VEHICLE_CLASSES", "{0,1}", "z-score", "analytics.behavior_learning.extract_behavior_features"),
    ("is_asset", "Asset class indicator", "class_name in ASSET_CLASSES", "{0,1}", "z-score", "analytics.behavior_learning.extract_behavior_features"),
]

BEHAVIOR_ROWS = [
    ("Intrusion", "Configured counting line; confirmed person track", "Consecutive centers or segment/bbox contact cross the line", "None", "Direction resolves to IN", "Per-track side/contact state; downstream alert cooldown", "analytics.intrusion.IntrusionDetector._analyze_lines"),
    ("Loitering", "Person center inside a loitering polygon", "Continuous dwell session with 3 s state grace", "None", "Zone threshold or default 20 s", "One alert per tier/session; session purge after grace/gap", "analytics.loitering.LoiteringDetector"),
    ("Unknown person", "Confirmed stranger outside explicit intrusion zones", "Continuous camera-level presence; 2 s absence resets state", "InsightFace identity state", "identity_kind == stranger", "One alert per continuous camera presence; 12 s downstream cooldown override", "analytics.unknown_person.UnknownPersonDetector"),
    ("Suspicious stranger", "Confirmed stranger inside stranger_watch polygon", "Dwell threshold; default 180 s; >=5 track-history points", "None", "standing: displacement <=0.04 diagonal; pacing: path >=0.18 and net <=0.08 diagonal", "One alert per tier/session; 3 s state grace", "analytics.suspicious_stranger.SuspiciousStrangerDetector"),
    ("Asset monitoring", "Asset inside asset_watch polygon and confirmed stranger within 0.22 frame diagonal", "Asset present >=2 s; removed/missing for zone threshold or default 6 s; person window 12 s", "None", "asset_removed or asset_missing only after proximity and timing gates", "Track handoff 2 s; reappearance confirm 1 s; default repeat suppression 300 s", "analytics.asset_watch.AssetWatchDetector"),
    ("Suspicious theft", "Confirmed stranger and vehicle inside asset_watch zone and within 1.5/12 frame diagonal", "Near duration 10 s; pacing side changes separated by >=0.8 s", "Wrist-to-vehicle distance <=0.055 frame diagonal; keypoint confidence >=0.25", "At least 2 of 5 cues, including near-duration and >=1 vehicle cue", "One alert per camera/zone/vehicle-class until camera reset; pair gap 2 s", "analytics.theft_behavior.SuspiciousTheftDetector"),
    ("Directional line crossing", "Configured line intersects center path or newly touches bbox", "Two consecutive track centers", "None", "Signed-side change; configured forward/reverse mapping produces IN/OUT", "Per-track side/contact state; emits alert only for IN, counts both", "analytics.intrusion.IntrusionDetector._analyze_lines"),
]

PRIORITY_BEHAVIOR_ROWS = [
    (
        "intrusion",
        "Confirmed tracked person; configured CountingLine; >=2 center-history points (analytics/intrusion.py:160-177)",
        "Center segment crosses the line OR the line newly intersects the bbox; only direction IN is alert-eligible",
        "Two consecutive tracked centers; no dwell-time gate",
        "None",
        "Signed-side/direction mapping; no numeric confidence threshold beyond upstream detector/tracker",
        "direction == 'in' emits alert type intrusion; current analyze() does not call polygon _analyze_zones",
        "Per camera/line/track side+touch state; state removed when track disappears; no detector-level time cooldown",
    ),
    (
        "loitering",
        "Tracked class_name == person; explicit loitering polygon; optional identity dwell policy (analytics/loitering.py:45-80)",
        "Person center is inside/on the loitering polygon",
        "Accumulated dwell; 3 s state grace; optional session_gap; threshold from zone/dwell policy or 20 s default",
        "None",
        "Current default 20 s; zone threshold and identity/time multipliers may override it",
        "duration >= effective threshold (or escalation tier time) emits loitering/tier alert",
        "One alert per tier per session; session state purged after grace/session-gap rules",
    ),
    (
        "suspicious behavior",
        "Confirmed stranger; stranger_watch polygon (explicit or auto full-frame); >=5 history points (analytics/suspicious_stranger.py:58-92,246-270)",
        "Center inside zone AND stationary displacement <=0.04 frame diagonal OR pacing path >=0.18 with net displacement <=0.08",
        "Dwell >= effective threshold; default 180 s; 3 s state grace",
        "None; this rule uses track geometry, not pose keypoints",
        "5 history points; 0.04/0.18/0.08 diagonal ratios; 180 s default",
        "confirmed stranger AND suspicious geometry reason AND duration threshold all true",
        "One alert per tier/session; session purged after grace/session-gap rules",
    ),
    (
        "theft",
        "Confirmed stranger + tracked vehicle inside asset_watch zone; motion histories; optional wrists 9/10 (analytics/theft_behavior.py:53-99,103-176)",
        "Both centers inside asset_watch; person-to-vehicle bbox distance <=1.5/12 = 0.125 frame diagonal",
        "Near >=10 s; pacing side changes >=2 with >=0.8 s between changes; pair resets after 2 s gap",
        "Wrist 9/10 confidence >=0.25 and wrist-to-vehicle bbox distance <=0.055 frame diagonal provides one cue",
        "At least 2 of 5 cues, while near-duration and >=1 vehicle cue are mandatory; move >=0.025 diagonal; cosine >=0.65",
        "score gate + mandatory near_vehicle_duration + one of vehicle_started_moving/moving_same_direction/pose_push_contact",
        "One alert per (camera, zone, vehicle class) until camera reset; pair stale default 30 s",
    ),
    (
        "line crossing",
        "Confirmed tracked person; configured CountingLine; >=2 center-history points (analytics/intrusion.py:160-224)",
        "Center segment crosses line or newly touches bbox; forward/reverse mapping yields IN or OUT",
        "Two consecutive centers; no dwell-time gate",
        "None",
        "Signed-side change/contact; no independent score threshold",
        "Both directions increment counters; current code emits no distinct line_crossing alert, and IN is emitted as intrusion",
        "Per camera/line/track side+touch state; state removed when track disappears",
    ),
]

FAILURE_ROWS = [
    ("test_does_not_assign_face_outside_person_box", "pending_person", "stranger", "Unmatched/outside face still incremented failed attempts", "analytics/person_identity.py", "Premature stranger alert", "Increment failure count only when track_id is in assessed_track_ids", "PASS"),
    ("test_person_without_usable_face_stays_pending", "pending_person", "stranger", "No usable face still incremented failed attempts", "analytics/person_identity.py", "Premature stranger alert", "Do not count an attempt unless the track was actually face-assessed", "PASS"),
    ("test_pending_person_retries_without_waiting_recognition_interval", "pending until scheduled retry", "stranger on immediate retry", "pending_person bypassed recognition_interval and retried every frame", "analytics/person_identity.py", "Rapid exhaustion of unknown_confirmation_attempts", "Make _recognition_due depend only on elapsed recognition_interval", "PASS (renamed to respects_recognition_interval)"),
    ("test_far_new_track_does_not_inherit_recent_known_identity", "pending_person", "stranger", "Inheritance was rejected correctly, but the unassessed track was counted as a failed face attempt", "analytics/person_identity.py", "Correct non-inheritance followed by wrong stranger state", "Count failures only for assessed tracks", "PASS"),
    ("test_small_nearby_track_does_not_inherit_recent_known_identity", "pending_person", "stranger", "Area-ratio inheritance gate rejected the track, then unassessed fallback was counted as failure", "analytics/person_identity.py", "Correct area gate followed by wrong stranger state", "Count failures only for assessed tracks", "PASS"),
]


def _write(path: Path, headers: list[str], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--priority-only", action="store_true")
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    names = [row[0] for row in FEATURE_ROWS]
    if names != FEATURE_NAMES:
        raise ValueError(f"Feature catalog drift: {names} != {FEATURE_NAMES}")
    feature_rows = [(str(index),) + row for index, row in enumerate(FEATURE_ROWS, start=1)]
    if args.priority_only:
        observed = {}
        if args.audit_json:
            observed = json.loads(args.audit_json.read_text(encoding="utf-8"))["features"]["value_statistics"]
        priority_rows = []
        for index, row in enumerate(FEATURE_ROWS, start=1):
            name, definition, formula, unit_range, normalization, source = row
            stats = observed.get(name)
            observed_range = f"; observed [{stats['min']}, {stats['max']}]" if stats else ""
            priority_rows.append((str(index), name, definition, source, formula, unit_range + observed_range, normalization))
        _write(
            args.out_dir / "risk_features_22.csv",
            ["ID", "Feature Name", "Definition", "Source Module", "Formula/Calculation", "Range/Unit", "Normalization"],
            priority_rows,
        )
        _write(
            args.out_dir / "behavior_definitions.csv",
            ["Behavior", "Input Evidence", "Spatial Condition", "Temporal Condition", "Pose Condition", "Threshold", "Trigger Rule", "Cooldown/Dedup"],
            PRIORITY_BEHAVIOR_ROWS,
        )
        print(f"wrote={args.out_dir / 'risk_features_22.csv'}")
        print(f"wrote={args.out_dir / 'behavior_definitions.csv'}")
        return 0
    _write(args.out_dir / "risk_features_22.csv", ["feature_id", "feature_name", "definition", "formula_source", "unit_range", "normalization", "source_module"], feature_rows)
    _write(args.out_dir / "behavior_rules.csv", ["behavior", "spatial_evidence", "temporal_evidence", "pose_evidence", "threshold_trigger", "dedup_cooldown", "source_module"], BEHAVIOR_ROWS)
    _write(args.out_dir / "historical_failure_analysis.csv", ["test_case", "expected", "actual", "root_cause", "module", "impact", "fix", "current_status"], FAILURE_ROWS)
    print(f"wrote={args.out_dir / 'risk_features_22.csv'}")
    print(f"wrote={args.out_dir / 'behavior_rules.csv'}")
    print(f"wrote={args.out_dir / 'historical_failure_analysis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
