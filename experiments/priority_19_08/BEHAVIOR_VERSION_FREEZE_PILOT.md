# Behavior Version Freeze — Single-Participant Pilot

Experiment design version: `priority-pilot-v2-single-participant-2026-08-18`

Scope: one-participant deadline-safe pilot only. This is not full-scale validation.

## Frozen implementation

The behavior implementation and thresholds are unchanged from the previous freeze. Production behavior code is not modified for this redesign. The new version identifies a different experimental design so its clips, configs and annotations cannot be mixed with the earlier five-participant plan.

Historical `data/behavior_events.jsonl`, historical alerts, and previous pilot artifacts are prohibited as inputs. Only version-v2 P01 clips, their confirmed YAML files, fresh runner outputs, and verified v2 ground truth may enter this experiment.

## Current semantics retained

### Intrusion

- `IntrusionDetector.analyze()` calls `_analyze_lines()` only.
- Eligible objects are confirmed tracks in `intrusion_classes`; current default is `person`.
- A crossing occurs when the two-center path crosses/changes side of the line, or the line newly intersects the tracked bounding box.
- Both directions increment the internal counter, but only `direction == "in"` emits production alert `type: intrusion`.
- There is no dwell threshold and person identity kind is not a gate.

### Line crossing

- Uses the same internal geometry as intrusion.
- Both IN and OUT increment per-camera/per-line counters.
- Production emits no independent `line_crossing` alert.
- The experiment runner observes counter deltas and exports versioned `behavior=line_crossing` rows. This does not change production notifications.

### Polygon `_analyze_zones`

The method exists but is not called. Commit `2d0f9bb2` and current tests establish polygon-only non-alerting as the current contract. No polygon-intrusion metric is claimed.

## Single-participant design limitation

- Only P01 is recorded. Participant-level generalization cannot be estimated or claimed.
- There is no participant-disjoint train/validation/test split.
- Repetition 1 is `development`; repetition 2 is `test`.
- Development and test use different recording-session IDs. A session never crosses splits.
- `group_id` is the recording session, not the participant. This reduces frame/clip leakage but cannot remove dependence caused by the same person appearing in both splits.
- There is no validation/tuning stage. All thresholds stay fixed at the reported configuration; development clips are only for pipeline/annotation checks.
- Any test metric is descriptive evidence for P01 under these recorded conditions, not an estimate for unseen participants.

## Independent intrusion/line-crossing evaluation

- Intrusion positive: IN crossing. Intrusion negative: OUT crossing without returning IN.
- Line-crossing positive: OUT crossing. Line-crossing negative: the entire person bbox stays clear of the line.
- The evaluator requires the expected IN/OUT direction. Wrong-direction detections remain false positives.
- The two outputs share geometry and must be described as `line-crossing mechanism` and `intrusion alert (IN subset)`, not as independent algorithms.

## Reproducibility guard

`PILOT_EXPERIMENT_FREEZE.yaml` stores SHA-256 hashes for behavior modules, detector/tracker/pose, settings, runner/evaluator and model weights. The runner rejects changed frozen files or mismatched manifest/config versions.

Per-clip ROI/line YAML coordinates remain manually editable before recording. The fresh run summary captures final config and video SHA-256 values.

