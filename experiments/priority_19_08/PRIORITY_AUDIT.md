# Priority Audit - 2026-08-18

| Requirement | Already Available | Missing | AI Can Do | Manual Work Needed | Status |
|---|---|---|---|---|---|
| Dataset video inventory and duration | 6 config-referenced local clips; 361.127 s; hashes/resolution/FPS verified | Source video for 3,990 logged events | Audit files/config/backups and calculate metadata | Confirm whether other videos belong to the dataset | PARTIAL |
| Participants / people | Track IDs and visual contact sheets exist | True unique participant count | Prepare annotation sheet | Identify participants per clip and cross-clip identity | NEEDS MANUAL INPUT |
| Event count/type | 36,674 system predictions across 11 alert types | True event intervals/types | Count predictions and generate GT template | Annotate continuous video; do not treat alerts as GT | BLOCKED |
| Lighting / viewpoint | Visual conditions described for six clips | Formal acquisition notes for unmapped sources | Extract multi-time contact sheets | Confirm acquisition conditions where uncertain | PARTIAL |
| Camera model / source provenance | Local filenames/config linkage; visible brand/stock overlays | Exact model, original URL/owner/license | Preserve evidence and mark unknowns | Supply model and provenance/license | NEEDS MANUAL INPUT |
| Train/validation/test protocol | None in current training script or data | Leakage-safe grouping and actual assignments | Group-disjoint evaluator and generated recording plan | Record/verify clips with participant/group IDs | BLOCKED |
| Risk labeled dataset | 36,674 complete 22-feature candidates; 18,218 blank label rows | All positive/negative labels | Build 301-row review queue; derive labels from GT matches | Provide/verify video evidence | BLOCKED |
| Risk class distribution | 0 positive, 0 negative | At least 100/100 under this pilot protocol | Count labels automatically | Record/top up minority class | BLOCKED |
| Risk leakage audit | 0 duplicate event IDs; 15,888 repeated feature records beyond first | Source/scene/person groups for historical rows | Reject event-level split; keep each `group_id` in one split | Enter group IDs or use new recording plan | BLOCKED |
| Runtime risk threshold | Static `min_risk_score=0.65`; `gate_alerts=false`; no model file | Validated threshold | Select threshold on validation F1 after labels | None beyond verified labels/groups | BLOCKED |
| Risk Accuracy/Precision/Recall/F1/ROC-AUC | Evaluation code and blank status CSV | Any valid labeled test set | Train-only normalization; validation threshold; test metrics; CM/ROC SVG | Complete labels/GT | BLOCKED - no metric reported |
| Exact 22-dimensional vector | Names/order/formulas verified; all 36,674 rows complete | No data blocker for definition table | Export requested table with observed ranges | None | COMPLETE |
| Five behavior definitions | Current trigger code and thresholds verified | None for definitions | Export exact 5-row rule table | None | COMPLETE |
| Behavior quantitative metrics | Prediction counts exist: intrusion 3,511; loitering 214; suspicious 54; theft 2,874; historical line_crossing 506 | Continuous-video GT and negatives | Run pipeline, time-align, match, calculate metrics/latency | Record and verify 200 planned clips | BLOCKED - no metric reported |
| Manual experiment protocol | 200-row recording plan, prefilled annotation sheet, label queue | Physical recordings, ROI/line YAML, dual review | Automate all post-recording processing | Perform real-world recording/annotation only | READY FOR MANUAL WORK |

## Audit scope confirmed from the supplied documents

The `uu tien.docx` file requests Risk Assessment, the 22-dimensional vector, Behavior Analytics, and Dataset/Experimental Setup. The supplied revision-plan DOCX contains broader later work, but scalability, MOT benchmark, ablation, baseline comparison, and whole-paper editing were intentionally not executed. Neither DOCX was modified.

## Priority 1 - Dataset / Experimental Setup

- Verified local video dataset: 6 clips, 361.127 s (6.019 min). Four are referenced by active camera YAML and two by the phase-0 backup YAML.
- Current active subset: 4 clips, 284.061 s.
- Resolution range: 1048x582 to 1920x1080; FPS range: 17.948 to 30.
- Event log: 36,674 unique prediction records, 11 camera IDs, 11 predicted alert types, all with 22 features.
- Proven source mapping covers 32,684 records. Five camera IDs contributing 3,990 records have no provable current source-video mapping.
- SQLite is not a hidden GT source: `alerts=12,208`, `behavior_events=0`, `behavior_labels=0`, `video_clips=0`, `alert_clips=0`.
- Unique people, true event count/type, exact camera model, original data provenance/license, and train/test use are `NEEDS MANUAL INPUT`.
- Visually auditable conditions are documented in `DATASET_STATISTICS.md` and `tables/dataset_statistics.csv`. Watermarks are not treated as camera-model evidence.

## Priority 2 - Risk Assessment

### Current data

| Item | Verified result |
|---|---:|
| Candidate events | 36,674 |
| Rows with complete 22 features | 36,674 |
| Label CSV rows | 18,218 |
| Nonblank recognized labels | 0 |
| Positive | 0 |
| Negative | 0 |
| Existing model | absent (`models/behavior_classifier.npz`) |
| Current split | none |
| Metrics | not computable |

The technical evaluator precondition is 5 positive and 5 negative samples, so even that minimal code path is short by exactly 10 labels. For the defined pilot protocol, the actionable deficit is 100 positive + 100 negative emitted-event labels with at least five independent participant groups. The latter is the reporting target; 5+5 must not be presented as scientifically meaningful.

### Leakage and threshold findings

- The original trainer computes mean/std on every labeled row, trains on those same rows, and reports accuracy on the same rows at threshold 0.5 (`scripts/train_behavior_classifier.py:64-75,172-196`). That is in-sample training accuracy, not a test metric.
- Event IDs are unique, but only 20,786 distinct feature vectors exist; 15,888 records are repeated-vector copies beyond the first. Random event-level splitting would therefore be unsafe even before considering same-video temporal correlation.
- The runtime loads `min_risk_score=0.65`, but `gate_alerts` is false and no model exists (`config/settings.yaml:161-164`). The value 0.65 was not selected from validation data.
- The revised evaluator fits normalization on train only, keeps a `group_id` wholly in one partition, selects threshold on validation F1, then computes Accuracy/Precision/Recall/F1/ROC-AUC only on test.
- `risk_evaluation/risk_metrics.csv` contains blank metric cells and the blocker. No confusion matrix or ROC was created because both classes are absent; the evaluator will create CSV and SVG artifacts only after valid labels exist.

## Priority 3 - 22-dimensional Feature Vector

The code-authoritative order is:

`duration`, `threshold_seconds`, `duration_ratio`, `near_seconds`, `score`, `score_ratio`, `pacing_passes`, `object_confidence`, `bbox_area_ratio`, `path_length_ratio`, `net_distance_ratio`, `displacement_ratio`, `speed_ratio`, `has_vehicle_signal`, `vehicle_started_moving`, `moving_same_direction`, `pose_push_contact`, `object_count`, `zone_configured`, `is_person`, `is_vehicle`, `is_asset`.

Evidence: `analytics/behavior_learning.py:22-44,229-280`. Scoring applies `(x - mean) / scale` and sigmoid logistic regression (`analytics/behavior_learning.py:83-90`). `tables/risk_features_22.csv` contains the requested seven columns plus observed range from all 36,674 records.

The supplied paper plan does not enumerate feature names, so a name-by-name paper/code comparison is impossible. It only states that the table must be added. No paper wording was changed.

## Priority 4 - Behavior Analytics

Exact rules are in `tables/behavior_definitions.csv`. Critical implementation findings:

1. Current `IntrusionDetector.analyze()` calls only `_analyze_lines`; polygon `_analyze_zones` exists but is not invoked (`analytics/intrusion.py:70-92,102-158`). Current intrusion is therefore an inbound line-crossing alert, not polygon entry.
2. Current code does not emit a distinct `line_crossing` alert. It increments IN/OUT counters and emits `intrusion` only for IN (`analytics/intrusion.py:197-212,226-253`). Historical `line_crossing=506` records were generated by an older code path.
3. Loitering needs an explicit `loitering` polygon; auto-global zones cover intrusion/stranger-watch, not loitering (`analytics/behavior_engine.py:26,229-265`).
4. Suspicious behavior requires confirmed stranger + zone dwell + stationary or pacing geometry; it does not use pose keypoints.
5. Theft requires `asset_watch` zones at the BehaviorEngine call site, confirmed stranger, a person-vehicle proximity of 0.125 frame diagonal, at least two cues, mandatory 10 s near duration, and at least one vehicle cue.
6. The event log spans behavior-code versions: its first record is a polygon-zone `intrusion` for class `bicycle`, which current line-only/person-only logic cannot emit. Historical and newly recorded data must not be mixed without a code-version/group field.

Behavior metrics remain uncomputed because alert logs provide predictions but no false-negative intervals or verified negative clips. `behavior_evaluation/behavior_metrics.csv` intentionally contains blank metrics and `NEEDS MANUAL INPUT`.

## Artifacts created

- `raw/dataset_audit.json` and `raw/dataset_audit.csv`
- `DATASET_STATISTICS.md` and `tables/dataset_statistics.csv`
- `tables/risk_features_22.csv`
- `tables/behavior_definitions.csv`
- `risk_evaluation/risk_metrics.csv` and `risk_evaluation/evaluation_status.json`
- `behavior_evaluation/behavior_metrics.csv` and `behavior_evaluation/behavior_evaluation.json`
- `manual/recording_plan.csv` (200 clips)
- `manual/behavior_ground_truth_template.csv` (200 prefilled rows, all unverified)
- `manual/risk_label_queue.csv` (301 historical candidates; evidence required)
- `MANUAL_EXPERIMENT_PROTOCOL.md`
- Evaluation/automation scripts under `scripts/`

No metric, participant count, camera model, provenance, or ground-truth event count was invented.
