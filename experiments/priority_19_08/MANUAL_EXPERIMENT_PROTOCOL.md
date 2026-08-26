# Manual experiment protocol - Priority requirements only

## 1. Blockers proven by the audit

- Risk labels currently available: **0 positive, 0 negative**. `data/behavior_labels.csv` has 18,218 rows, but every label is blank.
- Behavior ground truth currently available: **0 verified positive events and 0 verified negative clips** for all five required behaviors.
- The SQLite database has 0 `behavior_events`, 0 `behavior_labels`, 0 `video_clips`, and 0 `alert_clips`; therefore existing alert timestamps cannot be aligned reliably to source-video time.
- Six config-referenced clips exist (361.127 s total), but participant count, original provenance/license, camera model, and ground-truth event intervals are not recorded.
- The current runtime treats every local video-file source as `stranger`; offline negatives must therefore be based on motion/spatial evidence, not a claimed known-person identity.

## 2. Exact minimum work defined for this experiment

The generated `manual/recording_plan.csv` contains **200 clips**:

| Behavior | Positive clips missing now | Negative clips missing now | Total | Duration per positive / negative |
|---|---:|---:|---:|---|
| intrusion | 20 | 20 | 40 | 25 s / 25 s |
| loitering | 20 | 20 | 40 | 35 s / 25 s |
| suspicious behavior | 20 | 20 | 40 | 200 s / 200 s |
| theft | 20 | 20 | 40 | 35 s / 35 s |
| line crossing | 20 | 20 | 40 | 25 s / 25 s |
| **Total** | **100** | **100** | **200** | about **198.3 min** of raw video |

Design: 5 participants (`P01`-`P05`) x 2 lighting conditions x 2 viewpoints x positive/negative. `P01`-`P03` are train, `P04` is validation, and `P05` is test. This is a project-defined minimum pilot protocol, not a statistical-power guarantee.

Risk evaluation has a separate floor of **100 positive and 100 negative labeled emitted events**, so the current exact deficit is **100 positive + 100 negative labels**. Labels will be derived automatically from matched behavior ground truth. If the 200 clips do not emit at least 100 events in each class, record additional minority-class/hard-negative clips until both counts reach 100.

## 3. Camera and recording setup

For every row in `manual/recording_plan.csv`:

1. Use a fixed camera, 2.5-3 m high, tilted downward 20-35 degrees. Do not pan or zoom during a clip.
2. Record at 1280x720 or higher and 25-30 FPS. Keep the full person and the relevant line/ROI/vehicle visible.
3. `daylight`: normal daylight or bright room. `low_light`: night/artificial light while the person and object remain identifiable.
4. `frontal_oblique` and `side_oblique` must be two genuinely different fixed viewpoints.
5. Save the video at the exact `video_path` in the recording plan.
6. Create the exact YAML named in `camera_config_path`:
   - intrusion/line crossing: draw one `lines` entry with normalized endpoints and correct `direction`;
   - loitering: draw a `zones` entry with `type: loitering`, `threshold_seconds: 20`;
   - suspicious behavior: draw `type: stranger_watch`, `threshold_seconds: 180`;
   - theft: draw `type: asset_watch` covering both person and vehicle;
   - set `auto_global_zone: false` so the evaluated ROI/line is explicit.
7. After checking that video and YAML open correctly, change the plan row `status` from `TO_RECORD` to `RECORDED`.

## 4. Required action in each clip

- **Intrusion positive:** person crosses the line once in the configured IN direction. Negative: approaches/turns before the line or crosses only OUT.
- **Loitering positive:** person stays in ROI for at least 25 s (20 s threshold). Negative: exits before 15 s.
- **Suspicious positive:** stranger stands nearly still or paces for at least 190 s. Negative: stranger moves purposefully without the stationary/pacing geometry.
- **Theft positive:** stranger remains near the watched vehicle for at least 10 s and moves/pushes it. Negative: remains nearby but does not create any vehicle-motion/same-direction/wrist-contact cue.
- **Line crossing positive:** person crosses exactly once; include both IN and OUT cases across the 20 positives. Negative: walk parallel, touch, or turn before the line without crossing.

## 5. Ground-truth annotation

Use `manual/behavior_ground_truth_template.csv`, which already contains all 200 planned rows.

- `start_seconds`: first moment the actor begins the behavior episode.
- `eligible_time_seconds`: first moment the code is allowed to trigger:
  - intrusion/line crossing: geometric crossing moment;
  - loitering: ROI entry + effective threshold;
  - suspicious behavior: dwell threshold reached while stationary/pacing condition is true;
  - theft: first time score >=2, near-duration is true, and at least one vehicle cue is true.
- `end_seconds`: end of the event/evaluation window.
- For negative clips, keep the three time fields blank.
- First annotator fills `annotator`; a second person checks video and fills `reviewer`.
- Change `annotation_status` to `VERIFIED` only after both agree. The evaluator ignores every other status.
- Do not use system alert text as ground truth.

## 6. Automated execution after recording

Run from `E:\SCT_Camera`:

```powershell
.\.venv\Scripts\python.exe scripts\run_behavior_experiment.py `
  --manifest experiments\priority_19_08\manual\recording_plan.csv `
  --detections-out experiments\priority_19_08\behavior_evaluation\detections.csv `
  --summary-out experiments\priority_19_08\behavior_evaluation\run_summary.csv `
  --risk-events-out experiments\priority_19_08\risk_evaluation\risk_events.jsonl

.\.venv\Scripts\python.exe scripts\evaluate_behavior_experiment.py `
  --ground-truth experiments\priority_19_08\manual\behavior_ground_truth_template.csv `
  --detections experiments\priority_19_08\behavior_evaluation\detections.csv `
  --out-dir experiments\priority_19_08\behavior_evaluation `
  --split test

.\.venv\Scripts\python.exe scripts\derive_risk_labels_from_behavior.py `
  --detections experiments\priority_19_08\behavior_evaluation\detections.csv `
  --matches experiments\priority_19_08\behavior_evaluation\behavior_event_matches.csv `
  --out experiments\priority_19_08\risk_evaluation\risk_labels.csv

.\.venv\Scripts\python.exe scripts\evaluate_behavior_risk.py `
  --events experiments\priority_19_08\risk_evaluation\risk_events.jsonl `
  --labels experiments\priority_19_08\risk_evaluation\risk_labels.csv `
  --out-dir experiments\priority_19_08\risk_evaluation
```

The runner uses video time for dwell rules, exports detections and the exact 22-feature risk records, and does not append to the production event log. Risk splitting is group-disjoint by participant (`group_id`), normalization is fit only on train, threshold is selected on validation F1, and final metrics are computed only on test.

## 7. Metric definitions used by the scripts

- Detection Rate = TP / (TP + FN), event-level.
- False Alarm Rate = negative clips containing at least one FP / all negative clips.
- Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 is their harmonic mean.
- Detection latency = first matched detection time - `eligible_time_seconds`; report median and p95.
- Risk metrics: Accuracy, Precision, Recall, F1, and ROC-AUC on the group-disjoint test partition.
- Risk threshold: maximum validation F1; tie-break by recall, then distance to 0.5.

## 8. Existing queue warning

`manual/risk_label_queue.csv` contains 301 sampled historical events. Label a row only when `video_evidence_path` and exact evidence time can be supplied. Because historical clip linkage is currently absent, blank evidence rows must remain unlabeled and must not be used for metrics.
