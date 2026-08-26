# FDSE 2026 Revision Experiments

This file records reproducible commands and the scientific limits of the available data. Run all commands from the repository root with the project virtual environment.

## 1. Dataset audit

```powershell
.\.venv\Scripts\python.exe scripts\audit_revision_dataset.py `
  --video "E:\Recording 2026-08-13 230208.mp4" `
  --video "E:\65ac3594-5c95-4908-909b-3ae28db47e60.mp4" `
  --video "E:\7856352689065780164.mp4" `
  --video "E:\Recording 2026-07-11 113042.mp4" `
  --video "E:\people_720p25.mp4" `
  --json-out experiments\results\dataset_audit.json `
  --csv-out experiments\results\dataset_audit.csv
```

Verified on 2026-08-17:

- 36,674 unique JSONL events; no invalid JSON or duplicate `event_id`.
- All 36,674 records contain the complete 22-dimensional feature mapping.
- `behavior_labels.csv` contains 18,218 event IDs, but every label is blank; the remaining 18,456 events have no label row.
- No `models/behavior_classifier.npz` exists.
- Four configured local videos are readable; the historical benchmark source `people_720p25.mp4` is missing.

## 2. Method tables

```powershell
.\.venv\Scripts\python.exe scripts\export_revision_tables.py --out-dir experiments\tables
```

Outputs:

- `experiments/tables/risk_features_22.csv`
- `experiments/tables/behavior_rules.csv`
- `experiments/tables/historical_failure_analysis.csv`

The exporter fails if the documented feature order differs from `analytics.behavior_learning.FEATURE_NAMES`.

## 3. Human labeling queue

```powershell
.\.venv\Scripts\python.exe scripts\build_behavior_label_queue.py `
  --per-type 30 `
  --out experiments\labeling\behavior_event_queue.csv
```

The current queue contains 301 events sampled deterministically across all 11 logged alert types and across their observed time ranges. Annotators must review the corresponding video evidence, fill `risk_label` with `true_positive` or `false_alarm`, and record `video_evidence_path`, `annotator`, and notes. Do not label from alert text alone.

Important limitation: a queue of emitted alerts can estimate precision/false-alarm proportion, but it cannot estimate false negatives or recall. Recall/F1 for each behavior requires continuous video intervals annotated with both events and non-events.

## 4. Risk-model evaluation

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_behavior_risk.py `
  --out-dir experiments\results\risk_evaluation
```

The evaluator uses a deterministic stratified train/validation/test split, fits normalization on the training split only, selects the threshold by validation F1, and writes predictions, threshold sensitivity, ROC points, a confusion matrix, metrics, and the model. Duplicate `event_id` values are rejected.

Current status is intentionally `blocked`: zero positive and zero negative labels. The default runtime threshold `min_risk_score = 0.65` is a configuration value, not a threshold justified by validation data.

## 5. Repeated runtime/scalability benchmark

```powershell
.\.venv\Scripts\python.exe scripts\run_revision_benchmark.py `
  --source "E:\7856352689065780164.mp4" `
  --camera-config config\cameras\motorcycle_016324.yaml `
  --cameras 1 2 4 `
  --duration 15 `
  --warmup-frames 30 `
  --repetitions 3 `
  --pose auto `
  --out-dir experiments\results\runtime_2026-08-17
```

Protocol: NVIDIA RTX 3050 Laptop GPU (4 GB), 1280x720 local video, batch size 1, YOLO11s detector, YOLO11n-Pose, `imgsz=640`, 30 warm-up frames, 15 measured seconds per camera-count case, three repetitions.

| Cameras | Aggregate FPS, mean ± SD | FPS/camera, mean ± SD | AI p50 ms, mean ± SD | AI p95 ms, mean ± SD | AI p99 ms, mean ± SD | Peak RSS MB, mean ± SD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.529 ± 1.029 | 26.529 ± 1.029 | 35.152 ± 0.694 | 42.844 ± 5.979 | 49.014 ± 7.022 | 1985.272 ± 11.487 |
| 2 | 27.671 ± 0.686 | 13.836 ± 0.343 | 68.011 ± 1.203 | 85.593 ± 5.683 | 98.117 ± 10.887 | 2136.724 ± 11.332 |
| 4 | 27.645 ± 1.572 | 6.911 ± 0.393 | 138.748 ± 5.262 | 172.825 ± 20.125 | 186.190 ± 31.246 | 2415.510 ± 11.062 |

`aggregate_fps` is total throughput across workers. It must not be presented as per-camera throughput. Aggregate throughput saturates near 27.6 FPS while per-camera throughput and latency degrade as workers share the serialized detector/pose inference lock.

The historical 20.76/24.47 FPS artifact used YOLO11n and a different video that is now missing. It is not directly comparable with this current YOLO11s benchmark.

Generate the paper-ready chart:

```powershell
.\.venv\Scripts\python.exe scripts\plot_revision_runtime.py `
  experiments\results\runtime_2026-08-17\summary.json `
  --out experiments\figures\runtime_scalability.png
```

## 6. Regression and historical failure reproduction

Current suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The five historical identity failures were reproduced from commit `2d0f9bb2` and are listed in `historical_failure_analysis.csv`. Two shared causes explain all five:

1. Unassessed tracks incremented `_failed_attempt_counts`, causing premature `stranger` classification.
2. `pending_person` bypassed `recognition_interval`, exhausting confirmation attempts on consecutive frames.

The current implementation increments failures only for `assessed_track_ids` and respects `recognition_interval`.

Latest complete run on 2026-08-17: 198 passed, 0 failed, with three Starlette deprecation warnings.

## 7. Metric feasibility

| Metric group | Current feasibility | Required evidence |
|---|---|---|
| Risk Accuracy/Precision/Recall/F1/ROC-AUC | Blocked | Positive and negative event labels with video evidence and leakage-safe grouping |
| Per-behavior precision | Partially feasible after alert review | Human labels for emitted alerts |
| Per-behavior recall/FN/F1 | Blocked | Continuous video ground truth including missed events |
| Detection Precision/Recall/mAP | Blocked | Box/class annotations for the evaluation frames |
| MOTA/IDF1/HOTA/ID switches | Blocked | MOT-format ground-truth trajectories |
| Identity accuracy/inheritance | Blocked | Consented identity ground truth and association annotations |
| Runtime/scalability | Completed for 1/2/4 cameras | Raw JSON/CSV and summary artifacts are present |
