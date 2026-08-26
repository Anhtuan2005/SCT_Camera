# Dataset Statistics (code/data audit, 2026-08-18)

## Paper-ready table with verified fields only

| Source | Clips | Duration | Resolution / FPS | Participants | Ground-truth events | Lighting | Camera view | Camera model | Train/Val/Test |
|---|---:|---:|---|---|---|---|---|---|---|
| Current config-referenced local videos | 4 | 284.061 s | 1280x720-1920x1080 / 17.948-30 FPS | NEEDS MANUAL INPUT | 0 labeled | indoor bright, indoor low/artificial, outdoor daylight | fixed high/oblique; one raw clip rotated 90 degrees | NEEDS MANUAL INPUT | not assigned |
| Historical-backup config-referenced local videos | 2 | 77.066 s | 1048x582-1050x584 / 30 FPS | NEEDS MANUAL INPUT | 0 labeled | monochrome/IR and mixed low-light night | high-oblique plus one multi-scene composite | NEEDS MANUAL INPUT | not assigned |
| **Verified video total** | **6** | **361.127 s (6.019 min)** | mixed | **NEEDS MANUAL INPUT** | **0 labeled** | mixed | mixed | **NEEDS MANUAL INPUT** | **no protocol** |
| Runtime behavior candidate log | not recoverable from log | NEEDS MANUAL INPUT | not stored | NEEDS MANUAL INPUT | 0 labeled / 36,674 predictions | NEEDS MANUAL INPUT | NEEDS MANUAL INPUT | NEEDS MANUAL INPUT | no protocol |

## Event-log statistics (predictions, not ground truth)

| Item | Verified value |
|---|---:|
| JSONL records | 36,674 |
| Unique event IDs | 36,674 |
| Duplicate event IDs | 0 |
| Records with all 22 features | 36,674 |
| Camera IDs | 11 |
| Predicted alert types | 11 |
| Timestamp span | 2026-06-10T22:11:28+07:00 to 2026-08-15T10:14:27+07:00 |
| External label rows | 18,218 |
| Nonblank labels | 0 |
| DB video clips / alert clips | 0 / 0 |

Predicted alert counts: `stranger_detected` 25,577; `intrusion` 3,511; `suspicious_theft_behavior` 2,874; `asset_missing` 2,842; `possible_fall` 568; `line_crossing` 506; `possible_unresponsive` 427; `loitering` 214; `fall_recovery` 100; `suspicious_stranger` 54; `asset_removed` 1.

These counts cannot be presented as true events because no ground truth exists. Six video sources account for 32,684 event records by proven config mapping; the source video for 3,990 records across five other camera IDs cannot be proven from the current project and is `NEEDS MANUAL INPUT`.

Original acquisition source/license, exact camera model, unique participant count, true event count/type, and train/validation/test assignment remain `NEEDS MANUAL INPUT`. Brand/watermark overlays visible in frames are not camera-model evidence.

Detailed per-clip fields and evidence links are in `tables/dataset_statistics.csv`; raw hashes and observed feature ranges are in `raw/dataset_audit.json`.
