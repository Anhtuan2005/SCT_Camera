# SCT Camera Phase 0 Baseline Report

Generated: 2026-06-21 (Asia/Saigon)

## Environment

- OS: Windows 10 build 26200
- Python: 3.10.11
- Logical CPUs: 16
- RAM: 15,628.8 MB
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MB
- NVIDIA driver: 596.36
- Detection: `yolo11n.pt`, CUDA, image size 640
- Pose: `yolo11n-pose.pt`, CUDA, image size 640

## Workload

- Source type: local video
- Source: `people_720p25.mp4`
- Resolution: 1280 x 720
- Processing height: 480 from current runtime settings
- Warm-up: 5 frames per logical camera
- Measured duration: 30 seconds per case
- Two-camera case: two independent capture/tracker workers replaying the same local video
- Stages measured: tracking/detection, pose, identity labeling, behavior rules

Raw results: `data/benchmarks/baseline.json` and `data/benchmarks/baseline.csv` (local runtime data, ignored by Git).

## Results

| Cameras | Frames | Aggregate FPS | AI p50 | AI p95 | AI p99 | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 623 | 20.76 | 40.17 ms | 54.34 ms | 64.83 ms | 1,939.70 MB |
| 2 | 736 | 24.47 | 73.23 ms | 92.71 ms | 104.45 ms | 2,097.41 MB |

### Stage latency

| Cameras | Stage | p50 | p95 | p99 |
|---:|---|---:|---:|---:|
| 1 | Track/detect | 19.97 ms | 27.21 ms | 34.81 ms |
| 1 | Pose | 18.79 ms | 25.91 ms | 32.17 ms |
| 1 | Identity | 0.12 ms | 0.17 ms | 0.22 ms |
| 1 | Behavior | 1.15 ms | 1.70 ms | 2.45 ms |
| 2 | Track/detect | 34.17 ms | 46.42 ms | 52.91 ms |
| 2 | Pose | 36.82 ms | 51.46 ms | 57.78 ms |
| 2 | Identity | 0.13 ms | 0.17 ms | 0.21 ms |
| 2 | Behavior | 1.28 ms | 1.66 ms | 2.02 ms |

Identity latency is not a face-recognition benchmark. Local-video policy labels people without running the full live-camera face matching path. A later benchmark must use consenting face-test data if identity performance becomes an acceptance criterion.

RSS includes lazy loading and caching of YOLO, pose, and InsightFace models. One-camera RSS rose from 730.14 MB to a 1,939.70 MB peak. The warm two-camera case peaked at 2,097.41 MB.

## Phase 0 decisions

- DB timestamps: Unix milliseconds UTC.
- SQLite: one writer thread, bounded queue size 1000, short-lived read connections, WAL, foreign keys, 5-second busy timeout.
- DB evidence retention: no automatic deletion of alerts, deliveries, behavior events, or labels during thesis evaluation.
- Clip retention: quota-based; 10 GB default, 2 GB minimum free disk, active clip excluded from cleanup.
- Recording window: 10 seconds before event and 20 seconds after event.
- Corrupt DB: quarantine and enter degraded mode; never overwrite or silently create a replacement.
- Fall dataset: train/tuning and test must not share camera or person IDs; test minimum is 30 fall and 50 non-fall events.

## Validation status

- Dev dependencies: pass. `pip install -r requirements-dev.txt` completed.
- Existing test suite: pass. Final result: 72 passed in 4.47 seconds. Initial run exposed three stale pipeline fixtures; fixtures were updated for the new parameter snapshot API.
- Backup: pass. `data/backups/phase0_20260621_155522` contains 7 files; YAML and JSONL parsed successfully; SHA-256 checksums recorded.
- Baseline: pass. Concrete 1-camera and 2-camera results recorded above.
- Migration bootstrap/rollback test: pass. Three migrations apply once, reruns are idempotent, and an intentionally broken migration rolls back both DDL and version marker.
- Fall dataset manifest: blocked. `docs/fall_dataset_manifest.json` records missing `ground_truth.csv`, 0 test fall events, and 0 test non-fall events.

Phase 1 gate remains closed until all six checks pass.
