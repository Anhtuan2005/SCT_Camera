# Frozen development diagnostic — single-participant pilot

## Scope

- Frozen version: `priority-pilot-v2-single-participant-2026-08-18`.
- Ten development clips (`r1`) were run separately from the immutable ten-clip test split (`r2`).
- The source set is now complete at 20/20 clips.
- This output is diagnostic only. It is not merged with the held-out test metrics and was not used to retune ROI, thresholds, implementation, or ground truth.

## Development results

| Behavior | TP | FP | TN | FN | Detection rate | FAR (negative clip) | Precision | Recall | F1 | Median latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intrusion | 1 | 1 | 1 | 0 | 1.000 | 0.000 | 0.500 | 1.000 | 0.666667 | 3.566 |
| loitering | 0 | 1 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| suspicious_behavior | 1 | 0 | 1 | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.185 |
| theft | 0 | 1 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| line_crossing | 1 | 4 | 0 | 0 | 1.000 | 1.000 | 0.200 | 1.000 | 0.333333 | 3.522 |

## Diagnostic observations

- **Intrusion:** the annotated `IN` event was detected, with one additional early `IN` event on the positive clip. The negative `OUT` clip produced no intrusion alert.
- **Loitering:** the system emitted at 26.899 s, 0.101 s before the independently annotated 27.0 s eligibility time. The strict evaluator therefore records the event as FP and the annotated event as FN. The negative clip produced no alert.
- **Suspicious behavior:** the positive event was detected at 180.185 s and the negative clip produced no alert.
- **Theft:** one event was emitted at 16.822 s, before the annotated 20.0 s theft interval; it is an FP and the eligible event remains an FN. The negative clip produced no alert.
- **Line crossing:** the annotated `OUT` crossing was detected. There was one additional early event on the positive clip and three false `OUT` events on the negative clip.

## Cross-split interpretation

- Suspicious behavior is the only behavior with TP=1, TN=1, FP=0, and FN=0 in both repetitions.
- Intrusion detects the positive `IN` crossing and rejects the negative `OUT` clip in both repetitions, but event duplication/early events reduce precision.
- Theft does not match the annotated positive interval in either repetition.
- Line crossing detects the positive crossing in both repetitions but raises a false alarm on the negative clip in both repetitions.
- Loitering passes the held-out test repetition; the development repetition is a strict near-boundary timing mismatch.

These observations are from one participant and two repetitions only. They do not establish participant-level generalization or full-scale validation.

## Evidence

- `run_summary.csv`: video/config hashes, duration, frame count, and runtime.
- `detections.csv`: raw frozen development output.
- `risk_events.jsonl`: emitted 22-feature records.
- `evaluation/behavior_metrics.csv`: computed development metrics.
- `evaluation/behavior_event_matches.csv`: exact TP/FP/FN assignments and timestamps.
