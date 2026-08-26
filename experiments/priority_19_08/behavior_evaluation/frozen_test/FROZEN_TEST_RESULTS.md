# Frozen test results — single-participant deadline-safe pilot

## Scope and integrity

- Frozen experiment version: `priority-pilot-v2-single-participant-2026-08-18`.
- This is a **single-participant pilot**, not full-scale validation.
- Test data: 10 clips from P01, comprising one positive and one negative clip for each of five behaviors.
- Development repetition (`r1`) and test repetition (`r2`) are kept separate at clip/session-group level. They are not participant-disjoint because only P01 was available.
- Video and camera-config SHA-256 hashes were recorded in `run_summary.csv`.
- Production behavior implementation and frozen thresholds were not changed. Test output was evaluated without post-test ROI, threshold, or ground-truth tuning.
- Ground truth was annotated from visual evidence before metric interpretation. Missing development data is excluded rather than treated as a negative result.

## Test-set results

| Behavior | Positive events | Negative clips | TP | FP | TN | FN | Detection rate | FAR (negative clip) | Precision | Recall | F1 | Median latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intrusion | 1 | 1 | 1 | 5 | 1 | 0 | 1.000 | 0.000 | 0.166667 | 1.000 | 0.285714 | 0.457 |
| loitering | 1 | 1 | 1 | 0 | 1 | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.725 |
| suspicious_behavior | 1 | 1 | 1 | 0 | 1 | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.186 |
| theft | 1 | 1 | 0 | 1 | 0 | 1 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | N/A |
| line_crossing | 1 | 1 | 1 | 2 | 0 | 0 | 1.000 | 1.000 | 0.333333 | 1.000 | 0.500000 | 4.084 |

Definitions:

- `FP` counts unmatched emitted events, including duplicate events on a positive clip.
- `FAR (negative clip)` is the fraction of negative clips with at least one false alert.
- Latency is measured from the independently annotated eligible time to the first matched detection.
- No aggregate score is reported because each behavior has only one positive event and one negative test clip.

## Observations that must not be hidden

- **Intrusion:** the physical `IN` crossing was detected after 0.457 s, and the negative `OUT` clip produced no intrusion alert. Five additional `IN` events were emitted for the same positive crossing, which lowers event-level precision.
- **Loitering:** the positive clip was detected after the frozen 20-second dwell, with no alarm on the negative clip.
- **Suspicious behavior:** the positive clip was detected after the frozen 180-second dwell, with no alarm on the negative clip.
- **Theft:** the positive theft clip was missed and the negative clip emitted one alert at 20.034 s. The recorded risk evidence for the false alert includes `near_seconds=10.0`, `pose_push_contact=1.0`, and `vehicle_started_moving=0.0`; this points to erroneous pose/contact evidence, not vehicle movement.
- **Line crossing:** the positive `OUT` crossing was matched at 12.084 s. An earlier unmatched event occurred at 7.226 s on the positive clip, and the negative clip emitted a false `OUT` crossing at 10.278 s.

## Evidence

- `run_summary.csv`: processed clips, hashes, frames, FPS, durations, detection counts, and runtime.
- `detections.csv`: raw frozen runner output.
- `risk_events.jsonl`: 22-feature evidence attached to emitted risk events.
- `evaluation/behavior_metrics.csv`: computed per-behavior metrics.
- `evaluation/behavior_event_matches.csv`: TP/FP/FN event matching and latency.
- `evaluation/behavior_evaluation_outcomes.png` / `.pdf`: corrected paper-ready outcome figure that separates event counts from negative-clip false-alert status.
- `evaluation/behavior_outcome_components.csv`: source table splitting unmatched positive-clip events from false-alert events on negative clips.
- `../../manual/behavior_ground_truth_pilot.csv`: independent pilot ground-truth annotations and status.

## Ingestion status

All 20 pilot clips are now present in the canonical dataset. The previously unavailable low-light negative loitering development clip was reattached, decoded, hashed, and ingested successfully. Its addition did not alter or rerun the frozen test evaluation.

## Scientific limitation

These numbers describe only a deadline-safe pilot with one person and one positive/negative test clip per behavior. They cannot support claims of participant-level generalization, participant-disjoint validation, or full-scale system performance.
