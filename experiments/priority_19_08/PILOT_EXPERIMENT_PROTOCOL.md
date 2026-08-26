# Single-Participant Deadline-Safe Pilot Protocol

Experiment design version: `priority-pilot-v2-single-participant-2026-08-18`

This is a single-participant pilot, not full-scale validation.

## Fixed design

- Participant: P01 only.
- Five behaviors: intrusion, loitering, suspicious_behavior, theft, line_crossing.
- Positive and negative cases for every behavior.
- Two independently recorded repetitions per behavior × label: 20 clips total.
- Repetition 1: 10 `development` clips in four development recording sessions.
- Repetition 2: 10 held-out `test` clips in four different recording sessions.
- No validation split and no threshold tuning.
- Total planned raw recording duration: 1,260 seconds = 21 minutes.
- Exactly 10 daylight/10 low-light clips and 10 frontal-oblique/10 side-oblique clips.
- Joint lighting × viewpoint cells are 6/4/4/6, while every behavior-label case flips both lighting and viewpoint between repetitions; this avoids label/lighting confounding.

The authoritative filename, repetition, session, action, duration, split, video path and YAML path are in `manual/recording_plan_pilot.csv`.

## What the split means

This is not participant-disjoint evaluation: P01 appears in both development and test. Do not use language such as “generalizes to unseen people.”

`group_id=session_id`. All clips made without changing the camera/lighting setup belong to the same session. Session IDs are disjoint between development and test. Development clips may be used to verify that the pipeline, ROI/line and annotation workflow work, but they must not be used to modify behavior thresholds. Reported pilot behavior metrics come from `split=test` only.

## Frozen thresholds

- Intrusion: no dwell threshold; confirmed person with at least two center-history points; IN only emits intrusion.
- Loitering: 20 seconds inside ROI; 3-second state grace.
- Suspicious behavior: 180 seconds; at least 5 history points; stationary displacement <=0.04 diagonal, or pacing path >=0.18 with net <=0.08.
- Theft: near vehicle for at least 10 seconds; score >=2; near-duration plus at least one vehicle/pose cue are mandatory.
- Line crossing: no dwell threshold; center crossing or new line/bbox contact; both IN and OUT counted.

Do not lower or tune these values. The machine-readable snapshot is `manual/pilot_behavior_thresholds.yaml`.

## Recording-session workflow

There are eight session folders: four development and four test. Complete development first. Record test sessions later as new takes; re-mount/reconfirm the camera instead of copying a development clip.

For each session:

1. Use the lighting and viewpoint encoded in `session_id`.
2. Fix the camera 2.5–3 m high with a 20–35 degree downward tilt, at 1280×720 or higher and 25–30 FPS.
3. Use a fresh test frame to confirm the YAML ROI/line for every assigned clip. Coordinates may be copied within the same unchanged session, but each YAML must be marked confirmed.
4. If the camera or lighting changes, end the current session and reconfirm before recording under another session ID.
5. Record every assigned clip as a new continuous take. Never duplicate, trim from, or reuse a development take as test data.

## Per-clip manual work

1. Follow the exact action and minimum duration in `recording_plan_pilot.csv`.
2. For intrusion/line_crossing, edit only line coordinates/direction/rotation and retain image-left→right as IN, image-right→left as OUT.
3. For loitering/suspicious_behavior/theft, replace the placeholder polygon with the physical test area while retaining its type and threshold.
4. Set YAML `manual_confirmation.status: CONFIRMED` and CSV `roi_line_confirmation=CONFIRMED`.
5. Save under the exact `video_path`/`filename`; verify the clip opens and meets the duration; change CSV `status` to `RECORDED`.

Behavior reminders:

- Intrusion positive is IN; negative is OUT only.
- Line-crossing positive is OUT; negative keeps the entire bbox clear of the line.
- Loitering positive stays inside >=22 seconds; negative stays under 15 seconds.
- Suspicious positive preserves the real 180-second threshold; negative passes purposefully through and remains outside.
- Theft positive preserves the 10-second proximity gate and moves/pushes the vehicle; negative keeps it stationary and avoids vehicle/pose cues.

## Ground-truth annotation

Use `manual/behavior_ground_truth_pilot.csv`; all timing fields are intentionally blank.

For positive clips, annotate observed `start_seconds`, earliest valid `eligible_time_seconds`, and `end_seconds`. For negatives, leave event/timing fields blank and verify that the target behavior did not occur. Two people must fill `annotator` and `reviewer`, set `config_confirmed=YES`, and set `annotation_status=VERIFIED` only after agreement.

- Intrusion/line crossing: eligible at the observed configured-direction crossing.
- Loitering: observed ROI entry plus 20 seconds of valid dwell.
- Suspicious behavior: watched-zone entry plus 180 seconds while its geometry condition is satisfied.
- Theft: when 10-second near-duration and the mandatory additional vehicle/pose cue are both satisfied.

## Run and evaluate

From `E:\SCT_Camera` after recording and annotation:

```powershell
.\.venv\Scripts\python.exe scripts\run_behavior_experiment.py `
  --manifest experiments\priority_19_08\manual\recording_plan_pilot.csv `
  --settings config\settings.yaml `
  --freeze-file experiments\priority_19_08\PILOT_EXPERIMENT_FREEZE.yaml `
  --detections-out experiments\priority_19_08\pilot_run\detections.csv `
  --summary-out experiments\priority_19_08\pilot_run\run_summary.csv `
  --risk-events-out experiments\priority_19_08\pilot_run\risk_events.jsonl
```

Evaluate held-out test sessions only:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_behavior_experiment.py `
  --ground-truth experiments\priority_19_08\manual\behavior_ground_truth_pilot.csv `
  --detections experiments\priority_19_08\pilot_run\detections.csv `
  --out-dir experiments\priority_19_08\pilot_evaluation `
  --split test
```

Do not use historical logs or the previous pilot plan. Do not regenerate this experiment under the same version after recording begins; any implementation/settings/model change requires a new version and new recordings.
