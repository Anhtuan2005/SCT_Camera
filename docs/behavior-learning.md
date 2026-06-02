# Behavior Learning Workflow

The runtime uses rules first, then logs every behavior alert candidate for supervised learning.

## 1. Collect events

Run the camera system normally. Candidate events are appended to:

```text
data/behavior_events.jsonl
```

Each line contains:

- `event_id`: stable id to label
- `alert_type`: rule that produced the candidate
- `camera_id`, `zone_id`, `track_id`
- `features`: numeric behavior features
- `label`: empty until labeled

## 2. Label events

Create a CSV:

```text
data/behavior_labels.csv
```

Format:

```csv
event_id,label,notes
<event id>,false_alarm,walked past the gate
<event id>,theft_attempt,staged bike removal
```

You can also inspect and label through the API:

```text
GET /api/behavior-events?limit=100
POST /api/behavior-events/{event_id}/label
```

POST body:

```json
{"label": "false_alarm", "notes": "walked past the gate"}
```

Supported negative labels:

```text
normal, benign, false_alarm, false_positive, negative, ignore
```

Supported positive labels:

```text
suspicious, theft, theft_attempt, asset_removed, asset_missing, loitering_bad, positive, true_positive
```

## 3. Train

```powershell
.\.venv\Scripts\python.exe scripts\train_behavior_classifier.py --labels data\behavior_labels.csv
```

The model is saved to:

```text
models/behavior_classifier.npz
```

## 4. Use the model

Restart the app after training, or leave it running and the runtime will reload the model when the file changes.

By default, the model only adds `behavior_risk_score` to alerts. It does not block alerts.

To let the model suppress low-risk alerts, set:

```yaml
behavior_learning:
  gate_alerts: true
  min_risk_score: 0.65
```
