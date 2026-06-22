CREATE TABLE IF NOT EXISTS behavior_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL UNIQUE,
    alert_type      TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    track_id        INTEGER,
    features        TEXT,
    risk_score      REAL,
    suppressed      INTEGER NOT NULL DEFAULT 0,
    timestamp       INTEGER NOT NULL,
    created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS behavior_labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL UNIQUE
                REFERENCES behavior_events(event_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    notes       TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
