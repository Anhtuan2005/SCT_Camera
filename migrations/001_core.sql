CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL UNIQUE,
    alert_type      TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    camera_name     TEXT NOT NULL DEFAULT '',
    track_id        INTEGER,
    class_name      TEXT,
    zone_id         TEXT,
    zone_name       TEXT,
    line_id         TEXT,
    line_name       TEXT,
    details         TEXT,
    timestamp       INTEGER NOT NULL,
    siren           INTEGER NOT NULL DEFAULT 0,
    suppressed      INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_camera_ts
    ON alerts(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_alerts_type_ts
    ON alerts(alert_type, timestamp);

CREATE TABLE IF NOT EXISTS notification_deliveries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id    INTEGER NOT NULL
                REFERENCES alerts(id) ON DELETE CASCADE,
    channel     TEXT NOT NULL
                CHECK(channel IN ('telegram', 'discord')),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'sent', 'failed', 'suppressed')),
    error       TEXT,
    sent_at     INTEGER,
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nd_alert
    ON notification_deliveries(alert_id);
