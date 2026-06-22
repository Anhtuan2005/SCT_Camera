CREATE TABLE IF NOT EXISTS video_clips (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id           TEXT NOT NULL,
    filename            TEXT NOT NULL UNIQUE,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'ready', 'failed')),
    start_time          INTEGER NOT NULL,
    end_time            INTEGER,
    duration_seconds    REAL,
    file_size_bytes     INTEGER,
    created_at          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vc_camera_start
    ON video_clips(camera_id, start_time);

CREATE TABLE IF NOT EXISTS alert_clips (
    alert_event_id  TEXT NOT NULL
                    REFERENCES alerts(event_id) ON DELETE CASCADE,
    clip_id         INTEGER NOT NULL
                    REFERENCES video_clips(id) ON DELETE CASCADE,
    PRIMARY KEY (alert_event_id, clip_id)
);
