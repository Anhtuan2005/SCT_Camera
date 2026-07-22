"""SQLite bootstrap and transactional schema migrations."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any


MIGRATION_NAME = re.compile(r"^(\d+)_.*\.sql$")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number if number >= 100_000_000_000 else number * 1000)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            number = float(raw)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return _now_ms()
        else:
            return int(number if number >= 100_000_000_000 else number * 1000)
    else:
        return _now_ms()
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return int(parsed.timestamp() * 1000)


class DatabaseManager:
    """Own SCT Camera schema bootstrap and migration execution."""

    def __init__(
        self,
        db_path: Path,
        migrations_dir: Path | None = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.migrations_dir = migrations_dir or Path(__file__).resolve().parents[1] / "migrations"
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))

    def bootstrap(self) -> None:
        """Create migration metadata tables before querying schema versions."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._bootstrap(conn)

    def run_migrations(self) -> list[int]:
        """Apply each pending numbered SQL migration transactionally."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        applied_now: list[int] = []
        with closing(self._connect()) as conn:
            self._bootstrap(conn)
            applied = {
                int(row[0])
                for row in conn.execute("SELECT version FROM schema_migrations")
            }
            for version, path in self._migration_files():
                if version in applied:
                    continue
                self._run_migration(conn, version, _parse_sql(path.read_text(encoding="utf-8")))
                applied_now.append(version)
        return applied_now

    def persist_alert(
        self,
        record: dict[str, Any],
        deliveries: list[dict[str, Any]],
    ) -> None:
        """Persist one processed alert and its notification outcomes atomically."""
        event_id = str(record.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Alert event_id is required")
        created_at = _now_ms()
        details = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO alerts(
                        event_id, alert_type, camera_id, camera_name, track_id,
                        class_name, zone_id, zone_name, line_id, line_name,
                        details, timestamp, siren, suppressed, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        str(record.get("type") or record.get("alert_type") or "alert"),
                        str(record.get("camera_id") or "unknown"),
                        str(record.get("camera_name") or ""),
                        record.get("track_id"),
                        record.get("class_name"),
                        record.get("zone_id"),
                        record.get("zone_name"),
                        record.get("line_id"),
                        record.get("line_name"),
                        details,
                        _timestamp_ms(record.get("timestamp")),
                        int(bool(record.get("siren"))),
                        int(bool(record.get("suppressed"))),
                        created_at,
                    ),
                )
                alert_id = int(cursor.lastrowid)
                for delivery in deliveries:
                    status = str(delivery.get("status") or "failed")
                    sent_at = delivery.get("sent_at")
                    if sent_at is None and status == "sent":
                        sent_at = created_at
                    conn.execute(
                        """
                        INSERT INTO notification_deliveries(
                            alert_id, channel, status, error, sent_at, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert_id,
                            str(delivery.get("channel") or ""),
                            status,
                            delivery.get("error"),
                            _timestamp_ms(sent_at) if sent_at is not None else None,
                            created_at,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get_recent_alerts(
        self,
        camera_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Read recent persisted alerts using a short-lived connection."""
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM alerts
                WHERE camera_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (str(camera_id), limit, offset),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            try:
                record = json.loads(row["details"] or "{}")
            except (TypeError, json.JSONDecodeError):
                record = {}
            record.setdefault("event_id", row["event_id"])
            record.setdefault("type", row["alert_type"])
            record.setdefault("camera_id", row["camera_id"])
            record.setdefault("camera_name", row["camera_name"])
            record.setdefault("timestamp", row["timestamp"])
            record["suppressed"] = bool(row["suppressed"])
            records.append(record)
        return records

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _bootstrap(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_migrations (
                name        TEXT PRIMARY KEY,
                applied_at  INTEGER NOT NULL
            )
            """
        )

    @staticmethod
    def _run_migration(
        conn: sqlite3.Connection,
        version: int,
        statements: list[str],
    ) -> None:
        if not statements:
            raise ValueError(f"Migration {version:03d} contains no SQL statements")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, _now_ms()),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _migration_files(self) -> list[tuple[int, Path]]:
        if not self.migrations_dir.is_dir():
            raise FileNotFoundError(f"Migrations directory not found: {self.migrations_dir}")
        migrations: list[tuple[int, Path]] = []
        versions: set[int] = set()
        for path in sorted(self.migrations_dir.glob("*.sql")):
            match = MIGRATION_NAME.match(path.name)
            if not match:
                continue
            version = int(match.group(1))
            if version in versions:
                raise ValueError(f"Duplicate migration version {version}: {path}")
            versions.add(version)
            migrations.append((version, path))
        return sorted(migrations)


def _parse_sql(sql: str) -> list[str]:
    """Split simple migration SQL into statements without executescript()."""
    uncommented = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return [statement.strip() for statement in uncommented.split(";") if statement.strip()]
