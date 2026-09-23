import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from core.database import DatabaseManager


class DatabaseMigrationTests(unittest.TestCase):
    def test_bootstrap_and_migrations_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sct_camera.db"
            manager = DatabaseManager(db_path)

            manager.bootstrap()
            self.assertEqual([1, 2, 3], manager.run_migrations())
            self.assertEqual([], manager.run_migrations())

            with closing(sqlite3.connect(db_path)) as conn:
                versions = [
                    row[0]
                    for row in conn.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ]
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }

            self.assertEqual([1, 2, 3], versions)
            self.assertTrue(
                {
                    "schema_migrations",
                    "data_migrations",
                    "alerts",
                    "notification_deliveries",
                    "behavior_events",
                    "behavior_labels",
                    "video_clips",
                    "alert_clips",
                }.issubset(tables)
            )

    def test_failed_migration_rolls_back_schema_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            migrations_dir = root / "migrations"
            migrations_dir.mkdir()
            (migrations_dir / "001_broken.sql").write_text(
                """
                CREATE TABLE should_rollback (id INTEGER PRIMARY KEY);
                INSERT INTO missing_table(id) VALUES(1);
                """,
                encoding="utf-8",
            )
            db_path = root / "sct_camera.db"
            manager = DatabaseManager(db_path, migrations_dir=migrations_dir)

            with self.assertRaises(sqlite3.OperationalError):
                manager.run_migrations()

            with closing(sqlite3.connect(db_path)) as conn:
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='should_rollback'"
                ).fetchone()
                versions = list(conn.execute("SELECT version FROM schema_migrations"))

            self.assertIsNone(table)
            self.assertEqual([], versions)

    def test_persists_alert_with_notification_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sct_camera.db"
            manager = DatabaseManager(db_path)
            manager.run_migrations()

            manager.persist_alert(
                {
                    "event_id": "alert-1",
                    "type": "intrusion",
                    "camera_id": "cam",
                    "camera_name": "Front Door",
                    "track_id": 7,
                    "zone_id": "porch",
                    "zone_name": "Porch",
                    "timestamp": "2026-07-15T12:30:00+07:00",
                    "suppressed": False,
                    "message": "Sent to Telegram",
                },
                [{"channel": "telegram", "status": "sent"}],
            )

            recent = manager.get_recent_alerts("cam")
            with closing(sqlite3.connect(db_path)) as conn:
                alert_row = conn.execute(
                    "SELECT event_id, timestamp FROM alerts"
                ).fetchone()
                delivery_row = conn.execute(
                    "SELECT channel, status, sent_at FROM notification_deliveries"
                ).fetchone()

            self.assertEqual("alert-1", recent[0]["event_id"])
            self.assertEqual("Sent to Telegram", recent[0]["message"])
            self.assertEqual("alert-1", alert_row[0])
            self.assertGreater(alert_row[1], 0)
            self.assertEqual(("telegram", "sent"), delivery_row[:2])
            self.assertGreater(delivery_row[2], 0)


if __name__ == "__main__":
    unittest.main()
