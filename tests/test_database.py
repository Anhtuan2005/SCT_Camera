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


if __name__ == "__main__":
    unittest.main()
