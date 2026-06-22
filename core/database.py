"""SQLite bootstrap and transactional schema migrations."""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


MIGRATION_NAME = re.compile(r"^(\d+)_.*\.sql$")


def _now_ms() -> int:
    return int(time.time() * 1000)


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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
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
