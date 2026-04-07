"""
env/database.py — SQLite query runner (Person A)

Provides a thin, safe wrapper around the sqlite3 standard library module.
All destructive SQL is blocked. All exceptions are caught and returned as
QueryResult objects with success=False.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from env.models import QueryResult


# Keywords that indicate a potentially destructive operation.
_BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE)\b",
    re.IGNORECASE,
)


class DatabaseRunner:
    """Execute read-only SQL against a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g. ``data/analyst.db``).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database file not found: {self._db_path}. "
                "Run `python data/seed.py` first."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schema_description(self) -> str:
        """Return a human-readable description of every table and column.

        The output is intended to be included verbatim in the agent's
        Observation so it knows what data is available.
        """
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Fetch all user-created tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        lines: list[str] = ["=== Database Schema ===", ""]
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            # Each column row: (cid, name, type, notnull, dflt_value, pk)
            lines.append(f"Table: {table}")
            lines.append("-" * (len(table) + 7))
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                pk = " [PK]" if col[5] else ""
                nn = " NOT NULL" if col[3] else ""
                lines.append(f"  {col_name} {col_type}{pk}{nn}")
            lines.append("")

        conn.close()
        return "\n".join(lines)

    def is_safe_query(self, sql: str) -> bool:
        """Return True if *sql* does not contain destructive keywords.

        Blocks: DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE.
        The check is case-insensitive and matches whole words.
        """
        return _BLOCKED_KEYWORDS.search(sql) is None

    def run_query(self, sql: str) -> QueryResult:
        """Execute *sql* and return the result as a :class:`QueryResult`.

        If the query fails for any reason (syntax error, runtime error,
        blocked keyword), the result will have ``success=False`` and the
        error message populated.  This method **never** raises.
        """
        # --- Safety check ---
        if not self.is_safe_query(sql):
            return QueryResult(
                sql=sql,
                success=False,
                rows=[],
                error="Query blocked: contains a destructive keyword "
                      "(DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE).",
                row_count=0,
            )

        # --- Execute ---
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row  # enables dict-like access
            cursor = conn.cursor()
            cursor.execute(sql)
            raw_rows = cursor.fetchall()
            # Convert sqlite3.Row objects to plain dicts
            rows = [dict(row) for row in raw_rows]
            conn.close()
            return QueryResult(
                sql=sql,
                success=True,
                rows=rows,
                error=None,
                row_count=len(rows),
            )
        except sqlite3.Error as exc:
            return QueryResult(
                sql=sql,
                success=False,
                rows=[],
                error=str(exc),
                row_count=0,
            )
        except Exception as exc:  # noqa: BLE001 — intentional catch-all
            return QueryResult(
                sql=sql,
                success=False,
                rows=[],
                error=f"Unexpected error: {exc}",
                row_count=0,
            )
