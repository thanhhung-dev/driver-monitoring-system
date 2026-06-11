"""SQLite database manager for DMS event persistence.

Handles connection management, schema creation, and basic CRUD operations
for driver monitoring events.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional

from analysis.events import RiskEvent, Severity


# Schema version for future migrations
SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time REAL NOT NULL,
    end_time REAL,
    total_events INTEGER DEFAULT 0,
    max_severity TEXT DEFAULT 'low'
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    score REAL NOT NULL,
    drowsiness_score REAL DEFAULT 0.0,
    distraction_score REAL DEFAULT 0.0,
    activity_score REAL DEFAULT 0.0,
    metadata TEXT,
    FOREIGN KEY (trip_id) REFERENCES trips(id)
);

CREATE TABLE IF NOT EXISTS system_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_trip_id ON events(trip_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
"""


class DBManager:
    """Manages SQLite database for event persistence."""

    def __init__(self, db_path: str = "data/dms.db") -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._current_trip_id: Optional[int] = None

    def connect(self) -> None:
        """Open database connection and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_schema(self) -> None:
        """Create tables if they don't exist."""
        if not self._conn:
            return
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def start_trip(self) -> int:
        """Start a new trip session. Returns trip ID."""
        if not self._conn:
            raise RuntimeError("Database not connected")
        cursor = self._conn.execute(
            "INSERT INTO trips (start_time) VALUES (?)",
            (time.time(),),
        )
        self._conn.commit()
        self._current_trip_id = cursor.lastrowid
        return self._current_trip_id

    def end_trip(self) -> None:
        """End the current trip session."""
        if not self._conn or not self._current_trip_id:
            return
        self._conn.execute(
            "UPDATE trips SET end_time = ? WHERE id = ?",
            (time.time(), self._current_trip_id),
        )
        self._conn.commit()
        self._current_trip_id = None

    def log_event(self, event: RiskEvent) -> Optional[int]:
        """Log a risk event to the database.

        Args:
            event: RiskEvent to persist.

        Returns:
            Event ID or None if not connected.
        """
        if not self._conn or not self._current_trip_id:
            return None

        cursor = self._conn.execute(
            """INSERT INTO events
               (trip_id, timestamp, event_type, severity, score,
                drowsiness_score, distraction_score, activity_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._current_trip_id,
                event.timestamp,
                event.event_type.value,
                event.severity.value,
                event.score,
                event.drowsiness_score,
                event.distraction_score,
                event.activity_score,
            ),
        )
        self._conn.commit()

        # Update trip stats
        self._update_trip_stats()

        return cursor.lastrowid

    def _update_trip_stats(self) -> None:
        """Update trip aggregate statistics."""
        if not self._conn or not self._current_trip_id:
            return
        self._conn.execute(
            """UPDATE trips SET
                total_events = (SELECT COUNT(*) FROM events WHERE trip_id = ?),
                max_severity = (
                    SELECT severity FROM events
                    WHERE trip_id = ?
                    ORDER BY CASE severity
                        WHEN 'critical' THEN 4
                        WHEN 'high' THEN 3
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 1
                    END DESC
                    LIMIT 1
                )
               WHERE id = ?""",
            (self._current_trip_id, self._current_trip_id, self._current_trip_id),
        )
        self._conn.commit()

    def get_trip_events(self, trip_id: Optional[int] = None) -> list[dict]:
        """Get all events for a trip.

        Args:
            trip_id: Trip ID, or current trip if None.

        Returns:
            List of event dicts.
        """
        if not self._conn:
            return []

        tid = trip_id or self._current_trip_id
        if not tid:
            return []

        cursor = self._conn.execute(
            """SELECT id, timestamp, event_type, severity, score,
                      drowsiness_score, distraction_score, activity_score
               FROM events WHERE trip_id = ?
               ORDER BY timestamp""",
            (tid,),
        )
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "severity": row[3],
                "score": row[4],
                "drowsiness_score": row[5],
                "distraction_score": row[6],
                "activity_score": row[7],
            }
            for row in cursor.fetchall()
        ]

    def get_trip_summary(self, trip_id: Optional[int] = None) -> dict:
        """Get summary statistics for a trip.

        Args:
            trip_id: Trip ID, or current trip if None.

        Returns:
            Summary dict with counts and max severity.
        """
        if not self._conn:
            return {}

        tid = trip_id or self._current_trip_id
        if not tid:
            return {}

        cursor = self._conn.execute(
            """SELECT
                COUNT(*) as total_events,
                SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
                SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_count,
                SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_count,
                SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_count,
                MAX(score) as max_score,
                AVG(score) as avg_score
               FROM events WHERE trip_id = ?""",
            (tid,),
        )
        row = cursor.fetchone()
        if not row:
            return {}

        return {
            "trip_id": tid,
            "total_events": row[0],
            "critical_count": row[1],
            "high_count": row[2],
            "medium_count": row[3],
            "low_count": row[4],
            "max_score": row[5],
            "avg_score": row[6],
        }
