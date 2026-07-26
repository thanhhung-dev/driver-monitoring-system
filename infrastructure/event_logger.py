"""Event Logger — persists risk events to database and structured log files.

Subscribes to RiskEvent via EventBus and writes to both SQLite (via DBManager)
and a structured JSON-lines log file.
"""

import json
import time
from pathlib import Path
from typing import Optional

from features.risk.events import RiskEvent
from infrastructure.database import DBManager


class EventLogger:
    """Logs risk events to database and file."""

    def __init__(
        self,
        db_manager: Optional[DBManager] = None,
        log_dir: str = "logs",
        log_file: str = "events.jsonl",
        min_severity_to_log: str = "medium",
    ) -> None:
        self._db = db_manager
        self._log_dir = Path(log_dir)
        self._log_file = self._log_dir / log_file
        self._min_severity = min_severity_to_log
        self._file = None

        # Severity ordering for filtering
        self._severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def start(self) -> None:
        """Initialize logging resources."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_file, "a", encoding="utf-8")

        if self._db:
            self._db.connect()
            self._db.start_trip()

    def stop(self) -> None:
        """Close logging resources."""
        if self._db:
            self._db.end_trip()
            self._db.close()

        if self._file:
            self._file.close()
            self._file = None

    def on_risk_event(self, event: RiskEvent) -> None:
        """Log a risk event.

        Args:
            event: RiskEvent from RiskEngine.
        """
        # Filter by minimum severity
        event_sev = self._severity_order.get(event.severity.value, 0)
        min_sev = self._severity_order.get(self._min_severity, 0)

        if event_sev < min_sev:
            return

        # Write to JSONL file
        self._write_to_file(event)

        # Write to database
        if self._db:
            self._db.log_event(event)

    def _write_to_file(self, event: RiskEvent) -> None:
        """Append event as JSON line to log file."""
        if not self._file:
            return

        record = {
            "timestamp": event.timestamp,
            "type": event.event_type.value,
            "severity": event.severity.value,
            "score": round(event.score, 4),
            "drowsiness": round(event.drowsiness_score, 4),
            "distraction": round(event.distraction_score, 4),
            "activity": round(event.activity_score, 4),
            "alert": event.requires_alert,
        }

        try:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
        except Exception:
            pass  # Don't crash on logging failure
