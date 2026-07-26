"""Typed events for the DMS event bus.

All events are frozen dataclasses to ensure immutability.
Events are published via EventBus and consumed by Action layer components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    """Event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of events emitted by the analysis layer."""
    DROWSINESS = "drowsiness"
    DISTRACTION = "distraction"
    ACTIVITY = "activity"
    RISK = "risk"
    FACE_LOST = "face_lost"
    SYSTEM = "system"


@dataclass(frozen=True)
class AnalysisEvent:
    """Base event emitted by analyzers."""
    event_type: EventType
    severity: Severity
    score: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DrowsinessEvent(AnalysisEvent):
    """Emitted when drowsiness is detected."""
    event_type: EventType = field(default=EventType.DROWSINESS, init=False)
    perclos: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    yawn_count: int = 0


@dataclass(frozen=True)
class DistractionEvent(AnalysisEvent):
    """Emitted when distraction is detected."""
    event_type: EventType = field(default=EventType.DISTRACTION, init=False)
    gaze_away_duration: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0


@dataclass(frozen=True)
class RiskEvent(AnalysisEvent):
    """Emitted by RiskEngine after aggregating all analysis results."""
    event_type: EventType = field(default=EventType.RISK, init=False)
    drowsiness_score: float = 0.0
    distraction_score: float = 0.0
    activity_score: float = 0.0
    requires_alert: bool = False


@dataclass(frozen=True)
class FaceLostEvent(AnalysisEvent):
    """Emitted when face is lost for extended period."""
    event_type: EventType = field(default=EventType.FACE_LOST, init=False)
    lost_duration_frames: int = 0
