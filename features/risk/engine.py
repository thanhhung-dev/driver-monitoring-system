"""Risk Assessment Engine — aggregates analysis results into a unified risk score.

Collects scores from DrowsinessAnalyzer, DistractionAnalyzer (and future
ActivityAnalyzer), computes a weighted risk score, and emits RiskEvent via
the EventBus when risk exceeds thresholds.
"""

import time
from typing import Optional

from features.risk.events import RiskEvent, Severity


class RiskEngine:
    """Aggregates drowsiness, distraction, and activity scores into risk."""

    def __init__(
        self,
        w_drowsiness: float = 0.45,
        w_distraction: float = 0.35,
        w_activity: float = 0.20,
        threshold_warning: float = 0.4,
        threshold_critical: float = 0.7,
    ) -> None:
        self.w_drowsiness = w_drowsiness
        self.w_distraction = w_distraction
        self.w_activity = w_activity
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical

        # Latest scores
        self.drowsiness_score: float = 0.0
        self.distraction_score: float = 0.0
        self.activity_score: float = 0.0
        self.risk_score: float = 0.0
        self.severity: Severity = Severity.LOW

    def update(
        self,
        drowsiness_score: float = 0.0,
        distraction_score: float = 0.0,
        activity_score: float = 0.0,
    ) -> RiskEvent:
        """Compute risk from latest analysis scores.

        Args:
            drowsiness_score: [0.0, 1.0] from DrowsinessAnalyzer.
            distraction_score: [0.0, 1.0] from DistractionAnalyzer.
            activity_score: [0.0, 1.0] from ActivityAnalyzer (future).

        Returns:
            RiskEvent with aggregated scores and severity.
        """
        self.drowsiness_score = drowsiness_score
        self.distraction_score = distraction_score
        self.activity_score = activity_score

        # Weighted sum
        self.risk_score = min(
            self.w_drowsiness * drowsiness_score
            + self.w_distraction * distraction_score
            + self.w_activity * activity_score,
            1.0,
        )

        # Severity classification
        if self.risk_score >= self.threshold_critical:
            self.severity = Severity.CRITICAL
        elif self.risk_score >= self.threshold_warning:
            self.severity = Severity.HIGH
        elif self.risk_score >= 0.2:
            self.severity = Severity.MEDIUM
        else:
            self.severity = Severity.LOW

        requires_alert = self.severity in (Severity.HIGH, Severity.CRITICAL)

        return RiskEvent(
            severity=self.severity,
            score=self.risk_score,
            timestamp=time.time(),
            drowsiness_score=self.drowsiness_score,
            distraction_score=self.distraction_score,
            activity_score=self.activity_score,
            requires_alert=requires_alert,
        )
