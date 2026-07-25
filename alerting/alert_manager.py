"""Alert Manager — orchestrates audio and visual alerts based on risk events.

Subscribes to RiskEvent via EventBus and triggers appropriate alerts with
throttling to avoid spam.
"""

import time
from typing import Optional

from analysis.events import RiskEvent, Severity


class AlertManager:
    """Manages alert delivery with throttling and escalation.

    Alert levels:
    - LOW/MEDIUM: No alert
    - HIGH: Warning beep + LED blink
    - CRITICAL: Continuous alarm + LED fast blink
    """

    def __init__(
        self,
        cooldown_sec: float = 5.0,
        critical_cooldown_sec: float = 2.0,
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.critical_cooldown_sec = critical_cooldown_sec

        self._last_alert_time: float = 0.0
        self._last_severity: Severity = Severity.LOW
        self._alert_count: int = 0

        # Public state for visualization
        self.current_alert_level: str = "none"
        self.is_alerting: bool = False

    def on_risk_event(self, event: RiskEvent) -> Optional[str]:
        """Process a risk event and decide whether to alert.

        Args:
            event: RiskEvent from RiskEngine.

        Returns:
            Alert action string ("warning", "critical", "none") or None if throttled.
        """
        now = time.time()

        if not event.requires_alert:
            self.current_alert_level = "none"
            self.is_alerting = False
            return None

        # Determine cooldown based on severity
        cooldown = (
            self.critical_cooldown_sec
            if event.severity == Severity.CRITICAL
            else self.cooldown_sec
        )

        # Throttle: skip if within cooldown
        if now - self._last_alert_time < cooldown:
            return None

        self._last_alert_time = now
        self._last_severity = event.severity
        self._alert_count += 1

        if event.severity == Severity.CRITICAL:
            self.current_alert_level = "critical"
            self.is_alerting = True
            return "critical"
        elif event.severity == Severity.HIGH:
            self.current_alert_level = "warning"
            self.is_alerting = True
            return "warning"

        return "none"

    def reset(self) -> None:
        """Reset alert state."""
        self._last_alert_time = 0.0
        self._last_severity = Severity.LOW
        self._alert_count = 0
        self.current_alert_level = "none"
        self.is_alerting = False
