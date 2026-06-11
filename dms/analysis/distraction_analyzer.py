"""Distraction analyzer — detects driver inattention from gaze and head pose.

Computes a distraction score (0.0 = focused, 1.0 = fully distracted) based on:
- Gaze direction deviation from forward
- Head yaw / pitch deviation
- Duration of sustained look-away
"""

from collections import deque
from typing import Optional

from analysis.drowsiness_analyzer import EMAFilter


class DistractionAnalyzer:
    """Analyzes driver distraction from gaze and head pose data."""

    def __init__(
        self,
        fps: int = 15,
        gaze_threshold: float = 0.4,
        yaw_threshold: float = 25.0,
        pitch_threshold: float = 20.0,
        lookaway_duration_sec: float = 2.0,
        ema_alpha: float = 0.3,
        window_sec: float = 10.0,
    ) -> None:
        self.gaze_threshold = gaze_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        # Duration tracking
        self._lookaway_frame_thresh = int(fps * lookaway_duration_sec)
        self._lookaway_counter = 0

        # Sliding window for distraction ratio
        window_size = int(fps * window_sec)
        self._distracted_buf: deque = deque(maxlen=max(window_size, 1))

        # EMA smoothing
        self._ema_gaze_x = EMAFilter(ema_alpha)
        self._ema_gaze_y = EMAFilter(ema_alpha)
        self._ema_yaw = EMAFilter(ema_alpha)
        self._ema_pitch = EMAFilter(ema_alpha)

        # Public state
        self.distraction_score: float = 0.0
        self.is_distracted: bool = False
        self.lookaway_duration: float = 0.0
        self.gaze_x: float = 0.0
        self.gaze_y: float = 0.0

    def update(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        head_yaw: Optional[float],
        head_pitch: Optional[float],
        fps: int = 15,
    ) -> float:
        """Process one frame. Returns distraction score [0.0, 1.0].

        Args:
            gaze_x: Horizontal gaze [-1 left, 0 center, +1 right].
            gaze_y: Vertical gaze [-1 up, 0 center, +1 down].
            head_yaw: Head yaw angle in degrees.
            head_pitch: Head pitch angle in degrees.
            fps: Current FPS for duration calculation.

        Returns:
            Distraction score in [0.0, 1.0].
        """
        # Smooth inputs
        if gaze_x is not None:
            self.gaze_x = self._ema_gaze_x.update(gaze_x)
        if gaze_y is not None:
            self.gaze_y = self._ema_gaze_y.update(gaze_y)

        smooth_yaw = self._ema_yaw.update(abs(head_yaw)) if head_yaw is not None else 0.0
        smooth_pitch = self._ema_pitch.update(abs(head_pitch)) if head_pitch is not None else 0.0

        # Check if gaze is away
        gaze_away = (
            abs(self.gaze_x) > self.gaze_threshold
            or abs(self.gaze_y) > self.gaze_threshold
        )

        # Check if head is turned
        head_away = (
            smooth_yaw > self.yaw_threshold
            or smooth_pitch > self.pitch_threshold
        )

        is_looking_away = gaze_away or head_away

        # Track look-away duration
        if is_looking_away:
            self._lookaway_counter += 1
        else:
            self._lookaway_counter = 0

        self.lookaway_duration = self._lookaway_counter / max(fps, 1)

        # Update sliding window
        self._distracted_buf.append(1 if is_looking_away else 0)

        # Compute distraction score
        # Factor 1: Current gaze deviation (normalized)
        gaze_dev = min(
            (abs(self.gaze_x) + abs(self.gaze_y)) / (2 * self.gaze_threshold),
            1.0,
        ) if self.gaze_threshold > 0 else 0.0

        # Factor 2: Head deviation
        yaw_dev = min(smooth_yaw / self.yaw_threshold, 1.0) if self.yaw_threshold > 0 else 0.0
        pitch_dev = min(smooth_pitch / self.pitch_threshold, 1.0) if self.pitch_threshold > 0 else 0.0
        head_dev = max(yaw_dev, pitch_dev)

        # Factor 3: Duration penalty (longer look-away = higher score)
        duration_factor = min(
            self._lookaway_counter / max(self._lookaway_frame_thresh * 3, 1),
            1.0,
        )

        # Factor 4: Distraction ratio in window
        window_ratio = (
            sum(self._distracted_buf) / len(self._distracted_buf)
            if len(self._distracted_buf) > 0
            else 0.0
        )

        # Weighted fusion
        self.distraction_score = min(
            0.30 * gaze_dev
            + 0.25 * head_dev
            + 0.25 * duration_factor
            + 0.20 * window_ratio,
            1.0,
        )

        # Binary distracted flag
        self.is_distracted = (
            self._lookaway_counter >= self._lookaway_frame_thresh
        )

        return self.distraction_score
