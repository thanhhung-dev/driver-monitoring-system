from collections import deque
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


class DriverState(Enum):
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"
    DISTRACTED = "DISTRACTED"


# Colors (BGR) for each state
STATE_COLORS = {
    DriverState.AWAKE: (0, 255, 0),       # green
    DriverState.DROWSY: (0, 165, 255),     # orange
    DriverState.SLEEPING: (0, 0, 255),     # red
    DriverState.DISTRACTED: (0, 255, 255), # yellow
}


def compute_ear(eye_pts: np.ndarray) -> float:
    """Eye Aspect Ratio from 6 landmark points (p1..p6).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks: List[Tuple[int, int]]) -> float:
    """Mouth Aspect Ratio from outer lip landmarks.

    MAR = (||p52-p58|| + ||p51-p57|| + ||p50-p56||) / (2 * ||p49-p55||)
    Using 68-point indices: outer lips 48-59.
    """
    p48 = np.array(landmarks[48])
    p49 = np.array(landmarks[49])
    p50 = np.array(landmarks[50])
    p51 = np.array(landmarks[51])
    p54 = np.array(landmarks[54])
    p55 = np.array(landmarks[55])
    p56 = np.array(landmarks[56])
    p57 = np.array(landmarks[57])
    p58 = np.array(landmarks[58])
    p59 = np.array(landmarks[59])

    vertical_1 = np.linalg.norm(p50 - p58)
    vertical_2 = np.linalg.norm(p51 - p57)
    vertical_3 = np.linalg.norm(p49 - p59)
    horizontal = np.linalg.norm(p48 - p54)
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)


class EMAFilter:
    """Exponential Moving Average filter."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0


class DrowsinessAnalyzer:
    """Full analysis pipeline: EAR → PERCLOS → MAR → Fusion → State."""

    # 68-point landmark indices
    LEFT_EYE = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE = [42, 43, 44, 45, 46, 47]

    def __init__(
        self,
        fps: int = 15,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.65,
        yaw_threshold: float = 30.0,
        perclos_window_sec: float = 60.0,
        distracted_dur_sec: float = 2.0,
        drowsy_score_low: float = 0.3,
        drowsy_score_high: float = 0.7,
        perclos_sleeping: float = 80.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yaw_threshold = yaw_threshold
        self.drowsy_score_low = drowsy_score_low
        self.drowsy_score_high = drowsy_score_high
        self.perclos_sleeping = perclos_sleeping

        # PERCLOS sliding window buffer
        buffer_size = int(fps * perclos_window_sec)
        self._eye_closed_buf: deque = deque(maxlen=max(buffer_size, 1))

        # Distracted duration counter
        self._distracted_frames_thresh = int(fps * distracted_dur_sec)
        self._distracted_counter = 0

        # Yawn frequency: count yawns in last 60s
        self._yawn_buf: deque = deque(maxlen=max(buffer_size, 1))
        self._yawning_prev = False

        # EMA filters
        self._ema_ear = EMAFilter(ema_alpha)
        self._ema_mar = EMAFilter(ema_alpha)
        self._ema_yaw = EMAFilter(ema_alpha)
        self._ema_pitch = EMAFilter(ema_alpha)

        # Latest values (for HUD display)
        self.ear: float = 0.0
        self.mar: float = 0.0
        self.perclos: float = 0.0
        self.drowsy_score: float = 0.0
        self.state: DriverState = DriverState.AWAKE

    # ── Feature extraction ──────────────────────────────────────────────

    def _extract_ear(self, landmarks: List[Tuple[int, int]]) -> float:
        left = np.array([landmarks[i] for i in self.LEFT_EYE], dtype=np.float64)
        right = np.array([landmarks[i] for i in self.RIGHT_EYE], dtype=np.float64)
        return (compute_ear(left) + compute_ear(right)) / 2.0

    def _extract_mar(self, landmarks: List[Tuple[int, int]]) -> float:
        return compute_mar(landmarks)

    # ── PERCLOS ─────────────────────────────────────────────────────────

    def _update_perclos(self, ear: float) -> float:
        closed = 1 if ear < self.ear_threshold else 0
        self._eye_closed_buf.append(closed)
        if len(self._eye_closed_buf) == 0:
            return 0.0
        return (sum(self._eye_closed_buf) / len(self._eye_closed_buf)) * 100.0

    # ── Yawn frequency ──────────────────────────────────────────────────

    def _update_yawn(self, mar: float) -> float:
        yawning = mar > self.mar_threshold
        # Count rising edge (start of a yawn)
        new_yawn = 1 if (yawning and not self._yawning_prev) else 0
        self._yawning_prev = yawning
        self._yawn_buf.append(new_yawn)
        return sum(self._yawn_buf)

    # ── Fusion ──────────────────────────────────────────────────────────

    def update(
        self,
        landmarks: List[Tuple[int, int]],
        pitch: float,
        yaw: float,
    ) -> DriverState:
        """Run full pipeline for one frame. Returns the driver state."""

        # 1. Feature extraction
        raw_ear = self._extract_ear(landmarks)
        raw_mar = self._extract_mar(landmarks)

        # 2. Temporal smoothing
        self.ear = self._ema_ear.update(raw_ear)
        self.mar = self._ema_mar.update(raw_mar)
        smooth_yaw = self._ema_yaw.update(abs(yaw))
        smooth_pitch = self._ema_pitch.update(pitch)

        # 3. PERCLOS & yawn count
        self.perclos = self._update_perclos(self.ear)
        yawn_count = self._update_yawn(self.mar)

        # 4. Distracted check (yaw held beyond threshold)
        if smooth_yaw > self.yaw_threshold:
            self._distracted_counter += 1
        else:
            self._distracted_counter = 0

        if self._distracted_counter >= self._distracted_frames_thresh:
            self.state = DriverState.DISTRACTED
            self.drowsy_score = 0.0
            return self.state

        # 5. Drowsy score fusion
        perclos_norm = min(self.perclos / 100.0, 1.0)
        ear_inv = max(1.0 - (self.ear / 0.35), 0.0)  # 0.35 ≈ wide open
        # Normalize yawn: cap at 5 yawns/minute
        yawn_norm = min(yawn_count / 5.0, 1.0)
        # Head nod: pitch < -15° => nodding
        head_nod = max(min((-smooth_pitch - 15.0) / 15.0, 1.0), 0.0)

        self.drowsy_score = (
            0.45 * perclos_norm
            + 0.25 * ear_inv
            + 0.20 * yawn_norm
            + 0.10 * head_nod
        )

        # 6. State machine
        if self.perclos > self.perclos_sleeping or self.drowsy_score >= self.drowsy_score_high:
            self.state = DriverState.SLEEPING
        elif self.drowsy_score >= self.drowsy_score_low:
            self.state = DriverState.DROWSY
        else:
            self.state = DriverState.AWAKE

        return self.state
