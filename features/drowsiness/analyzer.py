from collections import deque
from enum import Enum
from typing import Dict, List, Tuple, Optional

import logging
import numpy as np

logger = logging.getLogger(__name__)


# Default fusion weights
_DEFAULT_WEIGHTS = {
    "perclos": 0.45,
    "ear_inv": 0.25,
    "yawn": 0.20,
    "head_nod": 0.10,
}

# Track previous attrib mode to log only on transition
_prev_attrib_mode: str = "none"


def _classify_attrib_mode(attribs: Optional[Dict[str, float]]) -> str:
    """Return a human-readable label for the current attribute combination."""
    if attribs is None:
        return "none"
    sg = attribs.get("sunglasses", 0.0) > 0.5
    gl = attribs.get("glasses", 0.0) > 0.5
    mk = attribs.get("mask", 0.0) > 0.5
    if sg and mk:
        return "sunglasses+mask"
    if sg:
        return "sunglasses"
    if gl and mk:
        return "glasses+mask"
    if gl:
        return "glasses"
    if mk:
        return "mask"
    return "normal"


def _adjust_weights_for_attribs(
    attribs: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Adjust drowsiness fusion weights based on detected facial attributes.

    EAR is now driven by the Qualcomm FaceAttribNet eye-openness output, which
    is trained to work through eyeglasses/sunglasses, so eye-based features
    (PERCLOS, ear_inv) are kept reliable regardless of eyewear.

    - mask → disable yawn (mouth occluded), redistribute its weight
    """
    global _prev_attrib_mode
    w = dict(_DEFAULT_WEIGHTS)

    if attribs is None:
        mode = "none"
        if mode != _prev_attrib_mode:
            logger.info("[attrib] No attributes → using default weights")
            _prev_attrib_mode = mode
        return w

    sunglasses = attribs.get("sunglasses", 0.0)
    glasses = attribs.get("glasses", 0.0)
    mask = attribs.get("mask", 0.0)

    # Mask: mouth invisible → skip yawn entirely, redistribute its weight
    # across the remaining eye- and head-based features.
    if mask > 0.5:
        yawn_redist = w["yawn"]
        w["yawn"] = 0.0
        w["perclos"] += yawn_redist * 0.6
        w["ear_inv"] += yawn_redist * 0.2
        w["head_nod"] += yawn_redist * 0.2

    # Log on mode transition
    mode = _classify_attrib_mode(attribs)
    if mode != _prev_attrib_mode:
        logger.info(
            "[attrib] Mode changed: %s → %s | "
            "sunglasses=%.2f glasses=%.2f mask=%.2f | "
            "weights: perclos=%.2f ear=%.2f yawn=%.2f head=%.2f",
            _prev_attrib_mode, mode,
            sunglasses, glasses, mask,
            w["perclos"], w["ear_inv"], w["yawn"], w["head_nod"],
        )
        _prev_attrib_mode = mode

    return w


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
        # EAR threshold is now probability-based [0,1] since we use the
        # Qualcomm FaceAttribNet eye-openness output by default.
        # Landmark EAR (geometric, ~0.15–0.35) is only a fallback.
        ear_threshold: float = 0.5,
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
        self.ear_source: str = "model"  # "model" (FaceAttribNet) or "landmark" (fallback)
        self._prev_ear_source: Optional[str] = None  # log only on source change
        self.mar: float = 0.0
        self.perclos: float = 0.0
        self.drowsy_score: float = 0.0
        self.state: DriverState = DriverState.AWAKE
        self._prev_state: DriverState = DriverState.AWAKE

        logger.info(
            "[drowsiness] DrowsinessAnalyzer initialized | "
            "ear_threshold=%.2f (prob-based) mar_threshold=%.2f "
            "yaw_threshold=%.1f drowsy=[%.2f,%.2f] perclos_sleep=%.1f",
            self.ear_threshold, self.mar_threshold, self.yaw_threshold,
            self.drowsy_score_low, self.drowsy_score_high,
            self.perclos_sleep,
        )

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
        attribs: Optional[Dict[str, float]] = None,
    ) -> DriverState:
        """Run full pipeline for one frame. Returns the driver state.

        Args:
            landmarks: 68-point facial landmarks.
            pitch: Head pitch in degrees.
            yaw: Head yaw in degrees.
            attribs: Optional dict from FaceAttribDetector with keys:
                left_eye_open, right_eye_open, glasses, mask, sunglasses.
                Used to adapt fusion weights (e.g., disable EAR under sunglasses).
        """

        # 1. Feature extraction
        # Eye openness: prefer the Qualcomm FaceAttribNet model output
        # (left_eye_open / right_eye_open probabilities in [0,1]); fall back
        # to the geometric landmark EAR only when the model did not run.
        raw_ear_landmark = self._extract_ear(landmarks)
        raw_mar = self._extract_mar(landmarks)

        model_ear = None
        if attribs is not None:
            left_open = attribs.get("left_eye_open", None)
            right_open = attribs.get("right_eye_open", None)
            if left_open is not None and right_open is not None:
                model_ear = (left_open + right_open) / 2.0

        if model_ear is not None:
            raw_ear = model_ear
            self.ear_source = "model"
        else:
            # Fallback: geometric EAR from 68-point landmarks.
            # NOTE: range differs (~0.15–0.35), so ear_threshold (0.5) is
            # biased toward model usage. Re-enable model ASAP.
            raw_ear = raw_ear_landmark
            self.ear_source = "landmark"

        # Log on EAR source transition (model ↔ landmark)
        if self.ear_source != self._prev_ear_source:
            if self.ear_source == "model":
                logger.info(
                    "[drowsiness] EAR source → model (FaceAttribNet) | "
                    "left_eye_open=%.3f right_eye_open=%.3f raw_ear=%.3f",
                    attribs.get("left_eye_open", 0.0) if attribs else 0.0,
                    attribs.get("right_eye_open", 0.0) if attribs else 0.0,
                    raw_ear,
                )
            else:
                logger.info(
                    "[drowsiness] EAR source → landmark (fallback) | "
                    "attribs=%s raw_ear=%.3f",
                    "missing" if attribs is None else "no eye_open",
                    raw_ear,
                )
            self._prev_ear_source = self.ear_source

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

        # 5. Drowsy score fusion — weights adapt to facial attributes
        w = _adjust_weights_for_attribs(attribs)

        perclos_norm = min(self.perclos / 100.0, 1.0)
        # ear_inv: probability that the eyes are closed.
        # self.ear is now a model probability [0,1] (1 = fully open),
        # so ear closed-ness is simply 1 - ear.
        ear_inv = max(1.0 - self.ear, 0.0)
        # Normalize yawn: cap at 5 yawns/minute
        yawn_norm = min(yawn_count / 5.0, 1.0)
        # Head nod: pitch < -15° => nodding
        head_nod = max(min((-smooth_pitch - 15.0) / 15.0, 1.0), 0.0)

        self.drowsy_score = (
            w["perclos"] * perclos_norm
            + w["ear_inv"] * ear_inv
            + w["yawn"] * yawn_norm
            + w["head_nod"] * head_nod
        )

        # 6. State machine
        if self.perclos > self.perclos_sleeping or self.drowsy_score >= self.drowsy_score_high:
            self.state = DriverState.SLEEPING
        elif self.drowsy_score >= self.drowsy_score_low:
            self.state = DriverState.DROWSY
        else:
            self.state = DriverState.AWAKE

        # Log state transitions
        if self.state != self._prev_state:
            logger.info(
                "[drowsiness] State: %s → %s | "
                "score=%.3f perclos=%.1f%% ear=%.3f mar=%.3f",
                self._prev_state.value, self.state.value,
                self.drowsy_score, self.perclos, self.ear, self.mar,
            )
            self._prev_state = self.state

        # Debug: per-frame values (only when debug logging enabled)
        if logger.isEnabledFor(logging.DEBUG):
            attrib_mode = _classify_attrib_mode(attribs)
            logger.debug(
                "[drowsiness] ear=%.3f(%s) mar=%.3f perclos=%.1f%% "
                "score=%.3f state=%s attrib_mode=%s",
                self.ear, self.ear_source, self.mar, self.perclos,
                self.drowsy_score, self.state.value, attrib_mode,
            )

        return self.state
