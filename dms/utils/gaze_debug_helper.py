"""
Gaze Debug Helper - Log và classify hướng nhìn real-time.

Sử dụng:
  từ code:  from utils.gaze_debug_helper import GazeDebugLogger
  trong pipeline:  debug_logger.log(pitch, yaw, frame_number)

Output: in ra terminal hướng nhìn + pitch/yaw raw values.
"""

import logging
import numpy as np

logger = logging.getLogger("gaze_debug")


# ── Thresholds (radians) ─────────────────────────────────────────────
YAW_CENTER = 0.15       # |yaw| < → đang nhìn thẳng (horizontal)
YAW_SIDE = 0.35         # |yaw| > → nhìn sang bên
PITCH_CENTER = 0.15     # |pitch| < → đang nhìn thẳng (vertical)
PITCH_UP = -0.25        # pitch < → nhìn lên
PITCH_DOWN = 0.25       # pitch > → nhìn xuống


def classify_gaze(pitch: float, yaw: float) -> str:
    """Classify [pitch, yaw] radians thành direction label.

    Model convention (Qualcomm EyeNet):
      yaw > 0 = subject looks LEFT → arrow LEFT on screen
      yaw < 0 = subject looks RIGHT → arrow RIGHT on screen
      pitch > 0 = subject looks UP → arrow UP on screen
      pitch < 0 = subject looks DOWN → arrow DOWN on screen
    """
    # Vertical
    if pitch > PITCH_DOWN:
        v = "UP"
    elif pitch < PITCH_UP:
        v = "DOWN"
    else:
        v = ""

    # Horizontal (yaw > 0 = LEFT)
    if yaw > YAW_SIDE:
        h = "LEFT"
    elif yaw < -YAW_SIDE:
        h = "RIGHT"
    elif abs(yaw) < YAW_CENTER:
        h = ""
    else:
        h = "SLIGHT_" + ("LEFT" if yaw > 0 else "RIGHT")

    if v and h:
        return f"{v}-{h}"
    if v:
        return v
    if h:
        return h
    return "STRAIGHT"


def direction_arrow(pitch: float, yaw: float) -> str:
    """ASCII arrow mô tả hướng nhìn TRÊN MÀN HÌNH.

    Model convention (Qualcomm EyeNet):
      yaw > 0 = LEFT → arrow ←
      yaw < 0 = RIGHT → arrow →
      pitch > 0 = UP → arrow ↑
      pitch < 0 = DOWN → arrow ↓
    """
    # Vertical: pitch > 0 → UP, pitch < 0 → DOWN
    if pitch > PITCH_DOWN:
        v = "↑"
    elif pitch < PITCH_UP:
        v = "↓"
    else:
        v = "·"

    # Horizontal: yaw > 0 → LEFT, yaw < 0 → RIGHT
    if yaw > YAW_SIDE:
        h = "←"
    elif yaw < -YAW_SIDE:
        h = "→"
    elif yaw > YAW_CENTER:
        h = "⇠"
    elif yaw < -YAW_CENTER:
        h = "⇢"
    else:
        h = "·"

    return f"{v}{h}"


class GazeDebugLogger:
    """Logger tiện lợi để debug gaze trong pipeline."""

    def __init__(self, log_every_n: int = 10, enabled: bool = True):
        """
        Args:
            log_every_n: Log mỗi N frame (giảm spam).
            enabled: Bật/tắt logging.
        """
        self._log_every_n = max(1, log_every_n)
        self._enabled = enabled
        self._frame_count = 0

    def log(
        self,
        pitch: float | None,
        yaw: float | None,
        frame_number: int | None = None,
        eye: str = "avg",
    ) -> str | None:
        """Log gaze direction. Trả về direction label (để dùng ở nơi khác)."""
        if not self._enabled or pitch is None or yaw is None:
            return None

        self._frame_count += 1
        if self._frame_count % self._log_every_n != 0:
            return None

        direction = classify_gaze(pitch, yaw)
        arrow = direction_arrow(pitch, yaw)
        fn = frame_number if frame_number is not None else self._frame_count

        logger.info(
            "[Frame %d] %s eye | pitch=%.3f (%.1f°) yaw=%.3f (%.1f°) → %s %s",
            fn, eye,
            pitch, np.degrees(pitch),
            yaw, np.degrees(yaw),
            arrow, direction,
        )
        return direction

    def log_avg(
        self,
        gaze_l: np.ndarray | None,
        gaze_r: np.ndarray | None,
        frame_number: int | None = None,
    ) -> str | None:
        """Log gaze trung bình 2 mắt."""
        if gaze_l is not None and gaze_r is not None:
            avg = (gaze_l + gaze_r) / 2.0
        elif gaze_l is not None:
            avg = gaze_l
        elif gaze_r is not None:
            avg = gaze_r
        else:
            return None

        return self.log(float(avg[0]), float(avg[1]), frame_number, eye="avg")

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
