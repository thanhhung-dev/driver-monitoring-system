import numpy as np

from pipeline.context import FrameContext
from utils.logger import setup_logger


# Ngưỡng góc (độ) để phân loại hướng
_YAW_THRESHOLD = 15.0    # ngưỡng yaw: trái/phải
_PITCH_THRESHOLD = 15.0  # ngưỡng pitch: trên/dưới


def classify_head_direction(yaw: float, pitch: float) -> str:
    """Phân loại hướng đầu từ góc yaw/pitch thành nhãn tiếng Việt.

    Args:
        yaw:   Góc quay trái(+)/phải(-), đơn vị độ.
        pitch: Góc cúi lên(+)/xuống(-), đơn vị độ.

    Returns:
        Chuỗi mô tả hướng, ví dụ: "Trái", "Phải-Trên", "Thẳng", ...
    """
    # Xác định thành phần ngang
    if yaw > _YAW_THRESHOLD:
        h = "Trái"
    elif yaw < -_YAW_THRESHOLD:
        h = "Phải"
    else:
        h = ""

    # Xác định thành phần dọc
    if pitch > _PITCH_THRESHOLD:
        v = "Trên"
    elif pitch < -_PITCH_THRESHOLD:
        v = "Dưới"
    else:
        v = ""

    if h and v:
        return f"{h}-{v}"
    if h:
        return h
    if v:
        return v
    return "Thẳng"


class DebugStage:
    """Debug logging for gaze and head pose data.

    Controlled by config["debug"]["gaze_logging"] flag.
    Logs HEAD-POSE + direction every frame for real-time terminal sync.
    """

    def __init__(self, logger=None, config: dict | None = None) -> None:
        self._logger = logger or setup_logger("DebugStage")
        self._enabled = (config or {}).get("debug", {}).get("gaze_logging", False)

    @property
    def name(self) -> str:
        return "debug"

    def process(self, ctx: FrameContext) -> FrameContext:
        if not self._enabled:
            return ctx

        if ctx.head_pose is not None:
            yh, ph, rh = ctx.head_pose
            direction = classify_head_direction(yh, ph)
            self._logger.info(
                f"[HEAD-POSE] P={ph:+6.1f}° Y={yh:+6.1f}° R={rh:+6.1f}°  dir={direction}"
            )
        return ctx
