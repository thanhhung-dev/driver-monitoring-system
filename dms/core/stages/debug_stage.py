import numpy as np

from core.frame_context import FrameContext
from utils.logger import setup_logger


class DebugStage:
    """Debug logging for gaze and head pose data.

    Controlled by config["debug"]["gaze_logging"] flag.
    Logs GAZE-RAW, GAZE-AVG, GAZE-VEC, HEAD-POSE, OFFSET every N frames.
    """

    def __init__(self, logger=None, config: dict | None = None) -> None:
        self._logger = logger or setup_logger("DebugStage")
        self._enabled = (config or {}).get("debug", {}).get("gaze_logging", False)
        self._counter = 0
        self._every = 15

    @property
    def name(self) -> str:
        return "debug"

    def _pitchyaw_to_vec(self, pitchyaw: np.ndarray) -> np.ndarray:
        """Same convention as GazeStage._pitchyaw_to_vec (screen space)."""
        pitch = float(pitchyaw[0])
        yaw = float(pitchyaw[1])
        sin_p = np.sin(pitch)
        cos_p = np.cos(pitch)
        sin_y = np.sin(yaw)
        cos_y = np.cos(yaw)
        x = -cos_p * sin_y     # yaw > 0 (LEFT) → x < 0 → arrow LEFT
        y = -sin_p              # pitch > 0 (UP) → y < 0 → arrow UP
        z = cos_p * cos_y
        return np.array([x, y, z], dtype=np.float32)

    def process(self, ctx: FrameContext) -> FrameContext:
        if not self._enabled:
            return ctx

        self._counter += 1
        if self._counter % self._every != 0:
            return ctx

        def _pdeg(g):
            return None if g is None else (np.degrees(float(g[0])), np.degrees(float(g[1])))

        pl = _pdeg(ctx.gaze_l)
        pr = _pdeg(ctx.gaze_r)
        pl_s = "  None        " if pl is None else f"P={pl[0]:+6.1f}° Y={pl[1]:+6.1f}°"
        pr_s = "  None        " if pr is None else f"P={pr[0]:+6.1f}° Y={pr[1]:+6.1f}°"
        self._logger.info(f"[GAZE-RAW]  L[{pl_s}]  R[{pr_s}]")

        if ctx.gaze_l is not None and ctx.gaze_r is not None:
            avg = (ctx.gaze_l + ctx.gaze_r) / 2.0
            self._logger.info(
                f"[GAZE-AVG]  P={np.degrees(float(avg[0])):+6.1f}° "
                f"Y={np.degrees(float(avg[1])):+6.1f}°"
            )

        # Blended gaze vector (eye + head) — world space, already includes
        # head rotation and confidence-based blending.
        if ctx.gaze_vec_world is not None:
            v = ctx.gaze_vec_world
            sign_y = "RIGHT" if v[0] > 0.05 else ("LEFT " if v[0] < -0.05 else "STR ")
            sign_p = "UP  " if v[1] < -0.05 else ("DOWN" if v[1] > 0.05 else "FLAT")
            self._logger.info(
                f"[GAZE-VEC]  x={v[0]:+.3f}({sign_y})  y={v[1]:+.3f}({sign_p})  "
                f"z={v[2]:+.3f}  ← blended eye+head"
            )

        if ctx.head_pose is not None:
            yh, ph, rh = ctx.head_pose
            self._logger.info(
                f"[HEAD-POSE] P={ph:+6.1f}° Y={yh:+6.1f}° R={rh:+6.1f}°"
            )

        self._logger.info(
            f"[OFFSET]    pitch_offset=+0.0° yaw_offset=+0.0°"
        )
        self._logger.info("-" * 60)

        return ctx
