from core.frame_context import FrameContext
from analysis.drowsiness_analyzer import DrowsinessAnalyzer


class DrowsinessStage:
    """Drowsiness analysis using EAR, PERCLOS, MAR, and head pose."""

    def __init__(self, analyzer: DrowsinessAnalyzer | None) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "drowsiness"

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._analyzer is None or ctx.landmarks is None:
            return ctx

        yaw_in = ctx.head_pose[0] if ctx.head_pose else 0
        pitch_in = ctx.head_pose[1] if ctx.head_pose else 0
        driver_state = self._analyzer.update(ctx.landmarks, pitch=pitch_in, yaw=yaw_in)

        return FrameContext(
            frame=ctx.frame,
            frame_number=ctx.frame_number,
            bbox=ctx.bbox,
            face_kpss=ctx.face_kpss,
            landmarks=ctx.landmarks,
            facemap_pose=ctx.facemap_pose,
            head_pose=ctx.head_pose,
            head_rotation_matrix=ctx.head_rotation_matrix,
            gaze_l=ctx.gaze_l,
            gaze_r=ctx.gaze_r,
            eye_center_l=ctx.eye_center_l,
            eye_center_r=ctx.eye_center_r,
            attribs=ctx.attribs,
            driver_state=driver_state,
        )
