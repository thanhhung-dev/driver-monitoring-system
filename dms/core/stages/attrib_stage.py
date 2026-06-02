from core.frame_context import FrameContext
from detection.face_attrib_detector import FaceAttribDetector


class AttribStage:
    """Facial attribute detection (age, gender, etc.)."""

    def __init__(self, attrib_detector: FaceAttribDetector | None) -> None:
        self._detector = attrib_detector

    @property
    def name(self) -> str:
        return "attrib"

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._detector is None or ctx.bbox is None:
            return ctx

        attribs = self._detector.detect(ctx.frame, ctx.bbox)
        if attribs is None:
            return ctx

        return FrameContext(
            frame=ctx.frame,
            frame_number=ctx.frame_number,
            bbox=ctx.bbox,
            face_kpss=ctx.face_kpss,
            landmarks=ctx.landmarks,
            facemap_pose=ctx.facemap_pose,
            head_pose=ctx.head_pose,
            gaze_l=ctx.gaze_l,
            gaze_r=ctx.gaze_r,
            eye_center_l=ctx.eye_center_l,
            eye_center_r=ctx.eye_center_r,
            attribs=attribs,
        )
