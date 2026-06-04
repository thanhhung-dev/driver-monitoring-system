import dataclasses

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

        return dataclasses.replace(ctx, attribs=attribs)
