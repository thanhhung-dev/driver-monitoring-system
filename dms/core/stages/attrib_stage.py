import dataclasses
import logging

from core.frame_context import FrameContext
from detection.face_attrib_detector import FaceAttribDetector

logger = logging.getLogger(__name__)


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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[attrib] frame=%d | left_eye=%.2f right_eye=%.2f "
                "glasses=%.2f mask=%.2f sunglasses=%.2f",
                ctx.frame_number,
                attribs.get("left_eye_open", 0),
                attribs.get("right_eye_open", 0),
                attribs.get("glasses", 0),
                attribs.get("mask", 0),
                attribs.get("sunglasses", 0),
            )

        return dataclasses.replace(ctx, attribs=attribs)
