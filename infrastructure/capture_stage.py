import cv2
import numpy as np

from infrastructure.camera import VideoCapture
from pipeline.context import FrameContext


class CaptureStage:
    """Reads frames from VideoCapture and applies horizontal flip for webcam."""

    def __init__(self, capture: VideoCapture) -> None:
        self._capture = capture
        self._frame_number = 0

    @property
    def name(self) -> str:
        return "capture"

    def process(self, ctx: FrameContext) -> FrameContext:
        ret, frame = self._capture.read_frame()
        if not ret:
            return ctx

        flipped = False
        if not self._capture._is_video_file:
            frame = cv2.flip(frame, 1)
            flipped = True

        self._frame_number += 1
        return FrameContext(frame=frame, frame_number=self._frame_number, frame_flipped=flipped)
