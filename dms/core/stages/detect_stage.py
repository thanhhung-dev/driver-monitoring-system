import dataclasses

import numpy as np

from core.frame_context import FrameContext
from detection.face_detector import FaceDetector


class DetectStage:
    """Face detection with interval-based skipping.

    Runs the detector every N frames; reuses cached results otherwise.
    Forces a detect when previous result was empty (lost tracking).
    """

    def __init__(self, detector: FaceDetector, interval: int = 5) -> None:
        self._detector = detector
        self._interval = interval
        self._counter = 0
        self._last_det = None
        self._last_kpss = None

    @property
    def name(self) -> str:
        return "face_detect"

    def process(self, ctx: FrameContext) -> FrameContext:
        self._counter += 1
        need_detect = (
            self._counter >= self._interval
            or self._last_det is None
            or len(self._last_det) == 0
        )

        if need_detect:
            self._counter = 0
            det, kpss = self._detector.detect(ctx.frame)
            self._last_det, self._last_kpss = det, kpss
        else:
            det, kpss = self._last_det, self._last_kpss

        if det is None or len(det) == 0:
            return ctx

        box = det[0]
        x1, y1, x2, y2 = box[:4].astype(int)
        face_kpss = kpss[0] if kpss is not None else None

        return dataclasses.replace(ctx, bbox=(x1, y1, x2, y2), face_kpss=face_kpss)
