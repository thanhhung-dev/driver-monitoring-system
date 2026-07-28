import dataclasses

from features.drowsiness.analyzer import DrowsinessAnalyzer
from pipeline.context import FrameContext


class DrowsinessStage:
    """Drowsiness analysis using EAR, PERCLOS, MAR, and head pose."""

    def __init__(self, analyzer: DrowsinessAnalyzer | None) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "drowsiness"

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.is_driver is False:
            if self._analyzer is not None:
                self._analyzer.reset()
            return ctx
        if self._analyzer is None or ctx.landmarks is None:
            return ctx

        yaw_in = ctx.head_pose[0] if ctx.head_pose else 0
        pitch_in = ctx.head_pose[1] if ctx.head_pose else 0
        driver_state = self._analyzer.update(
            ctx.landmarks,
            pitch=pitch_in,
            yaw=yaw_in,
            attribs=ctx.attribs,
        )

        return dataclasses.replace(
            ctx,
            driver_state=driver_state,
            drowsiness_score=self._analyzer.drowsy_score,
        )
