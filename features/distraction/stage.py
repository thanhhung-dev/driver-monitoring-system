"""Distraction stage — runs DistractionAnalyzer on each frame.

Reads gaze and head pose from FrameContext, computes distraction score,
and updates the context with results.
"""

import dataclasses

from features.distraction.analyzer import DistractionAnalyzer
from pipeline.context import FrameContext
from pipeline.stage import Stage


class DistractionStage:
    """Pipeline stage for distraction analysis."""

    name = "distraction"

    def __init__(self, analyzer: DistractionAnalyzer) -> None:
        self._analyzer = analyzer

    def process(self, ctx: FrameContext) -> FrameContext:
        """Analyze distraction from gaze and head pose data.

        Args:
            ctx: FrameContext with gaze and head pose data.

        Returns:
            Updated FrameContext with distraction_score.
        """
        if ctx.is_driver is False:
            self._analyzer.reset()
            return ctx

        # Extract gaze data
        gaze_x = None
        gaze_y = None
        if ctx.gaze_l is not None and ctx.gaze_r is not None:
            # Average left and right gaze
            gaze_x = float((ctx.gaze_l[0] + ctx.gaze_r[0]) / 2.0)
            gaze_y = float((ctx.gaze_l[1] + ctx.gaze_r[1]) / 2.0)

        # Extract head pose
        head_yaw = None
        head_pitch = None
        if ctx.head_pose is not None:
            head_yaw = ctx.head_pose[0]  # yaw
            head_pitch = ctx.head_pose[1]  # pitch

        # Run analyzer
        score = self._analyzer.update(
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
        )

        # Build distraction info for FrameContext
        distraction_info = {
            "score": score,
            "is_distracted": self._analyzer.is_distracted,
            "lookaway_duration": self._analyzer.lookaway_duration,
            "gaze_x": self._analyzer.gaze_x,
            "gaze_y": self._analyzer.gaze_y,
        }

        # Return new context with distraction data
        return dataclasses.replace(ctx, distraction_score=score)
