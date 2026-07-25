"""Distraction stage — runs DistractionAnalyzer on each frame.

Reads gaze and head pose from FrameContext, computes distraction score,
and updates the context with results.
"""

import time

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
            gaze_vec_world=ctx.gaze_vec_world,
            eye_center_l=ctx.eye_center_l,
            eye_center_r=ctx.eye_center_r,
            attribs=ctx.attribs,
            driver_state=ctx.driver_state,
            frame_flipped=ctx.frame_flipped,
            face_lost_extreme_pose=ctx.face_lost_extreme_pose,
            extreme_pose_mode=ctx.extreme_pose_mode,
            gaze_render_data=ctx.gaze_render_data,
            distraction_score=score,
        )
