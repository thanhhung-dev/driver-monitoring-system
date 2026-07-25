"""Risk assessment stage — aggregates analysis scores and emits RiskEvent.

Runs after DrowsinessStage and DistractionStage, combines their scores
via RiskEngine, and publishes RiskEvent to EventBus.
"""

from features.risk.engine import RiskEngine
from features.risk.events import RiskEvent
from pipeline.context import FrameContext
from pipeline.event_bus import EventBus
from pipeline.stage import Stage


class RiskStage:
    """Pipeline stage for risk assessment and event emission."""

    name = "risk"

    def __init__(
        self,
        risk_engine: RiskEngine,
        event_bus: EventBus | None = None,
    ) -> None:
        self._engine = risk_engine
        self._bus = event_bus

    def process(self, ctx: FrameContext) -> FrameContext:
        """Compute risk score from analysis results and publish event.

        Args:
            ctx: FrameContext with drowsiness_score and distraction_score.

        Returns:
            Updated FrameContext with risk_score and risk_severity.
        """
        # Get scores from context (set by previous stages)
        drowsiness = ctx.drowsiness_score if ctx.drowsiness_score is not None else 0.0
        distraction = ctx.distraction_score if ctx.distraction_score is not None else 0.0

        # Compute risk
        risk_event = self._engine.update(
            drowsiness_score=drowsiness,
            distraction_score=distraction,
        )

        # Publish event to bus
        if self._bus and risk_event.requires_alert:
            self._bus.publish("risk", risk_event)

        # Return updated context
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
            drowsiness_score=drowsiness,
            distraction_score=distraction,
            risk_score=risk_event.score,
            risk_severity=risk_event.severity.value,
        )
