import dataclasses

from features.landmarks.detector import FaceMap3DMMDetector
from pipeline.context import FrameContext


class LandmarkStage:
    """Runs FaceMap3DMM landmark detection on the detected face."""

    def __init__(self, facemap: FaceMap3DMMDetector | None) -> None:
        self._facemap = facemap

    @property
    def name(self) -> str:
        return "landmark"

    def process(self, ctx: FrameContext) -> FrameContext:
        # Extreme pose (|yaw|>80°): bỏ qua landmark → kéo theo gaze/attrib/
        # drowsiness cũng skip (chúng cần landmarks). Chỉ giữ SCRFD + head pose.
        if (
            self._facemap is None
            or ctx.bbox is None
            or ctx.extreme_pose_mode
            or ctx.is_driver is False
        ):
            return ctx

        facemap_out = self._facemap.detect(ctx.frame, ctx.bbox, ctx.face_kpss)
        if facemap_out is None:
            return ctx

        landmarks, facemap_pose = facemap_out
        return dataclasses.replace(ctx, landmarks=landmarks, facemap_pose=facemap_pose)
