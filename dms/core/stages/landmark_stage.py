from core.frame_context import FrameContext
from detection.facemap_3dmm import FaceMap3DMMDetector


class LandmarkStage:
    """Runs FaceMap3DMM landmark detection on the detected face."""

    def __init__(self, facemap: FaceMap3DMMDetector | None) -> None:
        self._facemap = facemap

    @property
    def name(self) -> str:
        return "landmark"

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._facemap is None or ctx.bbox is None:
            return ctx

        facemap_out = self._facemap.detect(ctx.frame, ctx.bbox, ctx.face_kpss)
        if facemap_out is None:
            return ctx

        landmarks, facemap_pose = facemap_out
        return FrameContext(
            frame=ctx.frame,
            frame_number=ctx.frame_number,
            bbox=ctx.bbox,
            face_kpss=ctx.face_kpss,
            landmarks=landmarks,
            facemap_pose=facemap_pose,
        )
