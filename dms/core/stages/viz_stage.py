from core.frame_context import FrameContext
from core.visualizer import Visualizer
from input.video_capture import VideoCapture


class VizStage:
    """Visualization stage: draws mesh, face info, FPS, and shows the frame.

    Must run on the main thread (OpenCV GUI requirement).
    """

    def __init__(self, visualizer: Visualizer | None, capture: VideoCapture) -> None:
        self._visualizer = visualizer
        self._capture = capture

    @property
    def name(self) -> str:
        return "viz"

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._visualizer is None:
            return ctx

        if ctx.bbox is None:
            self._visualizer.draw_no_face_warning(ctx.frame)
        else:
            if ctx.landmarks is not None:
                self._visualizer.draw_full_mesh(ctx.frame, ctx.landmarks)
            self._visualizer.draw_face_info(
                ctx.frame, ctx.bbox, ctx.landmarks,
                ctx.driver_state, head_pose=ctx.head_pose,
            )

        fps = self._capture.get_fps()
        self._visualizer.draw_fps(ctx.frame, fps)
        self._visualizer.show("Driver Monitoring", ctx.frame)

        return ctx
