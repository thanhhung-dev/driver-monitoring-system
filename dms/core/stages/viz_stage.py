from core.frame_context import FrameContext
from core.visualizer import Visualizer
from input.video_capture import VideoCapture
import numpy as np
import cv2
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
        frame = ctx.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if ctx.bbox is None:
                self._visualizer.draw_no_face_warning(canvas)
        else:
            if ctx.landmarks is not None and not ctx.extreme_pose_mode:
                self._visualizer.draw_full_mesh(canvas, ctx.landmarks)
            hp = None if ctx.extreme_pose_mode else ctx.head_pose
            self._visualizer.draw_face_info(
                canvas, ctx.bbox, ctx.landmarks,
                ctx.driver_state, head_pose=hp,
            )
            if ctx.face_lost_extreme_pose:
                self._visualizer.draw_extreme_pose_warning(canvas)
            # Vẽ gaze arrows (màu) trên canvas
            if ctx.gaze_render_data is not None:
                d = ctx.gaze_render_data
                kwargs = dict(
                    length=d["length"], focal_length=1000,
                    head_pose=d["head_pose"], opacity_scale=d["opacity_scale"],
                )
                if d["fallback"]:
                    kwargs.update(num_dots=7, max_radius=10,
                                  crosshair_size=0.3, show_crosshair=True)
                for center in (d["center_l"], d["center_r"]):
                    if center is not None:
                        self._visualizer.draw_gaze_3d(canvas, center, d["vec"], **kwargs)
        fps = self._capture.get_fps()
        self._visualizer.draw_fps(canvas, fps)
        self._visualizer.show("Driver Monitoring", canvas)
        return ctx
