import numpy as np
import cv2

from infrastructure.camera import VideoCapture
from pipeline.context import FrameContext
from presentation.opencv.visualizer import Visualizer
from utils.helpers import draw_bbox_info, draw_bbox
import logging
logger = logging.getLogger(__name__)


class VizStage:
    """Visualization stage: draws mesh, face info, FPS, and shows the frame.

    Must run on the main thread (OpenCV GUI requirement).
    """

    def __init__(self, visualizer: Visualizer | None, capture: VideoCapture) -> None:
        self._visualizer = visualizer
        self._capture = capture
        self._last_logged_driver = None

    @property
    def name(self) -> str:
        return "viz"

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._visualizer is None:
            return ctx
        frame = ctx.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        detections = ctx.face_detections
        has_detected_faces = detections is not None and len(detections) > 0
        
        # Draw ROI box if configured
        if ctx.driver_roi:
            h, w = canvas.shape[:2]
            x_min, y_min, x_max, y_max = ctx.driver_roi
            roi_box = (int(w * x_min), int(h * y_min), int(w * x_max), int(h * y_max))
            draw_bbox(canvas, roi_box, (117, 255, 117), thickness=2)
            cv2.putText(canvas, "", (roi_box[0] + 5, roi_box[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

        if ctx.is_driver is not None:
            for index, detection in enumerate(detections if has_detected_faces else []):
                is_driver = index == ctx.driver_face_index
                similarity = (
                    float(ctx.face_similarities[index])
                    if ctx.face_similarities is not None
                    else 0.0
                )
                if is_driver:
                    driver_name = ctx.driver_name or "DRIVER"
                    if driver_name != self._last_logged_driver:
                        logger.info(f"Identity: {driver_name} (Similarity: {similarity:.2f})")
                        self._last_logged_driver = driver_name
                else:
                    draw_bbox(
                        canvas,
                        detection[:4].astype(int).tolist(),
                        (255, 128, 255),
                        thickness=2
                    )

        if ctx.bbox is not None:
            if ctx.landmarks is not None and not ctx.extreme_pose_mode:
                self._visualizer.draw_full_mesh(canvas, ctx.landmarks)
            hp = None if ctx.extreme_pose_mode else ctx.head_pose
            # Chỉ vẽ bbox SCRFD khi extreme (|yaw|>85°). Bình thường (0–85°)
            # ẩn bbox, vẫn giữ mesh + trục head-pose.
            draw_box = ctx.extreme_pose_mode
            self._visualizer.draw_face_info(
                canvas, ctx.bbox, ctx.landmarks,
                ctx.driver_state, head_pose=hp, draw_box=draw_box,
            )
            # Vẽ gaze arrows (màu) trên canvas
            if ctx.gaze_render_data is not None:
                d = ctx.gaze_render_data
                kwargs = dict(
                    length=d["length"], focal_length=1000,
                    head_pose=d["head_pose"], opacity_scale=d["opacity_scale"],
                    show_crosshair=d.get("show_crosshair", True)
                )
                if d["fallback"]:
                    kwargs.update(num_dots=7, max_radius=10,
                                  crosshair_size=0.3)
                for center in (d["center_l"], d["center_r"]):
                    if center is not None:
                        self._visualizer.draw_gaze_3d(canvas, center, d["vec"], **kwargs)
        fps = self._capture.get_fps()
        self._visualizer.draw_fps(canvas, fps)
        self._visualizer.show("Driver Monitoring", canvas)
        return ctx
