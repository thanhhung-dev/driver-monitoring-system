import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

from input.video_capture import VideoCapture, CameraNotFoundError
from detection.face_detector import FaceDetector
from detection.facemap_3dmm import FaceMap3DMMDetector
from detection.face_attrib_detector import FaceAttribDetector
from analysis.drowsiness_analyzer import DrowsinessAnalyzer
from core.visualizer import Visualizer
from utils.logger import setup_logger
from utils.helpers import expand_bbox
from utils.general import compute_euler_angles_from_rotation_matrices

MIN_FPS = 15


class DMSPipeline:
    """Orchestrates the main Driver Monitoring System loop.

    Connects video capture, face detection, landmark estimation,
    head pose, facial attributes, drowsiness analysis, and visualization.
    """
    def __init__(
        self,
        capture: VideoCapture,
        detector: FaceDetector,
        facemap: FaceMap3DMMDetector,
        attrib_detector: FaceAttribDetector,
        head_pose,
        analyzer: DrowsinessAnalyzer,
        visualizer: Visualizer,
        device: torch.device = None,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.facemap = facemap
        self.attrib_detector = attrib_detector
        self.head_pose = head_pose
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.device = device or torch.device("cpu")
        self.logger = setup_logger("DMSPipeline")

    def start(self) -> None:
        """Open the capture source and run the main processing loop."""
        try:
            self.capture.open()
            self.logger.info("Capture loop started. Press 'q' to quit.")
            self._run_loop()
        except CameraNotFoundError as e:
            self.logger.error(str(e))
            sys.exit(1)
        finally:
            self.stop()

    def stop(self) -> None:
        """Release resources and destroy display windows."""
        self.capture.release()
        cv2.destroyAllWindows()
        self.logger.info("System shutdown")

    def _run_loop(self) -> None:
        """Core frame-processing loop."""
        with torch.no_grad():
            while True:
                ret, frame = self.capture.read_frame()
                if not ret:
                    break

                if not self.capture._is_video_file:
                    frame = cv2.flip(frame, 1)

                det, _ = self.detector.detect(frame)

                if det is None or len(det) == 0:
                    self.visualizer.draw_no_face_warning(frame)
                else:
                    for box in det:
                        x1, y1, x2, y2 = box[:4].astype(int)
                        bbox = (x1, y1, x2, y2)
                        bbox_width = x2 - x1

                        # Pad bbox slightly for better landmark fit
                        pad = int(0.05 * bbox_width)
                        h_frame, w_frame = frame.shape[:2]
                        lx1 = max(0, x1 - pad)
                        ly1 = max(0, y1 - pad)
                        lx2 = min(w_frame, x2 + pad)
                        ly2 = min(h_frame, y2 + pad)

                        # FaceMap 3DMM landmarks
                        landmarks = self.facemap.detect(frame, (lx1, ly1, lx2, ly2))

                        # Facial attribute detection
                        attribs = self.attrib_detector.detect(frame, bbox)

                        # Head pose estimation
                        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2)
                        h, w = frame.shape[:2]
                        ex2 = min(ex2, w)
                        ey2 = min(ey2, h)

                        driver_state = None
                        if landmarks:
                            if self.facemap:
                                self.facemap.draw_full_mesh(frame, landmarks)
                            driver_state = self.analyzer.update(landmarks, pitch=0, yaw=0)

                        self.visualizer.draw_face_info(frame, bbox, landmarks, driver_state)

                fps = self.capture.get_fps()
                self.visualizer.draw_fps(frame, fps)
                self.visualizer.show("Driver Monitoring", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if fps < MIN_FPS:
                    self.logger.warning(f"FPS dropped below requirement: {fps:.2f}")
