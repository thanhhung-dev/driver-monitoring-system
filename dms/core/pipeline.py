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
from utils.general import compute_euler_angles_from_rotation_matrices, get_gaze_world_vector
from detection.eye_gaze import EyeGazeEstimation

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
        facemap: FaceMap3DMMDetector | None = None,
        attrib_detector: FaceAttribDetector | None = None,
        head_pose=None,
        eye_gaze: EyeGazeEstimation | None = None,
        analyzer: DrowsinessAnalyzer | None = None,
        visualizer: Visualizer | None = None,
        device: torch.device = None,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.facemap = facemap
        self.eye_gaze = eye_gaze
        self.attrib_detector = attrib_detector
        self.head_pose = head_pose
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.device = device or torch.device("cpu")
        self.logger = setup_logger("DMSPipeline")

        # Pre-build transform cho MobileNetV2 head pose (ImageNet normalize).
        self._head_pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # EMA filters cho Gaze để giảm rung
        self._gaze_ema_alpha = 0.2
        self._gaze_pitch = None
        self._gaze_yaw = None

        # Throttle head_pose: chỉ chạy mỗi N frame để tránh lag, frame còn
        # lại tái sử dụng kết quả gần nhất (head pose ít đổi giữa 2 frame).
        self._head_pose_interval = 3
        self._head_pose_counter = 0
        self._last_head_pose = None  # (yaw, pitch, roll) độ

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

                det, kpss = self.detector.detect(frame)

                if det is None or len(det) == 0:
                    if self.visualizer is not None:
                        self.visualizer.draw_no_face_warning(frame)
                else:
                    for i, box in enumerate(det):
                        x1, y1, x2, y2 = box[:4].astype(int)
                        bbox = (x1, y1, x2, y2)
                        
                        # Get 5-point landmarks for this face
                        face_kpss = kpss[i] if kpss is not None else None

                        # FaceMap 3DMM landmarks (chỉ chạy nếu được bật)
                        landmarks = None
                        head_pose_3dmm = None # (pitch, yaw, roll) rad
                        if self.facemap is not None:
                            res = self.facemap.detect(frame, bbox, face_kpss)
                            if res is not None:
                                landmarks, head_pose_3dmm = res

                        if self.eye_gaze is not None and landmarks is not None:
                            gaze_l, gaze_r, eye_center_l, eye_center_r = self.eye_gaze.detect(frame, landmarks)

                            # ── Gaze Fusion & Smoothing ──────────────────
                            fused_gaze = None
                            if gaze_l is not None and gaze_r is not None:
                                fused_gaze = (gaze_l + gaze_r) / 2.0
                            elif gaze_l is not None:
                                fused_gaze = gaze_l
                            elif gaze_r is not None:
                                fused_gaze = gaze_r

                            if fused_gaze is not None:
                                # Apply EMA filtering
                                gp, gy = fused_gaze[0], fused_gaze[1]
                                if self._gaze_pitch is None:
                                    self._gaze_pitch, self._gaze_yaw = gp, gy
                                else:
                                    self._gaze_pitch = self._gaze_ema_alpha * gp + (1 - self._gaze_ema_alpha) * self._gaze_pitch
                                    self._gaze_yaw   = self._gaze_ema_alpha * gy + (1 - self._gaze_ema_alpha) * self._gaze_yaw
                                
                                # Compensate with head pose
                                if head_pose_3dmm is not None:
                                    hp_p, hp_y, hp_r = head_pose_3dmm
                                    v_world = get_gaze_world_vector(self._gaze_pitch, self._gaze_yaw, hp_p, hp_y, hp_r)
                                    
                                    # Vẽ cả hai mắt sử cùng một vector đã fusion (đảm bảo song song, đồng đều)
                                    if eye_center_l is not None:
                                        frame = self.visualizer.draw_gaze_3d(frame, eye_center_l, v_world, color=(0, 255, 255))
                                    if eye_center_r is not None:
                                        frame = self.visualizer.draw_gaze_3d(frame, eye_center_r, v_world, color=(0, 255, 255))
                                else:
                                    # Fallback 2D visualization (song song)
                                    fused_angle = np.array([self._gaze_pitch, self._gaze_yaw])
                                    if eye_center_l is not None:
                                        frame = self.visualizer.draw_gaze(frame, eye_center_l, fused_angle)
                                    if eye_center_r is not None:
                                        frame = self.visualizer.draw_gaze(frame, eye_center_r, fused_angle)

                        # Facial attribute detection (chỉ chạy nếu được bật)
                        attribs = None
                        if self.attrib_detector is not None:
                            attribs = self.attrib_detector.detect(frame, bbox)

                        # Head pose estimation — throttle mỗi N frame để tránh lag.
                        # Nếu đã có head_pose_3dmm, ta có thể bỏ qua model head_pose rời hoặc dùng 3DMM ghi đè.
                        if head_pose_3dmm is not None:
                            p_rad, y_rad, r_rad = head_pose_3dmm
                            head_pose_angles = (np.rad2deg(y_rad), np.rad2deg(p_rad), np.rad2deg(r_rad))
                        elif self.head_pose is not None:
                            self._head_pose_counter += 1
                            if self._head_pose_counter >= self._head_pose_interval:
                                self._head_pose_counter = 0

                                ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2)
                                h, w = frame.shape[:2]
                                ex1 = max(0, ex1)
                                ey1 = max(0, ey1)
                                ex2 = min(ex2, w)
                                ey2 = min(ey2, h)

                                head_crop = frame[ey1:ey2, ex1:ex2]
                                if head_crop.size != 0:
                                    rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                                    resized = cv2.resize(rgb, (224, 224))
                                    tensor = self._head_pose_transform(resized).unsqueeze(0).to(self.device)

                                    rot = self.head_pose(tensor)         # (1, 3, 3)
                                    euler = compute_euler_angles_from_rotation_matrices(rot)
                                    pitch_d = float(torch.rad2deg(euler[0, 0]).item())
                                    yaw_d   = float(torch.rad2deg(euler[0, 1]).item())
                                    roll_d  = float(torch.rad2deg(euler[0, 2]).item())
                                    head_pose_angles = (yaw_d, pitch_d, roll_d)
                                    self._last_head_pose = head_pose_angles

                        driver_state = None
                        if landmarks:
                            self.visualizer.draw_full_mesh(frame, landmarks)
                            if self.analyzer is not None:
                                # Prioritize 3DMM pose for analyzer if available
                                if head_pose_3dmm is not None:
                                    p_rad, y_rad, r_rad = head_pose_3dmm
                                    pitch_in = np.rad2deg(p_rad)
                                    yaw_in   = np.rad2deg(y_rad)
                                else:
                                    yaw_in   = head_pose_angles[0] if head_pose_angles else 0
                                    pitch_in = head_pose_angles[1] if head_pose_angles else 0
                                
                                driver_state = self.analyzer.update(landmarks, pitch=pitch_in, yaw=yaw_in)

                        if self.visualizer is not None:
                            self.visualizer.draw_face_info(
                                frame, bbox, landmarks, driver_state, head_pose=head_pose_angles
                            )

                fps = self.capture.get_fps()
                if self.visualizer is not None:
                    self.visualizer.draw_fps(frame, fps)
                    self.visualizer.show("Driver Monitoring", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if fps < MIN_FPS:
                    self.logger.warning(f"FPS dropped below requirement: {fps:.2f}")
