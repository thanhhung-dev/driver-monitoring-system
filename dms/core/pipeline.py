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


        self._head_pose_interval = 3
        self._head_pose_counter = 0
        self._last_head_pose = None  # (yaw, pitch, roll) độ

        self.pitch_offset = -0.20 
        self.yaw_offset = 0.0

        # Lưu trạng thái gaze cuối cùng để tránh bị mất khi nháy mắt
        self._last_gaze_l = None
        self._last_gaze_r = None
        self._last_center_l = None
        self._last_center_r = None

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

    def _pitchyaw_to_vec(self, g: np.ndarray) -> np.ndarray:
        # Khi nhắm mắt, EyeGaze module set g=[0,0] (sentinel "nhìn vào tâm mắt").
        # Bỏ qua offset trong trường hợp này để mũi tên đúng là hướng thẳng,
        # không bị ngước lên do pitch_offset.
        if float(g[0]) == 0.0 and float(g[1]) == 0.0:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        pitch, yaw = float(g[0]) + self.pitch_offset, float(g[1]) + self.yaw_offset

        x = -np.cos(pitch) * np.sin(yaw)
        y =  np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)

        return np.array([x, y, z], dtype=np.float32)

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
                        facemap_pose = None
                        if self.facemap is not None:
                            facemap_out = self.facemap.detect(frame, bbox, face_kpss)
                            if facemap_out is not None:
                                landmarks, facemap_pose = facemap_out

                        if self.eye_gaze is not None and landmarks is not None:
                            gaze_l, gaze_r, eye_center_l, eye_center_r = self.eye_gaze.detect(frame, landmarks)

                            # Cập nhật bộ nhớ gaze cuối cùng nếu detect thành công
                            if gaze_l is not None: self._last_gaze_l = gaze_l
                            if gaze_r is not None: self._last_gaze_r = gaze_r
                            if eye_center_l is not None: self._last_center_l = eye_center_l
                            if eye_center_r is not None: self._last_center_r = eye_center_r

                            # Sử dụng lại dữ liệu cũ nếu nhắm mắt (detect trả về None)
                            display_gaze_l = gaze_l if gaze_l is not None else self._last_gaze_l
                            display_gaze_r = gaze_r if gaze_r is not None else self._last_gaze_r
                            display_center_l = eye_center_l if eye_center_l is not None else self._last_center_l
                            display_center_r = eye_center_r if eye_center_r is not None else self._last_center_r

                            # Tính toán length động dựa trên khoảng cách 2 mắt (pixel)
                            gaze_length = 200 # Mặc định
                            if display_center_l is not None and display_center_r is not None:
                                eye_dist = np.linalg.norm(display_center_l - display_center_r)
                                gaze_length = 200 * (100.0 / max(eye_dist, 1.0))
                                gaze_length = np.clip(gaze_length, 80, 300)

                            # Tính gaze trung bình
                            if display_gaze_l is not None and display_gaze_r is not None:
                                gaze_avg = (display_gaze_l + display_gaze_r) / 2.0
                                vec = self._pitchyaw_to_vec(gaze_avg)

                                # Foreshorten 3D: nhìn càng ngang → đoạn càng ngắn
                                # (giống ellipse bị bẹp theo |z|). Clamp tối thiểu
                                # 0.25 để không biến mất hẳn khi nhìn ngang gắt.
                                gaze_length *= max(0.25, abs(float(vec[2])))

                                if display_center_l is not None:
                                    frame = self.visualizer.draw_gaze_3d(frame, display_center_l, vec, length=gaze_length, eye_side='l')
                                if display_center_r is not None:
                                    frame = self.visualizer.draw_gaze_3d(frame, display_center_r, vec, length=gaze_length, eye_side='r')
                            else:
                                # Trường hợp chỉ có 1 mắt hoặc dùng dữ liệu cũ của 1 mắt
                                current_gaze = display_gaze_l if display_gaze_l is not None else display_gaze_r
                                current_center = display_center_l if display_gaze_l is not None else display_center_r

                                if current_gaze is not None:
                                    vec = self._pitchyaw_to_vec(current_gaze)
                                    gaze_length *= max(0.25, abs(float(vec[2])))

                                    if current_center is not None:
                                        frame = self.visualizer.draw_gaze_3d(frame, current_center, vec, length=gaze_length, eye_side='l')

                        # Facial attribute detection (chỉ chạy nếu được bật)
                        attribs = None
                        if self.attrib_detector is not None:
                            attribs = self.attrib_detector.detect(frame, bbox)

                        # Head pose estimation — throttle mỗi N frame để tránh lag.
                        head_pose_angles = self._last_head_pose
                        if self.head_pose is not None:
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
                        if landmarks is not None:
                            self.visualizer.draw_full_mesh(frame, landmarks)
                            if self.analyzer is not None:
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
