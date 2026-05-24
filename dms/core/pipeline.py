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


        self._head_pose_interval = 5
        self._head_pose_counter = 0
        self._last_head_pose = None  # (yaw, pitch, roll) độ

        # NOTE: pitch_offset cộng trực tiếp vào pitch của model trước khi tính
        # y = sin(pitch). Trong tọa độ ảnh, +y hướng xuống, nên pitch âm ⇒ mũi
        # tên hướng LÊN. Giá trị -0.20 (cũ) gây bias ~11.5° hướng lên ngay cả
        # khi nhìn thẳng. Đặt 0.0 mặc định; nếu model có bias thật, hãy đo bằng
        # cách in `raw_l/raw_r` khi đối tượng nhìn thẳng và tinh chỉnh tại đây.
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        # Bật log gaze (raw model output + vector sau offset) để debug từng bước.
        # Tắt khi đã hết bug (in mỗi ~30 frame để tránh spam).
        self._gaze_debug = True
        self._gaze_debug_counter = 0
        self._gaze_debug_every = 15   # in mỗi N frame

        # Lưu trạng thái gaze cuối cùng để tránh bị mất khi nháy mắt
        self._last_gaze_l = None
        self._last_gaze_r = None
        self._last_center_l = None
        self._last_center_r = None

        # Skip face-detector mỗi N frame — bbox khuôn mặt giữa các frame
        # gần như không đổi (driver ngồi cố định), nên không cần detect mỗi frame.
        # Lần detect tiếp theo sẽ "tự refresh" khi counter chạm ngưỡng hoặc
        # khi mất tracking (det rỗng).
        self._det_interval = 5
        self._det_counter = 0
        self._last_det = None
        self._last_kpss = None

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

    def pitchyaw_to_vector(self, pitchyaws: np.ndarray) -> np.ndarray:
        """Convert pitch and yaw angles to unit gaze vectors."""
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])  
        out[:, 1] = sin[:, 0]                         
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    def _pitchyaw_to_vec(self, pitchyaw: np.ndarray) -> np.ndarray:
        """Convert a single [pitch, yaw] (radians) to a 3D unit gaze vector (x, y, z).

        Note: x is negated so that x>0 maps to the viewer's RIGHT on screen
        (model yaw convention is opposite of image x-axis after webcam mirror).
        """
        pitch = float(pitchyaw[0])
        yaw = float(pitchyaw[1])
        
        sin_p = np.sin(pitch)
        cos_p = np.cos(pitch)
        sin_y = np.sin(yaw)
        cos_y = np.cos(yaw)
        
        x = cos_p * -(sin_y)
        y = -sin_p
        z = cos_p * cos_y
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _run_loop(self) -> None:
        self._prev_gaze_length: float = 200.0
        self._gaze_length_alpha: float = 0.3
        """Core frame-processing loop."""
        import time
        _t_acc = {"read": 0.0, "det": 0.0, "facemap": 0.0, "head": 0.0,
                  "gaze": 0.0, "attrib": 0.0, "viz": 0.0, "show": 0.0, "total": 0.0}
        _t_n = 0
        with torch.no_grad():
            while True:
                _t_loop = time.perf_counter()
                _t0 = time.perf_counter()
                ret, frame = self.capture.read_frame()
                _t_acc["read"] += time.perf_counter() - _t0
                if not ret:
                    break

                if not self.capture._is_video_file:
                    frame = cv2.flip(frame, 1)

                # Skip face-detection most frames; chỉ chạy detector mỗi
                # `_det_interval` frame hoặc khi lần trước mất tracking.
                self._det_counter += 1
                need_detect = (
                    self._det_counter >= self._det_interval
                    or self._last_det is None
                    or len(self._last_det) == 0
                )
                if need_detect:
                    self._det_counter = 0
                    _t0 = time.perf_counter()
                    det, kpss = self.detector.detect(frame)
                    _t_acc["det"] += time.perf_counter() - _t0
                    self._last_det, self._last_kpss = det, kpss
                else:
                    det, kpss = self._last_det, self._last_kpss

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
                            _t0 = time.perf_counter()
                            facemap_out = self.facemap.detect(frame, bbox, face_kpss)
                            _t_acc["facemap"] += time.perf_counter() - _t0
                            if facemap_out is not None:
                                landmarks, facemap_pose = facemap_out

                        # Tính head_pose trước eye_gaze để có thể truyền vào visualizer
                        head_pose_angles = self._last_head_pose
                        if self.head_pose is not None:
                            self._head_pose_counter += 1
                            if self._head_pose_counter >= self._head_pose_interval:
                                self._head_pose_counter = 0
                                _t0 = time.perf_counter()

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
                                _t_acc["head"] += time.perf_counter() - _t0

                        if self.eye_gaze is not None and landmarks is not None:
                            _t0 = time.perf_counter()
                            gaze_l, gaze_r, eye_center_l, eye_center_r = self.eye_gaze.detect(frame, landmarks)
                            _t_acc["gaze"] += time.perf_counter() - _t0

                            # ── DEBUG: in pitch vs yaw từng tầng pipeline ──
                            #   PITCH (model): +up / -down  (Qualcomm EyeNet)
                            #   YAW   (model): +left / -right (subject perspective)
                            #   VEC y (math 3D, +Y=UP): +up / -down
                            #   Visualizer flip vy → ảnh hiển thị: math +Y → mũi tên UP
                            if self._gaze_debug:
                                self._gaze_debug_counter += 1
                                if self._gaze_debug_counter % self._gaze_debug_every == 0:
                                    def _pdeg(g):
                                        return None if g is None else (np.degrees(float(g[0])), np.degrees(float(g[1])))
                                    pl = _pdeg(gaze_l); pr = _pdeg(gaze_r)
                                    pl_s = "  None        " if pl is None else f"P={pl[0]:+6.1f}° Y={pl[1]:+6.1f}°"
                                    pr_s = "  None        " if pr is None else f"P={pr[0]:+6.1f}° Y={pr[1]:+6.1f}°"
                                    self.logger.info(f"[GAZE-RAW]  L[{pl_s}]  R[{pr_s}]")

                                    if gaze_l is not None and gaze_r is not None:
                                        avg = (gaze_l + gaze_r) / 2.0
                                        self.logger.info(f"[GAZE-AVG]  P={np.degrees(float(avg[0])):+6.1f}° Y={np.degrees(float(avg[1])):+6.1f}°")
                                        v = self._pitchyaw_to_vec(avg)
                                        # vy là math-Y (+UP). Sau khi visualizer flip dấu,
                                        # vy>0 sẽ vẽ UP trên ảnh, vy<0 sẽ vẽ DOWN.
                                        sign_p = "UP  " if v[1] > 0.05 else ("DOWN" if v[1] < -0.05 else "FLAT")
                                        sign_y = "LEFT" if v[0] < -0.05 else ("RIGHT" if v[0] > 0.05 else "STR ")
                                        self.logger.info(f"[GAZE-VEC]  x={v[0]:+.3f}({sign_y})  y={v[1]:+.3f}({sign_p})  z={v[2]:+.3f}  ← hướng mũi tên trên ảnh")

                                    if head_pose_angles is not None:
                                        yh, ph, rh = head_pose_angles
                                        self.logger.info(f"[HEAD-POSE] P={ph:+6.1f}° Y={yh:+6.1f}° R={rh:+6.1f}°  (so sánh với GAZE-AVG)")

                                    self.logger.info(f"[OFFSET]    pitch_offset={np.degrees(self.pitch_offset):+5.1f}° yaw_offset={np.degrees(self.yaw_offset):+5.1f}°")
                                    self.logger.info("-" * 60)

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
                            if display_center_l is not None and display_center_r is not None:
                                eye_dist = np.linalg.norm(display_center_l - display_center_r)
                                raw_gaze_length = 200 * (100.0 / max(eye_dist, 1.0))
                                raw_gaze_length = np.clip(raw_gaze_length, 80, 300)
                                
                                # EMA smoothing
                                a = self._gaze_length_alpha
                                gaze_length = a * raw_gaze_length + (1.0 - a) * self._prev_gaze_length
                                self._prev_gaze_length = gaze_length
                            else:
                                gaze_length = self._prev_gaze_length

                            _t0 = time.perf_counter()

                            # Tính gaze trung bình
                            if display_gaze_l is not None and display_gaze_r is not None:
                                gaze_avg = (display_gaze_l + display_gaze_r) / 2.0
                                
                                # Điều chỉnh length theo hướng nhìn (Yaw)
                                yaw_val = np.abs(float(gaze_avg[1]))
                                side_factor = np.clip(yaw_val / 0.3, 0.7, 1.0)
                                gaze_length *= side_factor
                                
                                vec = self._pitchyaw_to_vec(gaze_avg)
                                if display_center_l is not None:
                                    frame = self.visualizer.draw_gaze_3d(frame, display_center_l, vec, length=gaze_length,focal_length=1000, head_pose=head_pose_angles)
                                if display_center_r is not None:
                                    frame = self.visualizer.draw_gaze_3d(frame, display_center_r, vec, length=gaze_length,focal_length=1000, head_pose=head_pose_angles)
                            else:
                                # Trường hợp chỉ có 1 mắt hoặc dùng dữ liệu cũ của 1 mắt
                                current_gaze = display_gaze_l if display_gaze_l is not None else display_gaze_r
                                current_center = display_center_l if display_gaze_l is not None else display_center_r
                                
                                if current_gaze is not None:
                                    yaw_val = np.abs(float(current_gaze[1]))
                                    side_factor = np.clip(yaw_val / 0.3, 0.7, 1.0)
                                    gaze_length *= side_factor
                                    
                                    if current_center is not None:
                                        frame = self.visualizer.draw_gaze_3d(frame, current_center, self._pitchyaw_to_vec(current_gaze), length=gaze_length,  focal_length=1000,head_pose=head_pose_angles)
                            _t_acc["viz"] += time.perf_counter() - _t0

                        # Facial attribute detection (chỉ chạy nếu được bật)
                        attribs = None
                        if self.attrib_detector is not None:
                            _t0 = time.perf_counter()
                            attribs = self.attrib_detector.detect(frame, bbox)
                            _t_acc["attrib"] += time.perf_counter() - _t0

                        driver_state = None
                        if landmarks is not None:
                            _t0 = time.perf_counter()
                            self.visualizer.draw_full_mesh(frame, landmarks)
                            _t_acc["viz"] += time.perf_counter() - _t0
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
                    _t0 = time.perf_counter()
                    self.visualizer.draw_fps(frame, fps)
                    self.visualizer.show("Driver Monitoring", frame)
                    _t_acc["show"] += time.perf_counter() - _t0

                _t_acc["total"] += time.perf_counter() - _t_loop
                _t_n += 1
                if _t_n >= 30:
                    self.logger.info(
                        "[PROFILE ms/frame] " + " | ".join(
                            f"{k}={(_t_acc[k]/_t_n)*1000:6.1f}"
                            for k in ("read", "det", "facemap", "head",
                                      "gaze", "attrib", "viz", "show", "total")
                        ) + f"  (~{1.0/(_t_acc['total']/_t_n):.1f} FPS)"
                    )
                    for k in _t_acc: _t_acc[k] = 0.0
                    _t_n = 0

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
