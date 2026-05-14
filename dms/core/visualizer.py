# core/visualizer.py
from collections import deque

import cv2
import numpy as np
from utils import facial_constants as fc
from utils.helpers import draw_bbox
from utils.helpers import draw_axis
from typing import Tuple, List

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_warning = (0, 0, 255)
        self.color_normal = (0, 255, 0)
        self.gaze_history = deque(maxlen=8)
        # Tách history riêng cho từng mắt để vệt không bị nhảy qua lại
        self.gaze_history_l = deque(maxlen=8)
        self.gaze_history_r = deque(maxlen=8)


    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), self.font, 1, self.color_normal, 2)

    def draw_no_face_warning(self, frame):
        cv2.putText(frame, "No face detected", (20, 80), self.font, 1, self.color_warning, 2)

    def draw_face_info(self, frame, bbox, landmarks, driver_state, head_pose=None):
        """Vẽ bbox + trục head-pose (nếu có).

        Args:
            frame:        Ảnh BGR cần vẽ lên (in-place).
            bbox:         (x1, y1, x2, y2).
            landmarks:    List 68 (x, y) hoặc None — landmark mesh đã được vẽ
                          trực tiếp trong pipeline qua FaceMap3DMMDetector.draw_full_mesh.
            driver_state: Trạng thái driver (chưa dùng, để mở rộng cảnh báo).
            head_pose:    Tuple (yaw, pitch, roll) độ, hoặc None để không vẽ trục.
        """
        # Bounding box với corner-accent
        draw_bbox(frame, bbox, self.color_normal)

        # 3D head-pose axes (chỉ vẽ khi có giá trị yaw/pitch/roll)
        if head_pose is not None:
            yaw, pitch, roll = head_pose
            draw_axis(frame, yaw, pitch, roll, list(bbox))

        return frame
    
    def draw_full_mesh(self, image, landmarks, color=(255, 255, 0), radius=2):
        """Vẽ trực tiếp lên frame (không upscale/downscale) → nhanh hơn ~10× so
        với phiên bản cũ (vốn resize 640×480 ↔ 1280×960 mỗi frame)."""
        if landmarks is None:
            return image

        CYAN = (255, 255, 0)
        lm = np.asarray(landmarks, dtype=np.int32)

        def poly(idxs, closed=True, thickness=1):
            cv2.polylines(image, [lm[idxs]], closed, CYAN, thickness, cv2.LINE_AA)

        def dots(idxs):
            for i in idxs:
                cv2.circle(image, (int(lm[i, 0]), int(lm[i, 1])), radius, color, -1, cv2.LINE_AA)

        # Mắt + lông mày
        poly(fc.LEFT_EYE_INDICES, closed=True, thickness=1)
        dots(fc.LEFT_EYE_INDICES)
        poly(fc.RIGHT_EYE_INDICES, closed=True, thickness=1)
        dots(fc.RIGHT_EYE_INDICES)
        poly(fc.EYE_BROW_LEFT, closed=False, thickness=1)
        dots(fc.EYE_BROW_LEFT)
        poly(fc.EYE_BROW_RIGHT, closed=False, thickness=1)
        dots(fc.EYE_BROW_RIGHT)

        # Mũi
        poly([27, 28, 29, 30], closed=False, thickness=1)
        dots(fc.NOISE_INDICES)
        poly([31, 30, 35], closed=False, thickness=1)
        poly([31, 33, 35], closed=False, thickness=1)
        dots(fc.NOISE_TRIANGLE_INDICES)

        # Môi
        poly(fc.OUTER_LIPS_INDICES, closed=True, thickness=1)
        dots(fc.OUTER_LIPS_INDICES)
        poly(fc.INNER_LIPS_INDICES, closed=True, thickness=1)
        dots(fc.INNER_LIPS_INDICES)

        return image
    
    def draw_gaze_3d(
        self,
        image: np.ndarray,
        eye_pos: np.ndarray,
        v_world: np.ndarray,
        length: float | None = None,
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        eye_side: str = 'l',
        num_dots: int = 6,
        draw_eye_marker: bool = True,
        min_radius: int = 1,
        max_radius: int = 12,
        glow_size: int = 4,
    ) -> np.ndarray:
        """
        Vẽ vector gaze 3D dưới dạng một chuỗi hình tròn nối từ mắt đến endpoint.
        - Opacity giảm dần khi gaze ở gần trung tâm (giữa mắt).
        - Hình tròn ở gần mắt nhỏ và mờ, càng xa càng to và rõ.
        """
        H, W = image.shape[:2]

        # World space: +X phải, +Y xuống → chiếu thẳng lên ảnh
        dx = float(v_world[0]) * length
        dy = float(v_world[1]) * length

        x0, y0 = float(eye_pos[0]), float(eye_pos[1])

        # Tính độ mạnh của gaze (độ lệch khỏi trung tâm)
        gaze_strength = np.sqrt(v_world[0]**2 + v_world[1]**2)
        min_opacity = 0.05
        max_opacity = 0.8

        # Alpha global tỉ lệ thuận với gaze_strength (giảm khi ở giữa mắt)
        alpha_global = min_opacity + (max_opacity - min_opacity) * gaze_strength * 3
        alpha_global = np.clip(alpha_global, min_opacity, max_opacity)

        # Vẽ num_dots hình tròn dọc theo đoạn từ (x0,y0) đến (x1,y1)
        for i in range(1, num_dots + 1):
            t = i / num_dots
            # Dùng bình phương (t**2) để chấm ở gần mắt nhỏ lâu hơn
            t_s = t**2.0
            px = int(round(x0 + dx * t_s))
            py = int(round(y0 + dy * t_s))

            if not (0 <= px < W and 0 <= py < H):
                continue

            r = int(min_radius + (max_radius - min_radius) * t_s)
            alpha = (0.2 + 0.6 * t) * alpha_global

            # Sử dụng overlay tạm thời cho từng dot để tránh tích tụ opacity
            overlay = image.copy()
            # Glow cũng nhỏ dần về phía mắt
            curr_glow = int(glow_size * t)
            if curr_glow > 0:
                cv2.circle(overlay, (px, py), r + curr_glow, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Vẽ Endpoint (điểm cuối) cũng với alpha_global
        end_x = int(x0 + dx)
        end_y = int(y0 + dy)
        if 0 <= end_x < W and 0 <= end_y < H:
            overlay_end = image.copy()
            # Vòng tròn to ở cuối
            cv2.circle(overlay_end, (end_x, end_y), 6, color, -1, cv2.LINE_AA)
            # Dấu + trắng ở giữa
            cv2.line(overlay_end, (end_x - 6, end_y), (end_x + 6, end_y), (0, 255, 255), 2)
            cv2.line(overlay_end, (end_x, end_y - 6), (end_x, end_y + 6), (0, 255, 255), 2)
            cv2.addWeighted(overlay_end, alpha_global, image, 1 - alpha_global, 0, image)

        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)