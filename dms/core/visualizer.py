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
        eye_depth: float = 1.0,
        focal_length: float | None = None,
    ) -> np.ndarray:
        """
        Vẽ vector gaze theo PERSPECTIVE PROJECTION (pinhole camera).

        Mô hình:
          • Camera intrinsics (xấp xỉ webcam): fx = fy = max(W,H), cx = W/2, cy = H/2.
          • Back-project tâm mắt 2D về 3D ở độ sâu Z_e = `eye_depth`.
          • Gaze ray trong 3D: P(t) = P_eye + t·L·v_world, với t ∈ [0, 1].
          • Project ngược về 2D: (x', y') = (fx·X/Z + cx, fy·Y/Z + cy).

        Hệ quả:
          • Nhìn thẳng vào camera (vz ≈ -1) → gaze hội tụ về (cx, cy) và "to lên"
            do Z giảm → cảm giác đang đến gần.
          • Nhìn ngang (vz ≈ 0) → Z giữ nguyên → gaze trượt trên image plane.
          • Trail dots tự động foreshorten qua factor 1/Z, không cần tính tay.
        """
        H, W = image.shape[:2]

        # ── Pinhole camera params ─────────────────────────────────────────
        f = float(focal_length) if focal_length is not None else float(max(W, H))
        cx, cy = W * 0.5, H * 0.5
        Z_e = float(eye_depth)

        x0, y0 = float(eye_pos[0]), float(eye_pos[1])

        # Back-project eye position từ 2D → 3D tại Z = Z_e
        X_e = (x0 - cx) * Z_e / f
        Y_e = (y0 - cy) * Z_e / f

        # Convert "length in pixels" → 3D distance sao cho ở góc nhỏ
        # vẫn giữ độ dài quen thuộc giống code cũ:
        #   x_proj − x0 ≈ f·L·vx / (Z_e − L)  ≈ length·vx khi L = length·Z_e/(length+f)
        L = float(length) * Z_e / (float(length) + f) if length else 0.0

        vx, vy, vz = float(v_world[0]), float(v_world[1]), float(v_world[2])

        # Hàm project an toàn (kẹp Z để không chia 0 / âm)
        def _project(t: float):
            X = X_e + t * L * vx
            Y = Y_e + t * L * vy
            Z = Z_e + t * L * vz
            Z_safe = max(Z, 0.05)            # tránh đi sau camera
            u = f * X / Z_safe + cx
            v = f * Y / Z_safe + cy
            scale = Z_e / Z_safe             # >1 khi tới gần camera, <1 khi xa
            return u, v, Z_safe, scale

        # ── Opacity global (như cũ, dùng độ lệch xy của vector) ──────────
        gaze_strength = np.hypot(vx, vy)
        min_opacity, max_opacity = 0.05, 0.8
        alpha_global = min_opacity + (max_opacity - min_opacity) * gaze_strength * 3
        alpha_global = float(np.clip(alpha_global, min_opacity, max_opacity))

        # ── Trail dots: project từng điểm 3D rồi áp 1/Z scaling ──────────
        for i in range(1, num_dots + 1):
            t = i / num_dots
            t_s = t ** 2.0                   # gần mắt thì nhỏ lâu hơn
            u, v, _, scale = _project(t_s)
            px, py = int(round(u)), int(round(v))
            if not (0 <= px < W and 0 <= py < H):
                continue

            r_base = min_radius + (max_radius - min_radius) * t_s
            r = max(1, int(round(r_base * scale)))         # foreshorten 3D
            alpha = (0.2 + 0.6 * t) * alpha_global

            overlay = image.copy()
            curr_glow = int(glow_size * t * scale)
            if curr_glow > 0:
                cv2.circle(overlay, (px, py), r + curr_glow, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # ── Endpoint + dấu "+" ───────────────────────────────────────────
        u_end, v_end, _, end_scale = _project(1.0)
        end_x, end_y = int(round(u_end)), int(round(v_end))
        if 0 <= end_x < W and 0 <= end_y < H:
            overlay_end = image.copy()
            r_end = max(2, int(round(6 * end_scale)))
            cv2.circle(overlay_end, (end_x, end_y), r_end, color, -1, cv2.LINE_AA)

            bar = max(3, int(round(6 * end_scale)))
            cv2.line(overlay_end, (end_x - bar, end_y), (end_x + bar, end_y), (0, 255, 255), 2)
            cv2.line(overlay_end, (end_x, end_y - bar), (end_x, end_y + bar), (0, 255, 255), 2)
            cv2.addWeighted(overlay_end, alpha_global, image, 1 - alpha_global, 0, image)

        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)