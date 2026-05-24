# core/visualizer.py
from collections import deque

import cv2
import numpy as np
from utils import facial_constants as fc
from utils.helpers import draw_bbox
from utils.helpers import draw_axis
from utils.general import get_rotation_matrix
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
    v_world_test = np.array([0.0, 0.0, -1.0])
    def draw_gaze_3d(
        self,
        image: np.ndarray,
        eye_pos: np.ndarray,
        v_world: np.ndarray,
        length: float | None = None,
        color: tuple[int, int, int] = (255, 255, 0),
        num_dots: int = 6,
        draw_eye_marker: bool = True,
        min_radius: int = 1,
        max_radius: int = 8,       
        glow_size: int = 2,        
        stretch_gain: float = 1.0, 
        stretch_grow: float = 0.2,  
        eye_depth: float = 1.0,
        focal_length: float | None = None,
        head_pose: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """
        Vẽ vector gaze theo PERSPECTIVE PROJECTION với trail dots bị biến dạng ellipse.

        Nếu `head_pose=(yaw, pitch, roll)` (degree) được truyền vào, ellipse sẽ
        xoay & foreshorten theo HEAD LOCAL FRAME thay vì chỉ theo gaze 2D:
          • angle_deg  = arctan2 của trục "phải" mắt (R @ [1,0,0]) chiếu lên ảnh
          • minor_axis = bán kính × độ dài chiếu của trục "lên" mắt (R @ [0,1,0])
          • major_axis = bán kính × độ dài chiếu trục "phải" × stretch (gaze depth)
        ⇒ nghiêng đầu (roll) → ellipse nghiêng theo;
          xoay đầu (yaw/pitch) → ellipse bẹp đúng phía bị foreshorten.
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

        L = float(length) * Z_e / (float(length) + f) if length else 0.0

        vx, vy, vz = float(v_world[0]), float(v_world[1]), float(v_world[2])

        # Hàm project an toàn
        # v_world dùng convention math 3D: +Y = UP. Trục Y của ảnh OpenCV +Y = DOWN
        # → cần đảo dấu vy khi tích lũy vào Y pixel-space để chiều cao hiển thị
        # khớp với hướng nhìn thật (pitch UP của model = mũi tên UP trên ảnh).
        def _project(t: float):
            X = X_e + t * L * vx
            Y = Y_e + t * L * (-vy)
            Z = Z_e + t * L * vz
            Z_safe = max(Z, 0.05)
            u = f * X / Z_safe + cx
            v = f * Y / Z_safe + cy
            scale = Z_e / Z_safe
            return u, v, Z_safe, scale

        # ── Opacity global ──────────
        gaze_strength = np.hypot(vx, vy)
        min_opacity, max_opacity = 0.05, 0.8
        alpha_global = min_opacity + (max_opacity - min_opacity) * gaze_strength * 3
        alpha_global = float(np.clip(alpha_global, min_opacity, max_opacity))

        # ── ELLIPSE trail với perspective deformation ──────────
        if head_pose is not None:
            yaw_d, pitch_d, roll_d = head_pose
            # Convention khớp draw_cube/draw_axis: yaw đảo dấu để +yaw = quay phải
            R = get_rotation_matrix(
                np.deg2rad(pitch_d),
                np.deg2rad(-yaw_d),
                np.deg2rad(roll_d),
            )
            # Trục "phải" và "lên" của mắt sau khi đầu xoay
            right_3d = R @ np.array([1.0, 0.0, 0.0])
            up_3d    = R @ np.array([0.0, 1.0, 0.0])

            # Chiếu lên image plane (bỏ z) → độ dài còn lại = foreshorten factor
            major_proj = float(np.hypot(right_3d[0], right_3d[1]))   # ∈ [0, 1]
            minor_proj = float(np.hypot(up_3d[0], up_3d[1]))         # ∈ [0, 1]
            angle_deg  = float(np.degrees(np.arctan2(right_3d[1], right_3d[0])))
            # Clamp tối thiểu 0.15 để không sụp về 0 khi đầu xoay 90°
            major_proj = max(0.15, major_proj)
            minor_proj = max(0.15, minor_proj)
        else:
            angle_deg  = float(np.degrees(np.arctan2(vy, vx)))
            major_proj = 1.0
            minor_proj = 1.0

        # Stretch dọc theo trục mắt — tăng khi nhìn thẳng vào camera (|vz| lớn).
        stretch_base = 1.0 + abs(vz) * stretch_gain

        # Blend chỉ trong ROI nhỏ quanh dot thay vì copy() toàn frame.
        # Trước đây image.copy() (~6 MB cho 1080p) × num_dots × 2 mắt là
        # bottleneck CPU lớn nhất của visualizer.
        def _blend_roi(cx_p, cy_p, radius_px, alpha_blend, draw_fn):
            x0 = max(0, cx_p - radius_px); y0 = max(0, cy_p - radius_px)
            x1 = min(W, cx_p + radius_px + 1); y1 = min(H, cy_p + radius_px + 1)
            if x1 <= x0 or y1 <= y0:
                return
            roi = image[y0:y1, x0:x1]
            overlay = roi.copy()
            draw_fn(overlay, x0, y0)
            cv2.addWeighted(overlay, alpha_blend, roi, 1 - alpha_blend, 0, roi)

        for i in range(1, num_dots + 1):
            t = i / num_dots
            t_s = t ** 2.0
            u, v, _, scale = _project(t_s)
            px, py = int(round(u)), int(round(v))
            if not (0 <= px < W and 0 <= py < H):
                continue

            r_base = min_radius + (max_radius - min_radius) * t_s
            r = max(1, int(round(r_base * scale)))
            alpha = (0.2 + 0.6 * t) * alpha_global

            stretch = stretch_base * (1.0 + t * stretch_grow)

            major_axis = max(1, int(round(r * major_proj * stretch)))
            minor_axis = max(1, int(round(r * minor_proj)))

            curr_glow = int(glow_size * t * scale)
            r = max(major_axis, minor_axis)
            if curr_glow > 0:
                def _draw(ov, ox, oy, _r=r, _g=curr_glow, _ang=angle_deg, _px=px, _py=py):
                    cv2.ellipse(ov, (_px - ox, _py - oy),
                                (_r + _g, _r + _g), _ang, 0, 360, color, -1, cv2.LINE_AA)
                    cv2.ellipse(ov, (_px - ox, _py - oy),
                                (_r, _r), _ang, 0, 360, color, -1, cv2.LINE_AA)
                _blend_roi(px, py, r + curr_glow + 1, alpha, _draw)

        # ── Endpoint + dấu "+" ───────────────────────────────────────────
        u_end, v_end, _, end_scale = _project(1.0)
        end_x, end_y = int(round(u_end)), int(round(v_end))
        if 0 <= end_x < W and 0 <= end_y < H:
            r_end = max(2, int(round(6 * end_scale)))
            bar = max(3, int(round(6 * end_scale)))
            radius_total = max(r_end, bar) + 1
            def _draw_end(ov, ox, oy, _re=r_end, _b=bar, _x=end_x, _y=end_y):
                cv2.circle(ov, (_x - ox, _y - oy), _re, color, -1, cv2.LINE_AA)
                cv2.line(ov, (_x - _b - ox, _y - oy), (_x + _b - ox, _y - oy), (0, 255, 255), 2)
                cv2.line(ov, (_x - ox, _y - _b - oy), (_x - ox, _y + _b - oy), (0, 255, 255), 2)
            _blend_roi(end_x, end_y, radius_total, alpha_global, _draw_end)

        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)
        