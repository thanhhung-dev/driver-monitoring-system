from collections import deque

import cv2
import numpy as np
from features.landmarks import constants as fc
from utils.helpers import draw_bbox
from utils.helpers import draw_axis
from utils.helpers import draw_head_direction_arrow

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_normal = (255, 255, 255)
        self.gaze_history = deque(maxlen=8)
        self.gaze_history_l = deque(maxlen=8)
        self.gaze_history_r = deque(maxlen=8)


    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), self.font, 1, self.color_normal, 2)

    def draw_face_info(self, frame, bbox, landmarks, driver_state, head_pose=None, draw_box=True):
        """Vẽ bbox + trục head-pose (nếu có).

        Args:
            frame:        Ảnh BGR cần vẽ lên (in-place).
            bbox:         (x1, y1, x2, y2).
            landmarks:    List 68 (x, y) hoặc None — landmark mesh đã được vẽ
                          trực tiếp trong pipeline qua FaceMap3DMMDetector.draw_full_mesh.
            driver_state: Trạng thái driver (chưa dùng, để mở rộng cảnh báo).
            head_pose:    Tuple (yaw, pitch, roll) độ, hoặc None để không vẽ trục.
            draw_box:     False để ẩn bbox SCRFD (vùng transition 80–85°).
        """
        # Bounding box với corner-accent
        if draw_box:
            draw_bbox(frame, bbox, self.color_normal, fixed_size=(250, 250))


        # 3D head-pose axes
        if head_pose is not None:
            yaw, pitch, roll = head_pose
            draw_axis(frame, yaw, pitch, roll, list(bbox))
            if landmarks is not None:
                draw_head_direction_arrow(frame, landmarks, yaw, pitch, roll)

        return frame
    
    def draw_full_mesh(self, image, landmarks, color=(255, 255, 0), radius=2):
        if landmarks is None:
            return image
        lm = np.asarray(landmarks, dtype=np.int32)

        def poly(idxs, closed=True, thickness=1):
            cv2.polylines(image, [lm[idxs]], closed, color, thickness, cv2.LINE_AA)

        def dots(idxs):
            for i in idxs:
                cv2.circle(image, (int(lm[i, 0]), int(lm[i, 1])), radius, color, -1, cv2.LINE_AA)

        # Mat + long mày
        poly(fc.LEFT_EYE_INDICES, closed=True, thickness=1)
        dots(fc.LEFT_EYE_INDICES)
        poly(fc.RIGHT_EYE_INDICES, closed=True, thickness=1)
        dots(fc.RIGHT_EYE_INDICES)
        poly(fc.EYE_BROW_LEFT, closed=False, thickness=1)
        dots(fc.EYE_BROW_LEFT)
        poly(fc.EYE_BROW_RIGHT, closed=False, thickness=1)
        dots(fc.EYE_BROW_RIGHT)

        # Mui
        poly([27, 28, 29, 30], closed=False, thickness=1)
        dots(fc.NOISE_INDICES)
        poly([31, 30, 35], closed=False, thickness=1)
        poly([31, 33, 35], closed=False, thickness=1)
        dots(fc.NOISE_TRIANGLE_INDICES)

        # Moi
        poly(fc.OUTER_LIPS_INDICES, closed=True, thickness=1)
        dots(fc.OUTER_LIPS_INDICES)
        poly(fc.INNER_LIPS_INDICES, closed=True, thickness=1)
        dots(fc.INNER_LIPS_INDICES)

        return image
    def _project_circle_to_ellipse(
        self,
        center_3d: np.ndarray,
        radius: float,
        normal_3d: np.ndarray,
        focal_length: float,
        cx: float,
        cy: float
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        """
        Projects a small 3D circle to a 2D ellipse using the Jacobian of the projection.
        Args:
            center_3d: (X, Y, Z) in 3D
            radius: Radius of the 3D circle
            normal_3d: Unit normal vector of the circle's plane
            focal_length: Pinhole camera focal length
            cx, cy: Principal point
        Returns:
            (center_2d, axes_2d, angle_deg) compatible with cv2.ellipse
        """
        X, Y, Z = center_3d
        Z = max(Z, 0.01)
        u0 = focal_length * X / Z + cx
        v0 = focal_length * Y / Z + cy
        if abs(normal_3d[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u_3d = np.cross(normal_3d, ref)
        u_3d /= np.linalg.norm(u_3d)
        v_3d = np.cross(normal_3d, u_3d)
        f_Z = focal_length / Z
        f_Z2 = focal_length / (Z * Z)
        J = np.array([
            [f_Z, 0, -X * f_Z2],
            [0, f_Z, -Y * f_Z2]
        ])
        a = J @ (u_3d * radius)
        b = J @ (v_3d * radius)
        M = np.outer(a, a) + np.outer(b, b)
        evals, evecs = np.linalg.eigh(M)
        
        major_axis = float(np.sqrt(max(evals[1], 1e-6)))
        minor_axis = float(np.sqrt(max(evals[0], 1e-6)))
        min_ratio = 0.1
        minor_axis = max(minor_axis, major_axis * min_ratio)
        major_vec = evecs[:, 1].copy()
        if major_vec[1] < 0:  # Always point "downward" in image coords
            major_vec = -major_vec
        angle_rad = np.arctan2(major_vec[1], major_vec[0])
        angle_deg = float(np.degrees(angle_rad))
        
        return (u0, v0), (major_axis, minor_axis), angle_deg

    def draw_gaze_3d(
        self,
        image: np.ndarray,
        eye_pos: np.ndarray,
        v_world: np.ndarray,
        length: float | None = None,
        color: tuple[int, int, int] = (255, 255, 0),
        num_dots: int = 6,
        min_radius: int = 1,
        max_radius: int = 14,
        glow_size: int = 1,
        eye_depth: float = 1.0,
        start_offset: float = 0.15,
        dot_spacing_power: float = 2.0,
        focal_length: float | None = None,
        head_pose: tuple[float, float, float] | None = None,
        crosshair_size: float = 2.0,
        crosshair_thickness: int = 2,
        opacity_scale: float = 1.0,
        show_crosshair: bool = True,
    ) -> np.ndarray:
        """
        Proper 3D perspective projection with depth-based ellipse deformation.
        Each dot is treated as a 3D disk perpendicular to the gaze vector.
        """
        H, W = image.shape[:2]

        # Extract gaze vector components
        v = np.asarray(v_world, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-6 or eye_depth <= 0:
            return image
        v_unit = v / norm
        vx, vy, vz = float(v_unit[0]), float(v_unit[1]), float(v_unit[2])

        # ──────── PINHOLE CAMERA SETUP ────────
        f = float(focal_length) if focal_length is not None else float(max(W, H))
        cx, cy = W * 0.5, H * 0.5
        Z_e = float(eye_depth)
        x0, y0 = float(eye_pos[0]), float(eye_pos[1])
        X_e = (x0 - cx) * Z_e / f
        Y_e = (y0 - cy) * Z_e / f
        L = float(length) * Z_e / (float(length) + f) if length else 0.0
        # Tinh Toan Khi Ngieng
        gaze_strength = np.hypot(vx, vy)
        min_opacity, max_opacity = 0.05, 0.9
        t = float(np.clip(gaze_strength ** 0.6, 0.0, 1.0))
        alpha_global = min_opacity + (max_opacity - min_opacity) * t
        alpha_global *= float(np.clip(opacity_scale, 0.0, 1.0))

        gaze_normal = np.array([vx, vy, vz])
        gaze_normal /= np.linalg.norm(gaze_normal)
        progress_denominator = max(num_dots - 1, 1)
        spacing_power = max(float(dot_spacing_power), 1.0)
        eye_to_gaze_offset = float(np.clip(start_offset, 0.0, 1.0))

        # ──────── TRAIL DOTS: nhỏ ở gần mắt (t=0) -> to dần khi tiến ra xa/gần camera (t=1) ────────
        for i in range(num_dots):
            t = i / progress_denominator
            distance_progress = eye_to_gaze_offset + (1.0 - eye_to_gaze_offset) * t

            X = X_e + distance_progress * L * vx
            Y = Y_e + distance_progress * L * vy
            Z = Z_e + distance_progress * L * vz
            pos_3d = np.array([X, Y, Z])

            # Kich thuoc hien thi mong muon, monotonic tuyet doi theo t (khong phu thuoc Z)
            r_pixel_target = min_radius * (1.0 - t) + max_radius * t

            # Chi dung phep chieu de lay TI LE mep hinh (foreshortening) va goc nghieng,
            # KHONG lay do lon tuyet doi tu day (radius=1.0 co dinh)
            (u, v), (probe_major, probe_minor), angle_deg = self._project_circle_to_ellipse(
                pos_3d, radius=1.0, normal_3d=gaze_normal, focal_length=f, cx=cx, cy=cy
            )
            shape_ratio = probe_minor / max(probe_major, 1e-6)  # <= 1
            major = r_pixel_target
            minor = r_pixel_target * shape_ratio

            px, py = int(round(u)), int(round(v))
            if not (0 <= px < W and 0 <= py < H):
                continue

            alpha = alpha_global * t
            print(f"[major, minor, alpha, t]: [{major:.2f}, {minor:.2f}, {alpha:.2f}, {t:.2f}]")

            scale_factor = Z_e / max(Z, 0.1)
            curr_glow = glow_size * (0.5 + 0.5 * t) * scale_factor
            overlay = image.copy()
            if curr_glow > 0:
                cv2.ellipse(
                    overlay, (px, py),
                    (int(round(major + curr_glow)), int(round(minor + curr_glow))),
                    angle_deg, 0, 360, color, -1, cv2.LINE_AA
                )

            cv2.ellipse(
                overlay, (px, py),
                (int(round(major)), int(round(minor))),
                angle_deg, 0, 360, color, -1, cv2.LINE_AA
            )

            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # ──────── END-POINT CROSSHAIR (giu nguyen, khong doi) ────────
        X_end = X_e + L * vx
        Y_end = Y_e + L * vy
        Z_end = Z_e + L * vz
        pos_end_3d = np.array([X_end, Y_end, Z_end])

        u_end = f * X_end / max(Z_end, 0.1) + cx
        v_end = f * Y_end / max(Z_end, 0.1) + cy
        end_x, end_y = int(round(u_end)), int(round(v_end))

        if 0 <= end_x < W and 0 <= end_y < H:
            overlay_end = image.copy()
            # Kich thuoc blob nen cua crosshair KHONG con phu thuoc max_radius nua,
            # dieu chinh truc tiep qua tham so crosshair_size de thu nho tuy y
            marker_radius = crosshair_size
            # Chi dung phep chieu de lay TI LE mep hinh (shape), khong lay do lon
            # tuyet doi (radius=1.0 co dinh) -> kich thuoc marker khong con phu
            # thuoc Z_end, tranh 2 crosshair (trai/phai mat) bi lech size nhau
            _, (probe_maj_end, probe_min_end), ang_end = self._project_circle_to_ellipse(
                pos_end_3d, 1.0, gaze_normal, f, cx, cy
            )
            end_shape_ratio = probe_min_end / max(probe_maj_end, 1e-6)
            major_i = max(1, int(round(marker_radius)))
            minor_i = max(1, int(round(marker_radius * end_shape_ratio)))
            cv2.ellipse(overlay_end, (end_x, end_y), (major_i, minor_i), ang_end, 0, 360, color, -1, cv2.LINE_AA)

            if show_crosshair and crosshair_size > 0:
                bar_len = max(1, int(round(crosshair_size + major_i)))
                crosshair_color = (0, 255, 255)
                cv2.line(
                    overlay_end,
                    (end_x, end_y - bar_len),
                    (end_x, end_y + bar_len),
                    crosshair_color, crosshair_thickness, cv2.LINE_AA,
                )
                cv2.line(
                    overlay_end,
                    (end_x - bar_len, end_y),
                    (end_x + bar_len, end_y),
                    crosshair_color, crosshair_thickness, cv2.LINE_AA,
                )
            cv2.addWeighted(overlay_end, alpha_global, image, 1 - alpha_global, 0, image)

        return image
    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)
        
