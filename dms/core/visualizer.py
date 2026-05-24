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
        
        # 1. Projected center
        u0 = focal_length * X / Z + cx
        v0 = focal_length * Y / Z + cy
        
        # 2. Orthonormal basis in the circle's plane
        # Find a vector not parallel to normal
        if abs(normal_3d[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        u_3d = np.cross(normal_3d, ref)
        u_3d /= np.linalg.norm(u_3d)
        v_3d = np.cross(normal_3d, u_3d)
        
        # 3. Jacobian of projection [u, v] = [fX/Z + cx, fY/Z + cy]
        # J = [[f/Z, 0, -fX/Z^2],
        #      [0, f/Z, -fY/Z^2]]
        f_Z = focal_length / Z
        f_Z2 = focal_length / (Z * Z)
        J = np.array([
            [f_Z, 0, -X * f_Z2],
            [0, f_Z, -Y * f_Z2]
        ])
        
        # 4. Projected basis vectors in 2D
        a = J @ (u_3d * radius)
        b = J @ (v_3d * radius)
        
        # 5. Ellipse from two vectors p(theta) = p0 + a*cos(theta) + b*sin(theta)
        # The axes are the eigenvectors of M = a*a^T + b*b^T
        M = np.outer(a, a) + np.outer(b, b)
        
        # Eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(M)
        
        # eigh returns eigenvalues in ascending order
        # Semi-major axis is sqrt(evals[1]), semi-minor is sqrt(evals[0])
        major_axis = float(np.sqrt(max(evals[1], 1e-6)))
        minor_axis = float(np.sqrt(max(evals[0], 1e-6)))
        
        # Angle of the major axis
        angle_rad = np.arctan2(evecs[1, 1], evecs[0, 1])
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
        ✅ Proper 3D perspective projection with depth-based ellipse deformation.
        Each dot is treated as a 3D disk perpendicular to the gaze vector.
        """
        H, W = image.shape[:2]
    
        # Extract gaze vector components
        vx, vy, vz = float(v_world[0]), float(v_world[1]), float(v_world[2])
        v_unit = np.array([vx, vy, vz])
        v_unit /= np.linalg.norm(v_unit)
        
        # ──────── PINHOLE CAMERA SETUP ────────
        f = float(focal_length) if focal_length is not None else float(max(W, H))
        cx, cy = W * 0.5, H * 0.5
        Z_e = float(eye_depth)
        
        # Back-project eye position from 2D image coords to 3D world
        x0, y0 = float(eye_pos[0]), float(eye_pos[1])
        X_e = (x0 - cx) * Z_e / f
        Y_e = (y0 - cy) * Z_e / f
        
        # Trail length in 3D space
        L = float(length) * Z_e / (float(length) + f) if length else 0.0
        
        # ──────── HEAD POSE ROTATION (for trail orientation) ────────
        # Default normal is facing camera if head_pose is missing
        normal = np.array([0.0, 0.0, -1.0])
        R_yaw_only = None
        
        if head_pose is not None:
            yaw_d, pitch_d, roll_d = head_pose
            # ✅ ONLY use Yaw for orientation to keep it "thẳng dọc" (vertical)
            # ignoring Pitch and Roll prevents the "leaning" effect
            R_yaw_only = get_rotation_matrix(
                0,                  # No pitch
                np.deg2rad(-yaw_d), # Only yaw
                0                   # No roll
            )
            # Face normal in camera coords (pointing out of face)
            normal = R_yaw_only @ np.array([0.0, 0.0, -1.0])

        # Opacity based on gaze strength
        gaze_strength = np.hypot(vx, vy)
        min_opacity, max_opacity = 0.05, 0.8
        alpha_global = min_opacity + (max_opacity - min_opacity) * gaze_strength * 3
        alpha_global = float(np.clip(alpha_global, min_opacity, max_opacity))

        # ──────── TRAIL WITH DEPTH-BASED DEFORMATION ────────
        for i in range(1, num_dots + 1):
            t = i / num_dots
            t_s = t ** 2.0
            
            # 3D position of the dot
            X = X_e + t_s * L * vx
            Y = Y_e + t_s * L * (-vy)  # flip Y for image coords
            Z = Z_e + t_s * L * vz
            pos_3d = np.array([X, Y, Z])
            
            # Base radius in pixels, then convert to 3D units
            r_pixel = min_radius + (max_radius - min_radius) * t_s
            r_3d = r_pixel * Z_e / f
            
            # Project 3D disk (oriented with head Yaw) to 2D ellipse
            (u, v), (major, minor), angle_deg = self._project_circle_to_ellipse(
                pos_3d, r_3d, normal, f, cx, cy
            )
            
            px, py = int(round(u)), int(round(v))
            if not (0 <= px < W and 0 <= py < H):
                continue

            # ✅ Removed stretch_gain: dots stay as circles in 3D space.
            # They only become ellipses on-screen via perspective when the head turns.
            
            # Opacity
            alpha = (0.2 + 0.6 * t) * alpha_global
            
            # Glow size (scales with depth)
            scale_factor = Z_e / max(Z, 0.1)
            curr_glow = glow_size * t * scale_factor
            
            # Draw
            overlay = image.copy()
            
            if curr_glow > 0:
                # Glow: larger ellipse
                cv2.ellipse(
                    overlay, (px, py),
                    (int(round(major + curr_glow)), int(round(minor + curr_glow))),
                    angle_deg, 0, 360, color, -1, cv2.LINE_AA
                )
            
            # Main: ellipse (oriented with face plane)
            cv2.ellipse(
                overlay, (px, py),
                (int(round(major)), int(round(minor))),
                angle_deg, 0, 360, color, -1, cv2.LINE_AA
            )
            
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # ──────── ENDPOINT ────────
        X_end = X_e + L * vx
        Y_end = Y_e + L * (-vy)
        Z_end = Z_e + L * vz
        pos_end_3d = np.array([X_end, Y_end, Z_end])
        
        u_end = f * X_end / max(Z_end, 0.1) + cx
        v_end = f * Y_end / max(Z_end, 0.1) + cy
        end_x, end_y = int(round(u_end)), int(round(v_end))
        
        if 0 <= end_x < W and 0 <= end_y < H:
            overlay_end = image.copy()
            scale_end = Z_e / max(Z_end, 0.1)
            
            # 1. Draw center circle (projected)
            r_end_3d = 4.0 * Z_e / f  # radius of ~4px at eye_depth
            (u_c, v_c), (maj_c, min_c), ang_c = self._project_circle_to_ellipse(
                pos_end_3d, r_end_3d, normal, f, cx, cy
            )
            cv2.ellipse(overlay_end, (int(round(u_c)), int(round(v_c))), 
                        (int(round(maj_c)), int(round(min_c))), ang_c, 0, 360, color, -1, cv2.LINE_AA)
            
            # 2. Draw crosshair (+) projected in 3D
            # We project two 3D segments centered at pos_end_3d, 
            # oriented with the head pose's local 'right' and 'up' vectors.
            bar_len_3d = 8.0 * Z_e / f # length of ~8px at eye_depth
            
            # Length in pixels at this depth
            bar_len_px = int(round(8.0 * scale_end))
            
            # 1. Draw the horizontal bar (3D projected: rotates with head Yaw only)
            if R_yaw_only is not None:
                # Use the pre-calculated R_yaw_only for the crosshair horizontal bar
                vec_right = R_yaw_only @ np.array([1.0, 0.0, 0.0])
            else:
                vec_right = np.array([1.0, 0.0, 0.0])
            
            bar_len_3d = 8.0 * Z_e / f
            p1_h = pos_end_3d - vec_right * bar_len_3d
            p2_h = pos_end_3d + vec_right * bar_len_3d
            
            def _proj(p):
                return (int(round(f * p[0] / max(p[2], 0.1) + cx)), 
                        int(round(f * p[1] / max(p[2], 0.1) + cy)))
            
            cv2.line(overlay_end, _proj(p1_h), _proj(p2_h), (0, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Draw the vertical bar (2D fixed: strictly vertical on screen)
            # Center is (end_x, end_y), extending up and down
            cv2.line(overlay_end, (end_x, end_y - bar_len_px), (end_x, end_y + bar_len_px), (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.addWeighted(overlay_end, alpha_global, image, 1 - alpha_global, 0, image)
        
        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)
        