import cv2
import numpy as np

import dataclasses

from core.frame_context import FrameContext
from core.visualizer import Visualizer
from detection.eye_gaze import EyeGazeEstimation, _eye_indices
from utils.gaze_debug_helper import GazeDebugLogger, classify_gaze, direction_arrow
from utils.general import get_rotation_matrix


class GazeStage:
    """Eye gaze detection with EMA smoothing, dynamic length, and 3D visualization.

    This is the most complex stage -- it handles:
    - Eye gaze inference (with eye-closure fallback)
    - Dynamic gaze length based on inter-eye distance
    - EMA smoothing for stable arrows
    - Side factor adjustment based on yaw
    - 3D gaze arrow drawing on the frame
    """
    def __init__(
        self,
        eye_gaze: EyeGazeEstimation | None,
        visualizer: Visualizer | None,
        pitch_offset: float = 0.0,
        yaw_offset: float = 0.0,
    ) -> None:
        self._eye_gaze = eye_gaze
        self._visualizer = visualizer
        self._last_gaze_l = None
        self._last_gaze_r = None
        self._last_center_l = None
        self._last_center_r = None
        self._last_age_l = 999
        self._last_age_r = 999
        self._prev_gaze_length = 200.0
        self._gaze_length_alpha = 0.2
        self.pitch_offset = pitch_offset
        self.yaw_offset = yaw_offset
    MAX_FALLBACK_AGE = 5
    STALE_AGE = 2
    OPACITY_MIN = 0.5       
    OPACITY_MAX = 1.0        
    OPACITY_FULL_DEG = 25.0
    MAX_HEAD_YAW_FOR_GAZE = 55
    PROFILE_GAZE_LENGTH_SCALE = 1.5
    PROFILE_CROSSHAIR_SIZE = 0.3  

    def _gaze_opacity_scale(
        self, head_pose: tuple[float, float, float] | None
    ) -> float:
        """Opacity gradient: mờ khi nhìn thẳng, rõ khi nghiêng.

        Tỉ lệ tuyến tính theo |head_yaw|:
          |yaw| = 0°  → OPACITY_MIN  (gần như ẩn)
          |yaw| >= OPACITY_FULL_DEG → OPACITY_MAX  (rõ hoàn toàn)
        """
        if head_pose is None:
            return self.OPACITY_MAX
        yaw_abs = abs(float(head_pose[0]))
        t = np.clip(yaw_abs / self.OPACITY_FULL_DEG, 0.0, 1.0)
        return float(self.OPACITY_MIN + (self.OPACITY_MAX - self.OPACITY_MIN) * t)

    @property
    def name(self) -> str:
        return "gaze"

    def _pitchyaw_to_vec(
        self,
        pitchyaw: np.ndarray,
        head_pose: tuple[float, float, float] | None = None,
        head_rotation_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose eye gaze + head pose → world-space gaze vector.

        Công thức: gaze_world = R_head × gaze_eye_vector

        EyeNet trả gaze_eye = hướng mắt TƯƠNG ĐỐI so với đầu (eye-in-head).
        Phải rotate bởi R_head mới ra gaze thật trong world space.

        Ví dụ: head quay trái 30°, mắt nhìn phải 10°
          → EyeNet: +10° (eye-local)
          → World:  R_head × (+10°) = -30° + 10° = -20° (vẫn nhìn trái) ✓

        App convention:
          yaw > 0 = subject looks RIGHT → x positive → arrow RIGHT on screen
          pitch > 0 = subject looks UP → y = -sin(pitch) < 0 → arrow UP on screen
        """
        pitch = float(pitchyaw[0])
        yaw = float(pitchyaw[1])
        sin_p = np.sin(pitch)
        cos_p = np.cos(pitch)
        sin_y = np.sin(yaw)
        cos_y = np.cos(yaw)
        x = cos_p * sin_y
        y = -sin_p
        z = cos_p * cos_y
        vec = np.array([x, y, z], dtype=np.float64)
        if head_rotation_matrix is not None:
            vec = head_rotation_matrix @ vec
        elif head_pose is not None:
            yaw_d, pitch_d, roll_d = head_pose
            R = get_rotation_matrix(
                np.deg2rad(pitch_d),
                np.deg2rad(yaw_d),
                np.deg2rad(roll_d),
            )
            vec = R @ vec
        return vec.astype(np.float32)

    @staticmethod
    def _eye_centers_from_landmarks(
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Lấy tâm 2 mắt từ landmarks (không cần inference)."""
        if landmarks is None or len(landmarks) == 0:
            return None, None
        lm = np.asarray(landmarks, dtype=np.float32)
        left_idx, right_idx = _eye_indices(len(lm))
        try:
            cl = lm[left_idx, :2].mean(axis=0)
            cr = lm[right_idx, :2].mean(axis=0)
            return cl, cr
        except Exception:
            return None, None

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._eye_gaze is None or ctx.landmarks is None:
            return ctx
        use_head_fallback = (
            ctx.head_pose is not None
            and abs(ctx.head_pose[0]) > self.MAX_HEAD_YAW_FOR_GAZE
        )

        if use_head_fallback:
            eye_center_l, eye_center_r = self._eye_centers_from_landmarks(ctx.landmarks)
            synthetic = np.zeros(2, dtype=np.float32)
            gaze_l = synthetic if eye_center_l is not None else None
            gaze_r = synthetic if eye_center_r is not None else None
            self._eye_gaze._prev_gaze_l = None
            self._eye_gaze._prev_gaze_r = None
        else:
            gaze_l, gaze_r, eye_center_l, eye_center_r = self._eye_gaze.detect(
                ctx.frame, ctx.landmarks
            )

        if gaze_l is not None:
            gaze_l = gaze_l.copy()
            gaze_l[1] = -gaze_l[1]
        if gaze_r is not None:
            gaze_r = gaze_r.copy()
            gaze_r[1] = -gaze_r[1]
        if gaze_l is not None:
            gaze_l = gaze_l + np.array([self.pitch_offset, self.yaw_offset], dtype=np.float32)
        if gaze_r is not None:
            gaze_r = gaze_r + np.array([self.pitch_offset, self.yaw_offset], dtype=np.float32)
        if self._debug_logger is not None:
            self._debug_logger.log_avg(gaze_l, gaze_r, ctx.frame_number)
        if gaze_l is not None:
            self._last_gaze_l = gaze_l
            self._last_age_l = 0
        else:
            self._last_age_l += 1
        if gaze_r is not None:
            self._last_gaze_r = gaze_r
            self._last_age_r = 0
        else:
            self._last_age_r += 1
        if eye_center_l is not None:
            self._last_center_l = eye_center_l
        if eye_center_r is not None:
            self._last_center_r = eye_center_r

        fallback_l = self._last_gaze_l if self._last_age_l <= self.MAX_FALLBACK_AGE else None
        fallback_r = self._last_gaze_r if self._last_age_r <= self.MAX_FALLBACK_AGE else None
        display_gaze_l = gaze_l if gaze_l is not None else fallback_l
        display_gaze_r = gaze_r if gaze_r is not None else fallback_r
        display_center_l = eye_center_l if eye_center_l is not None else self._last_center_l
        display_center_r = eye_center_r if eye_center_r is not None else self._last_center_r
        if display_center_l is not None and display_center_r is not None:
            eye_dist = np.linalg.norm(display_center_l - display_center_r)
            raw_gaze_length = 40 * (100.0 / max(eye_dist, 1.0))
            raw_gaze_length = np.clip(raw_gaze_length, 50, 60)
            a = self._gaze_length_alpha
            gaze_length = a * raw_gaze_length + (1.0 - a) * self._prev_gaze_length
            self._prev_gaze_length = gaze_length
        else:
            gaze_length = self._prev_gaze_length
        vec_world = None
        is_gaze_fresh = (
            (gaze_l is not None or gaze_r is not None)
            or (
                (display_gaze_l is not None and self._last_age_l <= self.STALE_AGE)
                or (display_gaze_r is not None and self._last_age_r <= self.STALE_AGE)
            )
        )
        has_eye_gaze = display_gaze_l is not None or display_gaze_r is not None
        use_eye_gaze = has_eye_gaze and is_gaze_fresh and not use_head_fallback

        if use_eye_gaze:
            if display_gaze_l is not None and display_gaze_r is not None:
                gaze_avg = (display_gaze_l + display_gaze_r) / 2.0
            elif display_gaze_l is not None:
                gaze_avg = display_gaze_l
            else:
                gaze_avg = display_gaze_r

            vec = self._pitchyaw_to_vec(
                gaze_avg, head_pose=ctx.head_pose,
                head_rotation_matrix=ctx.head_rotation_matrix,
            )

            yaw_val = np.abs(float(gaze_avg[1]))
            side_factor = np.clip(1.0 + yaw_val / 0.5, 1.0, 1.3)
            gaze_length *= side_factor

            opacity_scale = self._gaze_opacity_scale(ctx.head_pose)
            gaze_render = {
                "vec": vec,
                "length": gaze_length,
                "fallback": False
            }
            vec_world = vec

        elif ctx.head_pose is not None and not use_eye_gaze:
            head_vec = self._pitchyaw_to_vec(
                np.zeros(2, dtype=np.float32), head_pose=ctx.head_pose,
                head_rotation_matrix=ctx.head_rotation_matrix,
            )
            profile_gaze_length = gaze_length * self.PROFILE_GAZE_LENGTH_SCALE
            opacity_scale = self._gaze_opacity_scale(ctx.head_pose)
            gaze_render = {"vec": head_vec, "length": profile_gaze_length, "fallback": True}
            vec_world = head_vec


        if self._debug_logger is not None and self._debug_logger._enabled:
            self._draw_debug_overlay(
                ctx.frame,
                display_gaze_l, display_gaze_r,
                ctx.head_pose,
                ctx.landmarks,
                display_center_l, display_center_r,
                use_head_fallback=not use_eye_gaze,
            )
        if not use_eye_gaze:
            h, w = ctx.frame.shape[:2]
            cv2.putText(
                ctx.frame, "HEAD-GAZE", (w - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA,
            )
        if 'gaze_render' not in locals():
            gaze_render = None
        else:
            gaze_render["center_l"] = display_center_l
            gaze_render["center_r"] = display_center_r
            gaze_render["head_pose"] = ctx.head_pose
            gaze_render["opacity_scale"] = opacity_scale
            
        return dataclasses.replace(
            ctx,
            gaze_l=display_gaze_l,
            gaze_r=display_gaze_r,
            gaze_vec_world=vec_world,
            eye_center_l=display_center_l,
            eye_center_r=display_center_r,
            gaze_render_data=gaze_render,
        )