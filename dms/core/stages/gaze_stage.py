import cv2
import numpy as np

import dataclasses

from core.frame_context import FrameContext
from core.visualizer import Visualizer
from detection.eye_gaze import EyeGazeEstimation, _eye_indices
from utils.gaze_debug_helper import GazeDebugLogger, classify_gaze, direction_arrow
from utils.general import get_rotation_matrix
from utils.telemetry import (
    BlinkRateTracker,
    classify_gaze_zone,
    compute_eye_loc_mm,
    compute_head_loc_mm,
    eye_openness_percent,
)


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
        debug_logger: GazeDebugLogger | None = None,
    ) -> None:
        self._eye_gaze = eye_gaze
        self._visualizer = visualizer
        self._last_gaze_l = None
        self._last_gaze_r = None
        self._last_center_l = None
        self._last_center_r = None
        # Age (frames) since each eye last produced a fresh gaze. Sau khi
        # vượt MAX_FALLBACK_AGE thì coi như eye cache đã stale → không vẽ
        # arrow nữa (tránh "ghost arrow" khi head quay nghiêng kéo dài).
        self._last_age_l = 999
        self._last_age_r = 999
        self._prev_gaze_length = 200.0
        self._gaze_length_alpha = 0.2
        self.pitch_offset = pitch_offset
        self.yaw_offset = yaw_offset
        self._debug_logger = debug_logger
        # Blink rate tracker — đếm số blink/sec qua sliding window 30s.
        self._blink_tracker = BlinkRateTracker(window_sec=30.0, closed_threshold=30.0)

    # Số frame tối đa giữ cache khi mắt mất tín hiệu (blink ~1-3 frame OK,
    # head-turn kéo dài thì cache stale phải bị clear).
    MAX_FALLBACK_AGE = 5
    STALE_AGE = 2
    DIM_YAW_RANGE = 10.0
    DIM_OPACITY_SCALE = 0.3

    def _gaze_opacity_scale(
        self, head_pose: tuple[float, float, float] | None
    ) -> float:
        """Mờ nhạt gaze khi head yaw nằm trong khoảng ~thẳng (|yaw| <= range)."""
        if head_pose is None:
            return 1.0
        if abs(float(head_pose[0])) <= self.DIM_YAW_RANGE:
            return self.DIM_OPACITY_SCALE
        return 1.0

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

        # Rotate eye-local gaze vector by head pose → gaze-in-world.
        # Prefer raw rotation matrix (avoids Euler decomposition/recomposition error).
        # Fallback to Euler angles only if raw R unavailable.
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

    MAX_HEAD_YAW_FOR_GAZE = 55
    PROFILE_GAZE_LENGTH_SCALE = 0.65
    PROFILE_CROSSHAIR_SIZE = 0.3

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

        # ── Hard fallback: yaw > 50° → skip ONNX entirely ────────────────
        # Ở góc head yaw lớn, eye crop quá foreshortened → ONNX output
        # không tin cậy. Dùng synthetic gaze = (0,0) + head rotation.
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

        # EyeNet yaw convention is opposite of the app/head-pose convention.
        # Convert once here so all downstream code uses yaw+ = screen/right.
        if gaze_l is not None:
            gaze_l = gaze_l.copy()
            gaze_l[1] = -gaze_l[1]
        if gaze_r is not None:
            gaze_r = gaze_r.copy()
            gaze_r[1] = -gaze_r[1]

        # Apply calibration offsets
        if gaze_l is not None:
            gaze_l = gaze_l + np.array([self.pitch_offset, self.yaw_offset], dtype=np.float32)
        if gaze_r is not None:
            gaze_r = gaze_r + np.array([self.pitch_offset, self.yaw_offset], dtype=np.float32)

        # Debug logging
        if self._debug_logger is not None:
            self._debug_logger.log_avg(gaze_l, gaze_r, ctx.frame_number)

        # ── Update last-known + age cache ─────────────────────────────────
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

        # Dynamic gaze length based on inter-eye distance
        if display_center_l is not None and display_center_r is not None:
            eye_dist = np.linalg.norm(display_center_l - display_center_r)
            raw_gaze_length = 60 * (100.0 / max(eye_dist, 1.0))
            raw_gaze_length = np.clip(raw_gaze_length, 50, 180)
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
            # ── Eye gaze available: compose eye + head ────────────────────
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
            if display_center_l is not None:
                self._visualizer.draw_gaze_3d(
                    ctx.frame, display_center_l, vec,
                    length=gaze_length, focal_length=1000, head_pose=ctx.head_pose,
                    opacity_scale=opacity_scale,
                )
            if display_center_r is not None:
                self._visualizer.draw_gaze_3d(
                    ctx.frame, display_center_r, vec,
                    length=gaze_length, focal_length=1000, head_pose=ctx.head_pose,
                    opacity_scale=opacity_scale,
                )
            vec_world = vec

        elif ctx.head_pose is not None and not use_eye_gaze:
            # ── Eye detection	fail: fallback to head direction ─────────
            # gaze_eye = (0,0) → R_head × [0,0,1] = head direction vector.
            head_vec = self._pitchyaw_to_vec(
                np.zeros(2, dtype=np.float32), head_pose=ctx.head_pose,
                head_rotation_matrix=ctx.head_rotation_matrix,
            )
            profile_gaze_length = gaze_length * self.PROFILE_GAZE_LENGTH_SCALE
            if display_center_l is not None:
                self._visualizer.draw_gaze_3d(
                    ctx.frame, display_center_l, head_vec,
                    length=profile_gaze_length, focal_length=1000, head_pose=ctx.head_pose,
                    num_dots=7, max_radius=12,
                    crosshair_size=self.PROFILE_CROSSHAIR_SIZE,
                    show_crosshair=True,
                )
            if display_center_r is not None:
                self._visualizer.draw_gaze_3d(
                    ctx.frame, display_center_r, head_vec,
                    length=profile_gaze_length, focal_length=1000, head_pose=ctx.head_pose,
                    num_dots=7, max_radius=12,
                    crosshair_size=self.PROFILE_CROSSHAIR_SIZE,
                    show_crosshair=True,
                )
            vec_world = head_vec

        # ── Debug overlay ─────────────────────────────────────────────────
        if self._debug_logger is not None and self._debug_logger._enabled:
            self._draw_debug_overlay(
                ctx.frame,
                display_gaze_l, display_gaze_r,
                ctx.head_pose,
                ctx.landmarks,
                display_center_l, display_center_r,
                use_head_fallback=not use_eye_gaze,
            )

        # ── On-frame indicator khi dùng head direction ────────────────────
        if not use_eye_gaze:
            h, w = ctx.frame.shape[:2]
            cv2.putText(
                ctx.frame, "HEAD-GAZE", (w - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA,
            )

        return dataclasses.replace(
            ctx,
            gaze_l=display_gaze_l,
            gaze_r=display_gaze_r,
            gaze_vec_world=vec_world,
            eye_center_l=display_center_l,
            eye_center_r=display_center_r,
        )

    def _draw_debug_overlay(
        self,
        frame: np.ndarray,
        gaze_l: np.ndarray | None,
        gaze_r: np.ndarray | None,
        head_pose: tuple[float, float, float] | None = None,
        landmarks: np.ndarray | None = None,
        center_l: np.ndarray | None = None,
        center_r: np.ndarray | None = None,
        use_head_fallback: bool = False,
    ) -> None:
        """
        Vẽ bảng debug Qualcomm-style ở góc trái dưới frame:
          EYE OPENNESS  L%  R%
          EYE BLINK     blinks/sec
          HEAD LOC      X Y Z (mm)
          EYE LOC       L: X Y Z   R: X Y Z
          HEAD DIR      P Y R (deg)
          GAZE DIR      P Y (deg)
          GAZE ZONE     <zone name>
        """
        if gaze_l is None and gaze_r is None and head_pose is None:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        color_label = (180, 180, 180)
        color_value = (0, 255, 255)
        color_head  = (255, 200, 0)
        color_zone  = (0, 200, 255)
        bg = (0, 0, 0)

        lines: list[tuple[str, tuple[int, int, int]]] = []

        # ── EYE OPENNESS + BLINK ──────────────────────────────────────────
        op_l = op_r = None
        if landmarks is not None and len(landmarks) > 0:
            lm = np.asarray(landmarks, dtype=np.float32)
            left_idx, right_idx = _eye_indices(len(lm))
            op_l = eye_openness_percent(lm, left_idx)
            op_r = eye_openness_percent(lm, right_idx)

        if op_l is not None or op_r is not None:
            avg_op = np.mean([v for v in (op_l, op_r) if v is not None])
            blink_per_sec = self._blink_tracker.update(float(avg_op))
            sL = f"{op_l:5.1f}%" if op_l is not None else "  ?  "
            sR = f"{op_r:5.1f}%" if op_r is not None else "  ?  "
            lines.append(("EYE OPENNESS", color_label))
            lines.append((f"  L:{sL}   R:{sR}", color_value))
            lines.append(("EYE BLINK", color_label))
            lines.append((f"  {blink_per_sec:.2f} /s", color_value))

        # ── HEAD LOC + EYE LOC (mm) ───────────────────────────────────────
        head_loc = compute_head_loc_mm(landmarks, frame.shape[:2]) if landmarks is not None else None
        if head_loc is not None:
            X, Y, Z = head_loc
            lines.append(("HEAD LOC (mm)", color_label))
            lines.append((f"  X={X:+6.0f}  Y={Y:+6.0f}  Z={Z:+6.0f}", color_value))

            el = compute_eye_loc_mm(center_l, frame.shape[:2], Z) if center_l is not None else None
            er = compute_eye_loc_mm(center_r, frame.shape[:2], Z) if center_r is not None else None
            if el is not None or er is not None:
                lines.append(("EYE LOC (mm)", color_label))
                if el is not None:
                    lines.append((f"  L: {el[0]:+6.0f} {el[1]:+6.0f} {el[2]:+6.0f}", color_value))
                if er is not None:
                    lines.append((f"  R: {er[0]:+6.0f} {er[1]:+6.0f} {er[2]:+6.0f}", color_value))

        # ── HEAD DIR (PYR) ────────────────────────────────────────────────
        if head_pose is not None:
            yaw_h, pitch_h, roll_h = head_pose
            lines.append(("HEAD DIR (PYR)", color_label))
            lines.append((f"  P={pitch_h:+6.1f}  Y={yaw_h:+6.1f}  R={roll_h:+6.1f}", color_head))

        # ── GAZE DIR + ZONE ───────────────────────────────────────────────
        gaze_avg = None
        if gaze_l is not None and gaze_r is not None:
            gaze_avg = (gaze_l + gaze_r) / 2.0
        elif gaze_l is not None:
            gaze_avg = gaze_l
        elif gaze_r is not None:
            gaze_avg = gaze_r

        if gaze_avg is not None:
            gaze_p_deg = float(np.degrees(gaze_avg[0]))
            gaze_y_deg = float(np.degrees(gaze_avg[1]))
            # Combined direction (head + eye) cho zone classification.
            if head_pose is not None:
                yaw_h, pitch_h, _ = head_pose
                total_p = pitch_h + gaze_p_deg
                total_y = yaw_h + gaze_y_deg
            else:
                total_p, total_y = gaze_p_deg, gaze_y_deg

            zone = classify_gaze_zone(total_p, total_y)
            if use_head_fallback:
                lines.append(("GAZE DIR [HEAD]", color_label))
                lines.append((f"  P={pitch_h:+6.1f}  Y={yaw_h:+6.1f}", (0, 165, 255)))
            else:
                lines.append(("GAZE DIR (eye+head)", color_label))
                lines.append((f"  P={total_p:+6.1f}  Y={total_y:+6.1f}", color_value))
            lines.append(("GAZE ZONE", color_label))
            lines.append((f"  {zone}", color_zone))

        # ── Render ────────────────────────────────────────────────────────
        if not lines:
            return
        h, w = frame.shape[:2]
        line_h = 18
        x0 = 10
        box_h = line_h * len(lines) + 16
        y0 = h - box_h - 10
        max_w = max(cv2.getTextSize(t, font, 0.45, 1)[0][0] for t, _ in lines) + 20

        cv2.rectangle(frame, (x0 - 5, y0 - 5), (x0 + max_w, y0 + box_h), bg, -1)
        for i, (text, color) in enumerate(lines):
            y = y0 + i * line_h
            cv2.putText(frame, text, (x0, y + 13), font, 0.45, color, 1, cv2.LINE_AA)
