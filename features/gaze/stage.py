import cv2
import numpy as np

import dataclasses

from features.gaze.debug import GazeDebugLogger, classify_gaze, direction_arrow
from features.gaze.estimator import EyeGazeEstimation, _eye_indices
from features.gaze.telemetry import (
    BlinkRateTracker,
    classify_gaze_zone,
    compute_eye_loc_mm,
    compute_head_loc_mm,
    eye_openness_percent,
)
from pipeline.context import FrameContext
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
        visualizer: object | None,
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
        self._last_age_l = 999
        self._last_age_r = 999
        self._prev_gaze_length = 200.0
        self._gaze_length_alpha = 0.2
        self.pitch_offset = pitch_offset
        self.yaw_offset = yaw_offset
        self._debug_logger = debug_logger
        self._blink_tracker = BlinkRateTracker(window_sec=30.0, closed_threshold=30.0)
    MAX_FALLBACK_AGE = 5
    STALE_AGE = 2
    OPACITY_MIN = 0.5      
    OPACITY_MAX = 1.0       
    OPACITY_FULL_DEG = 25.0
    MAX_HEAD_YAW_FOR_GAZE = 70
    PROFILE_GAZE_LENGTH_SCALE = 0.65
    GAZE_LENGTH_MIN = 50
    GAZE_LENGTH_MAX = 90
    PROFILE_GAZE_LENGTH = 120
    # Tinh Khi Nham Mat
    EYE_OPEN_FULL_PERCENT = 30.0
    EYE_CLOSED_LENGTH_SCALE = 0.4

    def _eye_open_length_scale(self, landmarks) -> float:
        """Hệ số [EYE_CLOSED_LENGTH_SCALE..1.0] theo độ mở mắt.

        Mắt càng nhắm -> hệ số càng nhỏ -> gaze càng ngắn.
        """
        if landmarks is None or len(landmarks) == 0:
            return 1.0
        lm = np.asarray(landmarks, dtype=np.float32)
        left_idx, right_idx = _eye_indices(len(lm))
        ops = [
            v for v in (
                eye_openness_percent(lm, left_idx),
                eye_openness_percent(lm, right_idx),
            ) if v is not None
        ]
        if not ops:
            return 1.0
        avg_op = float(np.mean(ops))
        t = float(np.clip(avg_op / self.EYE_OPEN_FULL_PERCENT, 0.0, 1.0))
        return self.EYE_CLOSED_LENGTH_SCALE + (1.0 - self.EYE_CLOSED_LENGTH_SCALE) * t

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

    def _adjust_gaze(self, gaze: np.ndarray | None) -> np.ndarray | None:
        """Đảo trục Y + cộng offset calibration."""
        if gaze is None:
            return None
        adjusted = gaze.copy()
        adjusted[1] = -adjusted[1]
        return adjusted + np.array([self.pitch_offset, self.yaw_offset], dtype=np.float32)

    def _update_tracking(
        self,
        gaze: np.ndarray | None,
        center: np.ndarray | None,
        last_gaze: np.ndarray | None,
        last_age: int,
        last_center: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int, np.ndarray | None, np.ndarray | None]:
        """Cập nhật gaze/center tracking, trả về (new_last_gaze, new_age, new_last_center, display_gaze)."""
        if gaze is not None:
            last_gaze, last_age = gaze, 0
        else:
            last_age += 1
        if center is not None:
            last_center = center
        fallback = last_gaze if last_age <= self.MAX_FALLBACK_AGE else None
        display_gaze = gaze if gaze is not None else fallback
        return last_gaze, last_age, last_center, display_gaze

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

        gaze_l = self._adjust_gaze(gaze_l)
        gaze_r = self._adjust_gaze(gaze_r)

        if self._debug_logger is not None:
            self._debug_logger.log_avg(gaze_l, gaze_r, ctx.frame_number)

        self._last_gaze_l, self._last_age_l, self._last_center_l, display_gaze_l = (
            self._update_tracking(gaze_l, eye_center_l, self._last_gaze_l, self._last_age_l, self._last_center_l)
        )
        self._last_gaze_r, self._last_age_r, self._last_center_r, display_gaze_r = (
            self._update_tracking(gaze_r, eye_center_r, self._last_gaze_r, self._last_age_r, self._last_center_r)
        )
        display_center_l = eye_center_l if eye_center_l is not None else self._last_center_l
        display_center_r = eye_center_r if eye_center_r is not None else self._last_center_r

        gaze_length = self.GAZE_LENGTH_MAX
        # Nhắm mắt -> rút ngắn gaze (tránh dài ra khi nhắm mắt + cúi xuống).
        eye_open_scale = self._eye_open_length_scale(ctx.landmarks)
        gaze_length *= eye_open_scale
        vec_world = None
        gaze_render_data = None
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

            # Liếc ngang kéo dài nhẹ, giới hạn 1.15 để không quá dài.
            yaw_val = np.abs(float(gaze_avg[1]))
            side_factor = np.clip(1.0 + yaw_val / 0.5, 1.0, 1.3)
            gaze_length *= side_factor

            opacity_scale = self._gaze_opacity_scale(ctx.head_pose)
            gaze_render_data = {
                "vec": vec,
                "center_l": display_center_l,
                "center_r": display_center_r,
                "length": gaze_length,
                "head_pose": ctx.head_pose,
                "opacity_scale": opacity_scale,
                "fallback": False,
                "show_crosshair": True,
            }
            vec_world = vec

        elif ctx.head_pose is not None and not use_eye_gaze:
            #TODO Đang Trùng Logic Với Draw direction arrow  -> common Nó sau
            yaw_h, pitch_h, roll_h = ctx.head_pose
            y = np.deg2rad(yaw_h)
            p = np.deg2rad(-pitch_h)  
            r = np.deg2rad(roll_h)

            Rx = np.array([[1, 0, 0],
                        [0, np.cos(p), -np.sin(p)],
                        [0, np.sin(p),  np.cos(p)]])
            Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                        [ 0,         1, 0        ],
                        [-np.sin(y), 0, np.cos(y)]])
            Rz = np.array([[np.cos(r), -np.sin(r), 0],
                        [np.sin(r),  np.cos(r), 0],
                        [0,          0,         1]])
            R = Rz @ Ry @ Rx
            head_vec = (R @ np.array([0.0, 0.0, 1.0])).astype(np.float32)

            profile_gaze_length = self.PROFILE_GAZE_LENGTH * eye_open_scale
            opacity_scale = self._gaze_opacity_scale(ctx.head_pose)
            gaze_render_data = {
                "vec": head_vec,
                "center_l": display_center_l,
                "center_r": display_center_r,
                "length": profile_gaze_length,
                "head_pose": ctx.head_pose,
                "opacity_scale": opacity_scale,
                "fallback": True,
                "show_crosshair": False,
            }
            vec_world = head_vec

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
            gaze_render_data=gaze_render_data,
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
            if use_head_fallback and head_pose is not None:
                # head_pose chắc chắn tồn tại → pitch_h/yaw_h đã được gán ở trên.
                lines.append(("GAZE DIR [HEAD]", color_label))
                lines.append((f"  P={pitch_h:+6.1f}  Y={yaw_h:+6.1f}", (0, 165, 255)))
            else:
                lines.append(("GAZE DIR (eye+head)+", color_label))
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
