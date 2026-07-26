import dataclasses

import numpy as np

from features.face.detector import FaceDetector
from features.head_pose.feedback import HeadPoseFeedback
from pipeline.context import FrameContext


class DetectStage:
    """Face detection with interval-based skipping.

    Runs the detector every N frames; reuses cached results otherwise.
    Forces a detect when previous result was empty (lost tracking).

    Extreme pose handling (hysteresis):
    - The previous frame's head yaw is read from a shared HeadPoseFeedback
      holder (HeadPoseStage runs after DetectStage, so it cannot be read from
      the current ctx).
    - Enter extreme_pose_mode when |yaw| >= ENTER_EXTREME_YAW (88°), exit when
      |yaw| < EXIT_EXTREME_YAW (80°). Hysteresis tránh nhấp nháy ở biên.
    - In extreme mode SCRFD vẫn chạy để bám bbox; landmark/gaze/attrib/
      drowsiness bị skip. Head pose VẪN chạy (kể cả khi mất mặt nghiêng, trên
      bbox gần nhất) để liên tục đo yaw và thoát extreme đúng lúc khi tài xế
      quay mặt về (<80°).
    - When SCRFD mất mặt nghiêng → giữ bbox gần nhất (face_lost_extreme_pose
      = True) để vẫn còn khung detection. Nếu mất mặt quá MAX_LOST_FRAMES frame
      → reset hẳn về normal (xoá bbox + yaw cũ) để không kẹt extreme vĩnh viễn.
    """

    ENTER_EXTREME_YAW = 88.0
    EXIT_EXTREME_YAW = 80.0
    MAX_LOST_FRAMES = 15  # ~1s @ 15 FPS: mất mặt lâu hơn → reset về normal

    def __init__(
        self,
        detector: FaceDetector,
        interval: int = 5,
        feedback: HeadPoseFeedback | None = None,
        enter_extreme_yaw: float | None = None,
        exit_extreme_yaw: float | None = None,
        max_lost_frames: int | None = None,
    ) -> None:
        self._detector = detector
        self._interval = interval
        self._counter = 0
        self._last_det = None
        self._last_kpss = None
        # Bbox tốt gần nhất — dùng để giữ khung khi SCRFD mất mặt nghiêng ở
        # góc lớn (extreme mode) → không bị "no detection".
        self._last_bbox = None
        self._last_face_kpss = None
        # Đếm số frame liên tiếp SCRFD mất mặt khi đang extreme.
        self._lost_counter = 0
        self._max_lost_frames = (
            max_lost_frames if max_lost_frames is not None else self.MAX_LOST_FRAMES
        )
        # Head pose của frame TRƯỚC, lấy qua holder chia sẻ (không thể đọc từ
        # ctx vì HeadPoseStage chạy sau DetectStage trong cùng một frame).
        self._feedback = feedback
        self._enter_yaw = (
            enter_extreme_yaw if enter_extreme_yaw is not None else self.ENTER_EXTREME_YAW
        )
        self._exit_yaw = (
            exit_extreme_yaw if exit_extreme_yaw is not None else self.EXIT_EXTREME_YAW
        )
        self._extreme_mode = False
        # EMA smoothing cho yaw — chống noise từ head pose model ở góc extreme.
        self._yaw_ema: float | None = None
        self._ema_alpha = 0.3  # 0.3 = 30% giá trị mới + 70% cũ

    @property
    def name(self) -> str:
        return "face_detect"

    def _prev_head_pose(self):
        return self._feedback.last_head_pose if self._feedback is not None else None

    def _update_extreme_mode(self) -> None:
        """Hysteresis: vào extreme khi |yaw|>=enter, ra khi |yaw|<exit.

        Yaw được EMA smooth trước khi so ngưỡng để chống noise từ model
        ở góc extreme (dao động ±5-10° giữa các frame).
        """
        prev = self._prev_head_pose()
        if prev is None:
            return
        raw_yaw = abs(float(prev[0]))
        if self._extreme_mode:
            # Đang extreme → EMA smooth để THOÁT mượt (chống noise ±5-10°),
            # không nhấp nháy ở biên.
            if self._yaw_ema is None:
                self._yaw_ema = raw_yaw
            else:
                self._yaw_ema = self._ema_alpha * raw_yaw + (1 - self._ema_alpha) * self._yaw_ema
            if self._yaw_ema < self._exit_yaw:
                self._extreme_mode = False
                self._yaw_ema = None
        else:
            # Chưa extreme → VÀO NGAY trên yaw thô (head pose / keypoint đã ép
            # bão hòa 88° khi tới profile) để không bị EMA làm trễ vào extreme.
            if raw_yaw >= self._enter_yaw:
                self._extreme_mode = True
                self._yaw_ema = raw_yaw

    def process(self, ctx: FrameContext) -> FrameContext:
        self._update_extreme_mode()

        # ── Run SCRFD with interval-based skipping (cả 2 chế độ) ────────
        self._counter += 1
        need_detect = (
            self._counter >= self._interval
            or self._last_det is None
            or len(self._last_det) == 0
        )
        if need_detect:
            self._counter = 0
            det, kpss = self._detector.detect(ctx.frame)
            self._last_det, self._last_kpss = det, kpss
        else:
            det, kpss = self._last_det, self._last_kpss

        has_face = det is not None and len(det) > 0
        if has_face:
            box = det[0]
            x1, y1, x2, y2 = box[:4].astype(int)
            bbox = (x1, y1, x2, y2)
            face_kpss = kpss[0] if kpss is not None else None
            self._last_bbox = bbox
            self._last_face_kpss = face_kpss

        # ── Extreme mode: tắt gaze/landmark, giữ khung detection ───────
        if self._extreme_mode:
            if has_face:
                self._lost_counter = 0
                return dataclasses.replace(
                    ctx, bbox=bbox, face_kpss=face_kpss,
                    extreme_pose_mode=True, face_lost_extreme_pose=False,
                )
            # SCRFD mất mặt nghiêng → đếm lost frames.
            self._lost_counter += 1
            if self._lost_counter >= self._max_lost_frames:
                # Mất mặt quá lâu → reset về normal, không kẹt extreme.
                self._extreme_mode = False
                self._lost_counter = 0
                if self._feedback is not None:
                    self._feedback.last_head_pose = None
                return ctx
            return dataclasses.replace(
                ctx, bbox=self._last_bbox, face_kpss=self._last_face_kpss,
                extreme_pose_mode=True, face_lost_extreme_pose=True,
            )

        # ── Normal mode ────────────────────────────────────────────────
        if not has_face:
            return ctx
        return dataclasses.replace(ctx, bbox=bbox, face_kpss=face_kpss)
