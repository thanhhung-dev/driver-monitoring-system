import dataclasses

import cv2
import numpy as np

from core.frame_context import FrameContext
from utils.helpers import expand_bbox
from utils.general import get_rotation_matrix
from utils.onnx_providers import make_session

import onnx


class HeadPoseStage:
    """Head pose estimation using SixDRepNet ONNX with interval-based skipping.

    Input  : BGR crop 224×224, float32 NCHW, ImageNet normalized
    Output : 3×3 rotation matrix → Euler angles (yaw, pitch, roll) degrees
    """

    # ImageNet normalization
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # EMA thích nghi: lệch nhỏ (đứng yên) → alpha thấp cho mượt;
    # lệch lớn (quay nhanh) → alpha cao để bám kịp, hết "trễ bò dần".
    _EMA_ALPHA_MIN = 0.35   # khi gần như đứng yên
    _EMA_ALPHA_MAX = 0.9    # khi quay nhanh
    _EMA_FAST_DELTA = 25.0  # độ lệch (deg) coi là "quay nhanh"

    def __init__(self, model_dir: str, interval: int = 2, ema_alpha: float = 0.4) -> None:
        self._interval = interval
        self._counter = 0
        self._last_head_pose = None
        self._last_R = None
        self._ema_alpha = ema_alpha

        # Load ONNX model with external data (resolves paths relative to onnx file)
        onnx_path = f"{model_dir}/pose_estimator.onnx"
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        model_bytes = onnx_model.SerializeToString()
        self._session = make_session(model_bytes)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "head_pose"

    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """BGR crop → ImageNet-normalized NCHW float32.

        ⚠️ Phải GIỮ aspect ratio như pipeline gốc của SixDRepNet:
            Resize(shorter side → 224) + CenterCrop(224)
        KHÔNG được cv2.resize thẳng về (224,224) vì crop không vuông sẽ bị
        bóp méo → mặt quay nghiêng trông "ít nghiêng hơn" → yaw bị bão hòa
        (vd: quay 60° chỉ ra ~40°).
        """
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Resize: cạnh ngắn → 224, giữ tỷ lệ
        scale = 224.0 / max(1, min(h, w))
        new_w = max(224, int(round(w * scale)))
        new_h = max(224, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h))

        # CenterCrop 224×224
        x0 = (new_w - 224) // 2
        y0 = (new_h - 224) // 2
        cropped = resized[y0:y0 + 224, x0:x0 + 224]

        normalized = (cropped.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        return normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1,3,224,224)

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.bbox is None:
            return ctx

        self._counter += 1
        if self._counter < self._interval:
            if self._last_head_pose is not None:
                return dataclasses.replace(
                    ctx,
                    head_pose=self._last_head_pose,
                    head_rotation_matrix=self._last_R,
                )
            return ctx

        self._counter = 0
        x1, y1, x2, y2 = ctx.bbox
        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, factor=0.3)

        h, w = ctx.frame.shape[:2]
        ex1 = max(0, ex1)
        ey1 = max(0, ey1)
        ex2 = min(ex2, w)
        ey2 = min(ey2, h)

        head_crop = ctx.frame[ey1:ey2, ex1:ex2]
        if head_crop.size == 0:
            return ctx

        tensor = self._preprocess(head_crop)
        outs = self._session.run(None, {self._input_name: tensor})
        R = np.asarray(outs[0], dtype=np.float64).reshape(3, 3)

        # Euler angles — SixDRepNet dùng R = Rz · Ry · Rx (standard ZYX).
        # Công thức gốc compute_euler_angles_from_rotation_matrices():
        #   pitch(x) = atan2(R[2,1], R[2,2])
        #   yaw(y)   = atan2(-R[2,0], sy)     sy = sqrt(R[0,0]^2 + R[1,0]^2)
        #   roll(z)  = atan2(R[1,0], R[0,0])
        # Model convention: yaw+ = quay TRÁI (subject's left).
        # App convention:   yaw+ = phải → negate yaw cho video (không flip).
        # Webcam (frame_flipped): ảnh đã mirror → model yaw+ = phải thực tế,
        # nên KHÔNG negate.
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2).clip(min=1e-12)
        pitch_d = -float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw_d   = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll_d  = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

        # Clamp to plausible human head range ±90°
        if abs(pitch_d) > 90 or abs(yaw_d) > 90:
            return dataclasses.replace(
                ctx,
                head_pose=self._last_head_pose,
                head_rotation_matrix=self._last_R,
            )

        head_pose_angles = (yaw_d, pitch_d, roll_d)

        # EMA thích nghi — giảm jitter khi đứng yên, bám kịp khi quay nhanh.
        # alpha tăng theo độ lệch so với frame trước (chủ yếu theo yaw/pitch).
        if self._last_head_pose is not None:
            prev = self._last_head_pose
            delta = max(abs(yaw_d - prev[0]), abs(pitch_d - prev[1]))
            t = min(1.0, delta / self._EMA_FAST_DELTA)
            a = self._EMA_ALPHA_MIN + (self._EMA_ALPHA_MAX - self._EMA_ALPHA_MIN) * t
            head_pose_angles = (
                a * yaw_d   + (1 - a) * prev[0],
                a * pitch_d + (1 - a) * prev[1],
                a * roll_d  + (1 - a) * prev[2],
            )

        # Flip yaw cho app convention (yaw+ = phải):
        # - Video (frame_flipped=False): model yaw+ = left → negate để yaw+ = phải.
        # - Webcam (frame_flipped=True): ảnh đã mirror → model left = real right → OK.
        sy_a, sp_a, sr_a = head_pose_angles
        if not ctx.frame_flipped:
            sy_a = -sy_a
        head_pose_angles = (sy_a, sp_a, sr_a)

        self._last_head_pose = head_pose_angles

        # Reconstruct R from (smoothed) angles — consistent with displayed
        # head pose and avoids using noisy raw R for gaze rotation.
        sy_d, sp_d, sr_d = head_pose_angles
        R_smooth = get_rotation_matrix(
            np.deg2rad(sp_d), np.deg2rad(sy_d), np.deg2rad(sr_d),
        )
        self._last_R = R_smooth

        return dataclasses.replace(
            ctx,
            head_pose=head_pose_angles,
            head_rotation_matrix=R_smooth,
        )
