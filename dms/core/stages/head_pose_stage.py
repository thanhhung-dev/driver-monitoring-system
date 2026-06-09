import dataclasses

import cv2
import numpy as np

from core.frame_context import FrameContext
from core.head_pose_feedback import HeadPoseFeedback
from utils.helpers import expand_bbox
from utils.general import get_rotation_matrix
from utils.onnx_providers import make_session

import onnx


def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to Euler angles (pitch, yaw, roll) in radians.
    ZYX convention: R = Rz · Ry · Rx.
    Numpy equivalent of qai_hub_models.utils.image_processing_3d.rotation_matrix_to_euler.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2).clip(min=1e-12)
    singular = sy < 1e-6
    x = np.arctan2(R[2, 1], R[2, 2])
    y = np.arctan2(-R[2, 0], sy)
    z = np.arctan2(R[1, 0], R[0, 0])
    xs = np.arctan2(-R[1, 2], R[1, 1])
    ys = np.arctan2(-R[2, 0], sy)
    zs = 0.0
    return np.array([
        x * (1 - singular) + xs * singular,
        y * (1 - singular) + ys * singular,
        z * (1 - singular) + zs * singular,
    ], dtype=np.float64)


class HeadPoseStage:
    """Head pose estimation using ResNet50 ONNX with interval-based skipping.

    Input  : BGR crop 224×224, float32 NCHW, ImageNet normalized
    Output : 3×3 rotation matrix → Euler angles (yaw, pitch, roll) degrees
    """

    # ImageNet normalization
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Bước nhảy yaw tối đa (độ) giữa 2 lần đo. Khi mặt gần profile (>~77°),
    # model đảo dấu yaw (vd +85° → −65°) = nhảy ~150°, bất khả thi về vật lý
    # → từ chối, giữ giá trị cũ để KHÔNG hiển thị giá trị đảo dấu.
    _MAX_YAW_JUMP = 70.0

    # Chỉ coi là "đảo gương gần profile" khi yaw frame trước đã đủ lớn. Tránh
    # nhầm các bước nhảy lớn ngẫu nhiên lúc mặt còn gần chính diện.
    _FLIP_NEAR_PROFILE_DEG = 55.0

    # Giá trị yaw bão hòa khi phát hiện lật gương. Lớn hơn ENTER_EXTREME_YAW
    # (80°) của DetectStage → kích hoạt extreme_pose_mode ở frame kế tiếp thay
    # vì kẹt dưới ngưỡng. Giữ <90° để không bị clamp loại bỏ.
    _YAW_SATURATION_DEG = 88.0

    def __init__(
        self,
        model_path: str,
        interval: int = 2,
        feedback: HeadPoseFeedback | None = None,
    ) -> None:
        self._interval = interval
        self._counter = 0
        self._last_head_pose = None
        self._last_R = None
        # Cross-frame channel so DetectStage (chạy trước) đọc được head pose
        # của frame trước → kích hoạt extreme_pose_mode.
        self._feedback = feedback
        
        onnx_model = onnx.load(model_path, load_external_data=True)
        model_bytes = onnx_model.SerializeToString()
        self._session = make_session(model_bytes)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "head_pose"

    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """BGR crop → ImageNet-normalized NCHW float32.

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
        if ctx.bbox is None or ctx.face_lost_extreme_pose:
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

        # Euler angles — ZYX convention, same as compute_euler_angles_from_rotation_matrices.
        euler = np.degrees(_rotation_matrix_to_euler(R))
        pitch_d, yaw_d, roll_d = -float(euler[0]), float(euler[1]), float(euler[2])

        # Clamp to plausible human head range ±90°
        if abs(pitch_d) > 90 or abs(yaw_d) > 90:
            return dataclasses.replace(
                ctx,
                head_pose=self._last_head_pose,
                head_rotation_matrix=self._last_R,
            )

        head_pose_angles = (yaw_d, pitch_d, roll_d)

        # Flip yaw only for mirrored webcam frames so app convention stays:
        # yaw+ = subject turns toward image/right side after display mirroring.
        sy_a, sp_a, sr_a = head_pose_angles
        if ctx.frame_flipped:
            sy_a = -sy_a
        head_pose_angles = (sy_a, sp_a, sr_a)

        # Xử lý yaw đảo dấu / nhảy vô lý ở gần profile.
        if self._last_head_pose is not None:
            last_yaw = self._last_head_pose[0]
            big_jump = abs(sy_a - last_yaw) > self._MAX_YAW_JUMP
            # "Lật gương": dấu ngược + frame trước yaw đã lớn + nhảy vô lý.
            # Đây là lỗi đặc trưng của model khi mặt gần profile (>~77°): nó
            # đoán ra rotation matrix của tư thế đối xứng → yaw đổi dấu.
            is_mirror_flip = (
                big_jump
                and sy_a * last_yaw < 0
                and abs(last_yaw) >= self._FLIP_NEAR_PROFILE_DEG
            )
            if is_mirror_flip:
                sy_a = float(np.sign(last_yaw) * self._YAW_SATURATION_DEG)
                sp_a, sr_a = self._last_head_pose[1], self._last_head_pose[2]
                head_pose_angles = (sy_a, sp_a, sr_a)
            elif big_jump and not ctx.extreme_pose_mode:
                return dataclasses.replace(
                    ctx,
                    head_pose=self._last_head_pose,
                    head_rotation_matrix=self._last_R,
                )

        self._last_head_pose = head_pose_angles
        if self._feedback is not None:
            self._feedback.last_head_pose = head_pose_angles

        # Reconstruct R from angles — consistent with displayed head pose.
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
