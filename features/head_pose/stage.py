import dataclasses
import os

import cv2
import numpy as np

from features.head_pose.feedback import HeadPoseFeedback
from features.head_pose.recorder import RECORDER, HeadPoseRecorder
from pipeline.context import FrameContext
from utils.helpers import expand_bbox
from utils.general import get_rotation_matrix
from utils.logger import setup_logger
from utils.onnx_providers import make_session

import onnx

# === DEBUG ============================================================
# Bật/tắt debug chi tiết khi yaw lọt vùng nguy hiểm (>75°).
# Cách dùng: set env var DMS_HEADPOSE_DEBUG=1 trước khi chạy main.py.
#   cmd:     set DMS_HEADPOSE_DEBUG=1 && python main.py
#   powershell: $env:DMS_HEADPOSE_DEBUG=1; python main.py
# Tắt lại: xóa biến hoặc set =0.
# Dùng .strip() vì cmd thường thêm trailing space vào env var.
_DEBUG = os.environ.get("DMS_HEADPOSE_DEBUG", "0").strip() == "1"
_debug_log = setup_logger("head_pose_debug")
# =====================================================================


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
    # (85°) của DetectStage → kích hoạt extreme_pose_mode ở frame kế tiếp thay
    # vì kẹt dưới ngưỡng. Giữ <90° để không bị clamp loại bỏ.
    _YAW_SATURATION_DEG = 88.0

    # ── Tín hiệu DẤU yaw từ SCRFD keypoints (độc lập model head pose) ──────
    # Model head pose nhập nhằng ở profile: vừa đảo DẤU vừa kẹt độ lớn ~70°.
    # SCRFD cho 5 điểm [mắt-trái-ảnh, mắt-phải-ảnh, mũi, ...] → vị trí mũi so
    # với 2 mắt cho biết hướng quay đáng tin. r = 2*nose_pos - 1 ∈ [-1,1]:
    #   r>0 mũi lệch sang phải-ảnh → yaw>0 (theo app convention sau khi flip)
    _KP_SIGN_R = 0.15            # |r| dưới mức này coi là gần chính diện → bỏ qua
    _KP_MIN_EYE_SPAN_PX = 8.0       # eye-span quá nhỏ → keypoint không tin được
    _KP_PROFILE_MODEL_ABS = 70.0    # model báo |yaw| lớn mà ngược dấu keypoint = artifact
                                    # (nâng lên 70 để chỉ cắt gaze ở góc thật ~85°, tắt muộn hơn)

    # ── Chống gimbal lock cho PITCH/ROLL ở vùng profile ───────────────────
    # Ở yaw≈±90° (sy→0) công thức Euler khuếch đại nhiễu model thành dao động
    # pitch/roll ±20° dù đầu đứng yên (xem test/test_headpose_flip.py phần A/B).
    # Giải pháp: khi vào profile, ĐÓNG BĂNG pitch/roll về giá trị ổn định cuối
    # (đo lúc yaw nhỏ, công thức còn tin được) thay vì hiển thị rác.
    _PROFILE_YAW_DEG = 80.0     # |yaw| ≥ mức này → vùng gimbal, pitch/roll không tin
    _STABLE_YAW_DEG = 70.0      # |yaw| < mức này → công thức tin được → cập nhật cache
    # Giữ yaw bão hòa thêm N frame khi MẤT keypoint cue ở profile, tránh yaw
    # sụp đột ngột 88°→16° (vật lý bất khả thi trong 1 frame) gây nhấp nháy.
    _PROFILE_HOLD_FRAMES = 8

    # ── KHÓA (latch) profile ở ~90° ───────────────────────────────────────
    # Ở full profile (~90°), CẢ keypoint LẪN model đều mất tín hiệu tin cậy:
    # nửa mặt biến mất → eye-span sụp, model gần gimbal lock → yaw/pitch/roll
    # nhảy loạn 29°↔73°. Giải pháp: khi đã vào bão hòa (±88°), KHÓA yaw ở
    # giá trị bão hòa + đóng băng pitch/roll, BỎ QUA độ lớn rác của model, cho
    # tới khi keypoint xác nhận mặt đã quay về vùng đo được (hết loạn).
    # Điều kiện THOÁT khóa (mặt quay về chính diện đủ rõ):
    #   |r| nhỏ (mũi gần giữa 2 mắt) VÀ eye-span đủ lớn (2 mắt lại hiện rõ).
    _KP_RECOVER_R = 0.55            # |r| < mức này → mũi gần giữa → hết profile
    _KP_RECOVER_EYE_RATIO = 0.16    # eye-span ≥ tỷ lệ này × bbox_h → 2 mắt rõ lại
    # Số frame LIÊN TIẾP keypoint phải xác nhận "đã quay về" trước khi nhả
    # khóa. Tránh 1 frame keypoint noise gây nhả sai → yaw rác lọt.
    _RECOVERY_SUSTAINED_FRAMES = 3
    # Yaw phải quay về dưới mức này (độ) để xác nhận thật sự hết profile.
    # Kết hợp với keypoint recovery → 2 điều kiện độc lập phải cùng True.
    _RECOVERY_YAW_THRESHOLD = 50.0

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
        # Pitch/roll ổn định cuối (đo lúc yaw nhỏ) → dùng để đóng băng ở profile.
        self._last_stable_pitch = None
        self._last_stable_roll = None
        # Đếm ngược số frame còn được giữ yaw bão hòa khi mất keypoint cue.
        self._profile_hold = 0
        # Khóa profile ở ~90°: khi True → ép yaw = _latch_sign*88°, đóng băng
        # pitch/roll cho tới khi keypoint báo mặt đã quay về vùng đo được.
        self._yaw_latched = False
        self._latch_sign = 1.0
        # ── Cải tiến: latch bền vững hơn ──────────────────────────────────
        # Đếm frame liên tục keypoint xác nhận "đã quay về" → chỉ nhả khi
        # recovery kéo dài đủ lâu (tránh nhả sai do 1 frame keypoint noise).
        self._recovery_count = 0
        # Vận tốc yaw lúc vào latch → dự đoán hướng, chỉ nhả khi yaw thật sự
        # quay về phía trung tâm (không nhả do model nhảy rác).
        self._latch_entry_yaw = 0.0
        self._latch_yaw_velocity = 0.0  # độ/frame, dương = đang quay phải
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

    def _keypoint_geom(self, ctx: FrameContext):
        """Hình học mũi/mắt từ SCRFD keypoints (frame hiện tại).

        Returns (r, eye_span, bbox_h) hoặc None nếu không đủ tin (mất mặt /
        eye-span quá nhỏ). r = 2*nose_pos-1 ∈ [-1.5,1.5]: vị trí mũi so với
        2 mắt (r>0: mũi lệch phải-ảnh). Dùng chung cho cả suy DẤU yaw lẫn
        phát hiện THOÁT khóa profile.

        QUAN TRỌNG: KHÔNG dùng khi face_lost_extreme_pose (keypoint cũ →
        kẹt saturation vĩnh viễn).
        """
        if ctx.face_kpss is None or ctx.bbox is None or ctx.face_lost_extreme_pose:
            return None

        kps = np.asarray(ctx.face_kpss, dtype=np.float32)
        if kps.shape[0] < 3 or not np.isfinite(kps[:3]).all():
            return None

        eye_l, eye_r, nose = kps[0], kps[1], kps[2]
        # Phòng thủ thứ tự: đảm bảo eye_l nằm bên trái ảnh.
        if eye_l[0] > eye_r[0]:
            eye_l, eye_r = eye_r, eye_l

        eye_vec = eye_r - eye_l
        eye_span = float(np.linalg.norm(eye_vec))

        x1, y1, x2, y2 = ctx.bbox
        bbox_h = max(1.0, float(y2 - y1))
        if eye_span < max(self._KP_MIN_EYE_SPAN_PX, 0.04 * bbox_h):
            return None

        # nose_pos: 0 ở mắt-trái, 1 ở mắt-phải (chiếu lên trục 2 mắt → chịu roll).
        u = eye_vec / eye_span
        nose_pos = float(np.dot(nose - eye_l, u) / eye_span)
        r = float(np.clip(2.0 * nose_pos - 1.0, -1.5, 1.5))
        return r, eye_span, bbox_h

    def _keypoint_yaw_cue(self, ctx: FrameContext):
        """Suy DẤU yaw từ SCRFD keypoints — đáng tin ở góc lớn.

        Returns kp_sign (±1.0) hoặc None nếu không đủ tin (gần chính diện /
        mất mặt / eye-span quá nhỏ).
        """
        geom = self._keypoint_geom(ctx)
        if geom is None:
            return None
        r = geom[0]
        if abs(r) < self._KP_SIGN_R:
            return None  # gần chính diện → dấu không quan trọng/không tin
        return 1.0 if r > 0 else -1.0

    def _keypoint_recovered(self, geom) -> bool:
        """True khi keypoint báo mặt đã quay về vùng đo được (thoát profile).

        Mũi gần giữa 2 mắt (|r| nhỏ) VÀ eye-span đủ lớn (2 mắt lại hiện rõ).
        Dùng để NHẢ khóa profile mà KHÔNG phụ thuộc model (vốn nhả rác ~90°).
        """
        if geom is None:
            return False
        r, eye_span, bbox_h = geom
        return abs(r) < self._KP_RECOVER_R and eye_span >= self._KP_RECOVER_EYE_RATIO * bbox_h

    def process(self, ctx: FrameContext) -> FrameContext:
        # Không có bbox nào để crop → bỏ qua.
        # LƯU Ý: KHÔNG bỏ qua khi face_lost_extreme_pose=True. Lúc mất mặt
        # nghiêng, DetectStage giữ bbox gần nhất; head pose VẪN chạy trên đó để
        # liên tục đo yaw → DetectStage mới thấy yaw giảm <75° và thoát
        # extreme_pose_mode khi tài xế quay mặt về (nếu không, yaw đóng băng ở
        # giá trị bão hòa và hệ thống kẹt extreme = "đứng hình").
        if ctx.bbox is None:
            RECORDER.log_skip(ctx.frame_number, "no_bbox")
            return ctx

        self._counter += 1
        if self._counter < self._interval:
            RECORDER.log_skip(ctx.frame_number, "skip_interval")
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

        if _DEBUG and (abs(yaw_d) > 75 or abs(pitch_d) > 75):
            ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
            det_R = float(np.linalg.det(R))
            sy_val = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            _debug_log.info(
                f"[DBG] f={ctx.frame_number} | RAW MODEL "
                f"yaw={yaw_d:+.1f} pitch={pitch_d:+.1f} roll={roll_d:+.1f} | "
                f"‖RᵀR−I‖={ortho_err:.4f} det={det_R:+.4f} sy={sy_val:.4f} | "
                f"flip={ctx.frame_flipped} extreme={ctx.extreme_pose_mode} "
                f"lost={ctx.face_lost_extreme_pose}"
            )

        # Clamp to plausible human head range ±90°
        if abs(pitch_d) > 90 or abs(yaw_d) > 90:
            RECORDER.log_reject(
                ctx.frame_number, ctx.frame_flipped, ctx.extreme_pose_mode,
                ctx.face_lost_extreme_pose, R, yaw_d, pitch_d, roll_d,
                None, "clamp_reject", self._last_head_pose,
            )
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

        # Branch tracker — tên nhánh logic đã chạy (lưu vào CSV).
        branch = "raw_ok"

        # ── Sửa DẤU + phát hiện profile bằng SCRFD keypoints ───────────────
        # Keypoint là tín hiệu hình học 2D độc lập, đáng tin hơn model head
        # pose vốn vừa đảo dấu vừa kẹt độ lớn (~70°) ở profile. KHÔNG negate
        # lại theo frame_flipped vì keypoint đã ở hệ toạ độ ảnh đã flip.
        raw_model_yaw = sy_a
        model_abs = abs(raw_model_yaw)
        geom = self._keypoint_geom(ctx)
        kp_sign = None
        if geom is not None and abs(geom[0]) >= self._KP_SIGN_R:
            kp_sign = 1.0 if geom[0] > 0 else -1.0
        forced_profile = False
        prev_yaw_for_log = (
            self._last_head_pose[0] if self._last_head_pose is not None else None
        )

        # ── KHÓA profile ở ~90°: nhả khóa khi keypoint báo mặt đã quay về ───
        # Cải tiến: yêu cầu keypoint recovery KÉO DÀI liên tục (_RECOVERY_SUSTAINED_FRAMES)
        # VÀ model yaw phải quay về dưới ngưỡng → tránh nhả sai do 1 frame noise.
        if self._yaw_latched:
            kp_ok = self._keypoint_recovered(geom)
            # Model yaw (trước khi ép bão hòa) có quay về trung tâm không?
            # Dùng raw_model_yaw (trước frame_flipped) vì đã ở hệ ảnh.
            yaw_returning = abs(raw_model_yaw) < self._RECOVERY_YAW_THRESHOLD
            if kp_ok and yaw_returning:
                self._recovery_count += 1
            else:
                self._recovery_count = 0  # reset nếu 1 trong 2 fail
            if self._recovery_count >= self._RECOVERY_SUSTAINED_FRAMES:
                self._yaw_latched = False
                self._recovery_count = 0

        if self._yaw_latched:
            # Đang khóa profile (~90°): ép yaw bão hòa + đóng băng pitch/roll,
            # BỎ QUA độ lớn rác của model. Cập nhật DẤU nếu keypoint báo đã
            # quay sang phía kia (vd từ trái sang phải).
            if kp_sign is not None:
                self._latch_sign = kp_sign
            sy_a = self._latch_sign * self._YAW_SATURATION_DEG
            if self._last_stable_pitch is not None:
                sp_a = self._last_stable_pitch
            if self._last_stable_roll is not None:
                sr_a = self._last_stable_roll
            forced_profile = True
            branch = "profile_latch"
            head_pose_angles = (sy_a, sp_a, sr_a)
        else:
            if kp_sign is not None:
                model_wrong_sign_large = (
                    model_abs >= self._KP_PROFILE_MODEL_ABS
                    and raw_model_yaw * kp_sign < 0
                )
                if model_wrong_sign_large:
                    # Artifact đảo dấu của model ở profile (model đọc |yaw| lớn
                    # nhưng NGƯỢC chiều keypoint) → bão hòa để kích hoạt extreme.
                    sy_a = kp_sign * self._YAW_SATURATION_DEG
                    forced_profile = True
                    self._profile_hold = self._PROFILE_HOLD_FRAMES
                    branch = "kp_forced_prof"
                elif model_abs >= 10.0:
                    # Chỉ sửa DẤU, giữ ĐỘ LỚN của model.
                    sy_a = kp_sign * model_abs
                    branch = "kp_fix_sign"

            head_pose_angles = (sy_a, sp_a, sr_a)

            # ── Fallback khi KHÔNG có keypoint cue (giữ logic chống nhảy cũ) ─
            if self._last_head_pose is not None and not forced_profile:
                last_yaw = self._last_head_pose[0]
                big_jump = abs(sy_a - last_yaw) > self._MAX_YAW_JUMP
                sign_reversed = sy_a * last_yaw < 0
                if kp_sign is None:
                    # Không có keypoint → dùng heuristic thời gian như trước.
                    is_mirror_flip = (
                        big_jump and sign_reversed
                        and abs(last_yaw) >= self._FLIP_NEAR_PROFILE_DEG
                    )
                    if is_mirror_flip:
                        sy_a = float(np.sign(last_yaw) * self._YAW_SATURATION_DEG)
                        head_pose_angles = (sy_a, sp_a, sr_a)
                        self._profile_hold = self._PROFILE_HOLD_FRAMES
                        forced_profile = True
                        branch = "heur_mirror_flip"
                    elif (
                        big_jump
                        and abs(last_yaw) >= self._PROFILE_YAW_DEG
                        and self._profile_hold > 0
                    ):
                        # Mất keypoint cue ở profile + yaw model nhảy mạnh (vd
                        # 88→16, vật lý bất khả thi trong 1 frame) → GIỮ yaw cũ
                        # thêm vài frame thay vì để nó sụp → hết nhấp nháy.
                        self._profile_hold -= 1
                        sy_a = last_yaw
                        head_pose_angles = (sy_a, sp_a, sr_a)
                        branch = "profile_hold"
                    elif big_jump and not ctx.extreme_pose_mode:
                        RECORDER.log_reject(
                            ctx.frame_number, ctx.frame_flipped, ctx.extreme_pose_mode,
                            ctx.face_lost_extreme_pose, R, yaw_d, pitch_d, roll_d,
                            kp_sign, "heur_reject", self._last_head_pose,
                        )
                        return dataclasses.replace(
                            ctx,
                            head_pose=self._last_head_pose,
                            head_rotation_matrix=self._last_R,
                        )
                elif big_jump and not ctx.extreme_pose_mode:
                    # Đã có dấu từ keypoint nhưng bước nhảy vẫn vô lý (ngoài
                    # profile) → coi là outlier, giữ giá trị cũ.
                    RECORDER.log_reject(
                        ctx.frame_number, ctx.frame_flipped, ctx.extreme_pose_mode,
                        ctx.face_lost_extreme_pose, R, yaw_d, pitch_d, roll_d,
                        kp_sign, "heur_reject", self._last_head_pose,
                    )
                    return dataclasses.replace(
                        ctx,
                        head_pose=self._last_head_pose,
                        head_rotation_matrix=self._last_R,
                    )

            # Vừa vào bão hòa profile → BẬT khóa cho các frame sau + đóng băng
            # pitch/roll ngay frame này (model đã ở vùng rác ~90°).
            if forced_profile:
                self._yaw_latched = True
                self._recovery_count = 0
                if sy_a != 0:
                    self._latch_sign = float(np.sign(sy_a))
                # Ghi vận tốc yaw lúc vào latch → dự đoán hướng quay.
                # prev_yaw_for_log = yaw frame trước (sau khi đã xử lý).
                if prev_yaw_for_log is not None:
                    self._latch_yaw_velocity = sy_a - prev_yaw_for_log
                else:
                    self._latch_yaw_velocity = 0.0
                self._latch_entry_yaw = sy_a
                if self._last_stable_pitch is not None:
                    sp_a = self._last_stable_pitch
                if self._last_stable_roll is not None:
                    sr_a = self._last_stable_roll
                head_pose_angles = (sy_a, sp_a, sr_a)

        # Cập nhật cache pitch/roll ỔN ĐỊNH khi yaw còn nhỏ (công thức tin được)
        # → dùng để đóng băng pitch/roll lúc vào profile.
        if not self._yaw_latched and abs(sy_a) < self._STABLE_YAW_DEG:
            self._last_stable_pitch = sp_a
            self._last_stable_roll = sr_a

        self._last_head_pose = head_pose_angles
        if self._feedback is not None:
            self._feedback.last_head_pose = head_pose_angles

        # Reconstruct R from angles — consistent with displayed head pose.
        sy_d, sp_d, sr_d = head_pose_angles
        R_smooth = get_rotation_matrix(
            np.deg2rad(sp_d), np.deg2rad(sy_d), np.deg2rad(sr_d),
        )
        self._last_R = R_smooth

        RECORDER.log(
            ctx.frame_number, ctx.frame_flipped, ctx.extreme_pose_mode,
            ctx.face_lost_extreme_pose, R, yaw_d, pitch_d, roll_d,
            kp_sign, forced_profile, branch,
            sy_a, sp_a, sr_a, prev_yaw_for_log,
        )

        if _DEBUG and (abs(sy_a) > 75 or abs(sp_a) > 75):
            prev_yaw = (
                self._last_head_pose[0]
                if self._last_head_pose is not None else None
            )
            _debug_log.info(
                f"[DBG] f={ctx.frame_number} | OUT "
                f"yaw={sy_a:+.1f} pitch={sp_a:+.1f} | "
                f"raw={raw_model_yaw:+.1f} kp_sign={kp_sign} "
                f"forced_profile={forced_profile} | prev_yaw={prev_yaw}"
            )

        return dataclasses.replace(
            ctx,
            head_pose=head_pose_angles,
            head_rotation_matrix=R_smooth,
            full_profile_locked=self._yaw_latched,
        )
