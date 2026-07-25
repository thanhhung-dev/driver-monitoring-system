"""
Test chẩn đoán flip yaw khi head pose ở góc lớn (>~75°).

CHẠY:
    cd D:\\Workspace\\driver_monitoring\\dms
    python -m test.test_headpose_flip

GỒM 3 PHẦN:
    A. test_formula_sweep()  — SYNTHETIC, KHÔNG dùng model.
       Quét yaw 0→90° qua đúng hàm decode _rotation_matrix_to_euler để CHỨNG
       MINH công thức Euler KHÔNG tự lật dấu ở 75-80° (chỉ "gập" sau 90°).
       → Nếu phần này pass mà ngoài đời vẫn flip ở 75-80° ⇒ thủ phạm là MODEL.

    B. test_video_flip()     — TEMPORAL, dùng model + crop SCRFD khớp runtime.
       Chạy resnet50.onnx trên CHUỖI frame video, crop theo bbox SCRFD +
       expand_bbox(0.3) y hệt HeadPoseStage, log mỗi frame (frame, raw_yaw,
       det_R, sy) và FLAG khi yaw[t-1] ≥ +75° rồi yaw[t] < 0 (lật sang âm ở
       góc lớn). Đây mới là "xác định log khi ở góc lớn tự lật âm".

    C. run_on_image()        — DIAGNOSTIC per-image (giữ từ bản cũ).
       In R / ‖RᵀR−I‖ / det(R) / sy / Euler cho từng ảnh để soi #1 vs #2.

CÁCH ĐỌC CHỈ SỐ:
    - ‖RᵀR − I‖ > 0.01  → R không còn ∈ SO(3): model OOD (nguyên nhân #1)
    - det(R) ≈ −1        → reflection → luôn flip dấu (nguyên nhân #1 nặng)
    - sy < 0.1           → gần gimbal lock yaw≈±90° (nguyên nhân #2)
    - R tốt + sy ok nhưng yaw lật → ambiguity model (#1), KHÔNG phải công thức
"""

import os
import sys

import numpy as np
import cv2
import onnx

# Cho phép chạy cả `python test/test_headpose_flip.py` lẫn `python -m test...`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.onnx_providers import make_session  # noqa: E402
from utils.helpers import expand_bbox  # noqa: E402
from utils.general import get_rotation_matrix  # noqa: E402
from detection.face_detector import FaceDetector  # noqa: E402
from core.stages.head_pose_stage import _rotation_matrix_to_euler  # noqa: E402

# ImageNet normalization (giống hệt HeadPoseStage._preprocess)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Ngưỡng "góc lớn" để coi là vùng flip nguy hiểm.
LARGE_YAW_DEG = 75.0


def preprocess(bgr_crop: np.ndarray):
    """Resize cạnh ngắn → 224 + CenterCrop 224 (giống HeadPoseStage._preprocess)."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = 224.0 / max(1, min(h, w))
    new_w = max(224, int(round(w * scale)))
    new_h = max(224, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h))
    x0 = (new_w - 224) // 2
    y0 = (new_h - 224) // 2
    crop = resized[y0:y0 + 224, x0:x0 + 224]
    norm = (crop.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = norm.transpose(2, 0, 1)[np.newaxis]
    return tensor, crop


def diagnose(R: np.ndarray):
    """Trả về (ortho_err, det_R, sy, euler_deg[pitch,yaw,roll])."""
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det_R = float(np.linalg.det(R))
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    euler_deg = np.degrees(_rotation_matrix_to_euler(R))
    return ortho_err, det_R, sy, euler_deg


def _model_yaw(R: np.ndarray) -> float:
    """yaw (độ) theo đúng convention HeadPoseStage: yaw_d = euler[1]."""
    euler_deg = np.degrees(_rotation_matrix_to_euler(R))
    return float(euler_deg[1])


# ──────────────────────────────────────────────────────────────────────────
# PHẦN A — SYNTHETIC: cô lập công thức Euler
# ──────────────────────────────────────────────────────────────────────────
def test_formula_sweep() -> bool:
    """Quét yaw 0→90° (R thuần xoay) qua hàm decode → công thức KHÔNG được lật dấu.

    Return True nếu công thức "vô tội" (không lật dấu trong [0,90]); False nếu
    chính công thức tự lật (khi đó nguyên nhân #2 nặng hơn dự kiến).
    """
    print("\n" + "=" * 70)
    print("PHẦN A — SYNTHETIC: công thức Euler có tự lật dấu ở 75-80° không?")
    print("=" * 70)
    print(f"{'yaw_in':>7} {'yaw_out':>8} {'pitch':>7} {'roll':>7} {'sy':>7} {'det':>6}")

    ok = True
    prev_out = None
    for y in range(0, 91, 5):
        R = get_rotation_matrix(0.0, np.deg2rad(y), 0.0)
        ortho, det_R, sy, e = diagnose(R)
        yaw_out, pitch_out, roll_out = float(e[1]), float(e[0]), float(e[2])
        print(f"{y:7d} {yaw_out:+8.2f} {pitch_out:+7.1f} {roll_out:+7.1f} "
              f"{sy:7.4f} {det_R:+6.2f}")
        # Trong [0,90], yaw_out phải không âm và đơn điệu tăng.
        if yaw_out < -1e-3:
            print(f"  [FAIL] yaw_in={y} → yaw_out={yaw_out:+.2f} ÂM trong [0,90]!")
            ok = False
        if prev_out is not None and yaw_out + 1e-3 < prev_out:
            print(f"  [FAIL] yaw giảm bất thường: {prev_out:+.2f} → {yaw_out:+.2f}")
            ok = False
        prev_out = yaw_out

    if ok:
        print("\n[PASS] Công thức KHÔNG lật dấu trong [0,90°] (chỉ gập sau 90°).")
        print("       ⇒ Flip ở 75-80° ngoài đời KHÔNG do công thức → nghi MODEL (#1).")
    else:
        print("\n[FAIL] Chính công thức decode lật dấu → nguyên nhân #2 nặng.")
    return ok


# ──────────────────────────────────────────────────────────────────────────
# PHẦN B — TEMPORAL: bắt flip thật trên video, crop khớp runtime
# ──────────────────────────────────────────────────────────────────────────
def _crop_like_runtime(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Crop head y hệt HeadPoseStage: expand_bbox(0.3) rồi clamp vào frame."""
    x1, y1, x2, y2 = bbox
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, factor=0.3)
    h, w = frame.shape[:2]
    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(ex2, w)
    ey2 = min(ey2, h)
    crop = frame[ey1:ey2, ex1:ex2]
    return crop if crop.size > 0 else None


def test_video_flip(
    session,
    inp_name: str,
    detector: FaceDetector,
    video_path: str,
    max_frames: int = 600,
) -> bool:
    """Chạy model trên chuỗi frame → log yaw mỗi frame, flag flip ở góc lớn.

    FLIP định nghĩa: yaw[t-1] ≥ +LARGE_YAW hoặc ≤ −LARGE_YAW, rồi yaw[t] ĐỔI DẤU
    (yaw[t-1]*yaw[t] < 0) trong khi |yaw[t]| cũng đáng kể (>40°). Đây là kiểu
    "+78° tự nhảy −65°" mô tả trong bug.

    Return True nếu KHÔNG có flip; False nếu phát hiện ≥1 flip.
    """
    print("\n" + "=" * 70)
    print("PHẦN B — TEMPORAL: dò flip dấu yaw ở góc lớn trên video")
    print("=" * 70)
    if not os.path.exists(video_path):
        print(f"[skip] Không thấy video: {video_path}")
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[skip] Không mở được video: {video_path}")
        return True

    print(f"Video: {os.path.basename(video_path)}  (tối đa {max_frames} frame)")
    print(f"{'frame':>6} {'raw_yaw':>8} {'det_R':>7} {'sy':>7} {'note':>10}")

    flips = []
    prev_yaw = None
    prev_frame = None
    fno = 0
    processed = 0
    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fno += 1
        det, _ = detector.detect(frame)
        if det is None or len(det) == 0:
            prev_yaw = None  # mất mặt → reset chuỗi
            continue
        x1, y1, x2, y2 = det[0][:4].astype(int)
        crop = _crop_like_runtime(frame, (x1, y1, x2, y2))
        if crop is None:
            prev_yaw = None
            continue

        tensor, _ = preprocess(crop)
        R = np.asarray(
            session.run(None, {inp_name: tensor})[0], dtype=np.float64
        ).reshape(3, 3)
        _, det_R, sy, _ = diagnose(R)
        yaw = _model_yaw(R)
        processed += 1

        note = ""
        is_flip = (
            prev_yaw is not None
            and max(abs(prev_yaw), abs(yaw)) > LARGE_YAW_DEG  # ít nhất 1 đầu góc lớn
            and prev_yaw * yaw < 0                            # đổi dấu
            and abs(prev_yaw) > 40 and abs(yaw) > 40          # cả 2 đáng kể
        )
        if is_flip:
            note = "*** FLIP"
            flips.append((prev_frame, fno, prev_yaw, yaw, det_R, sy))

        # Chỉ in các frame góc lớn / flip để gọn log.
        if abs(yaw) > 60 or is_flip:
            print(f"{fno:6d} {yaw:+8.2f} {det_R:+7.2f} {sy:7.4f} {note:>10}")

        prev_yaw = yaw
        prev_frame = fno

    cap.release()

    print(f"\nĐã xử lý {processed} frame có mặt.")
    if not flips:
        print("[PASS] Không phát hiện flip dấu ở góc lớn trong video này.")
        return True

    print(f"[FAIL] Phát hiện {len(flips)} event FLIP dấu ở góc lớn:")
    for pf, cf, py, cy, dR, sy in flips:
        cause = []
        if dR < 0:
            cause.append("det<0 reflection (#1)")
        if sy < 0.1:
            cause.append("sy nhỏ gimbal (#2)")
        if dR > 0 and sy >= 0.1:
            cause.append("R hợp lệ → model ambiguity (#1)")
        print(f"  frame {pf}→{cf}: yaw {py:+.1f} → {cy:+.1f} | "
              f"det={dR:+.2f} sy={sy:.3f} | {', '.join(cause)}")
    return False


# ──────────────────────────────────────────────────────────────────────────
# PHẦN C — DIAGNOSTIC per-image (giữ từ bản cũ)
# ──────────────────────────────────────────────────────────────────────────
def _verdict(ortho_err: float, det_R: float, sy: float) -> list[str]:
    notes = []
    if det_R < 0:
        notes.append("det(R)<0 → REFLECTION (R lật, luôn flip dấu). #1 nặng.")
    if ortho_err > 0.01:
        notes.append(f"‖RᵀR−I‖={ortho_err:.4f} > 0.01 → R hỏng, model OOD. #1.")
    if sy < 0.1:
        notes.append(f"sy={sy:.4f} < 0.1 → GẦN GIMBAL LOCK (yaw≈±90°). #2.")
    if ortho_err < 0.01 and det_R > 0 and sy >= 0.1:
        notes.append("R hợp lệ + sy đủ lớn → flip (nếu có) do model ambiguity. #1.")
    return notes


def run_on_image(session, inp_name: str, path: str, save_crop: bool = True):
    if not os.path.exists(path):
        print(f"[skip] {path} không tồn tại")
        return

    img = cv2.imread(path)
    if img is None:
        print(f"[skip] {path}: không đọc được ảnh")
        return

    tensor, crop = preprocess(img)
    R = np.asarray(
        session.run(None, {inp_name: tensor})[0], dtype=np.float64
    ).reshape(3, 3)
    ortho_err, det_R, sy, euler_deg = diagnose(R)

    print(f"\n=== {os.path.basename(path)} ===")
    print("R (3x3) =")
    for row in R:
        print("  [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    print(f"‖RᵀR − I‖ = {ortho_err:.6f}   (≈0 = R hợp lệ)")
    print(f"det(R)    = {det_R:+.6f}     (≈+1 = xoay đúng, ≈−1 = reflection)")
    print(f"sy        = {sy:.6f}     (≈0 = GẦN GIMBAL LOCK)")
    pitch_d, yaw_d, roll_d = -float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])
    print(f"Euler [pitch,yaw,roll] = [{pitch_d:+.2f}, {yaw_d:+.2f}, {roll_d:+.2f}] độ")
    for note in _verdict(ortho_err, det_R, sy):
        print(f"  → {note}")

    if save_crop:
        out = path.replace(".jpg", "_crop.png").replace(".png", "_crop.png")
        cv2.imwrite(out, crop)
        print(f"→ đã lưu crop: {out}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "resnet50.onnx")
    if not os.path.exists(model_path):
        print(f"[ERR] Không thấy model: {model_path}")
        sys.exit(1)

    # PHẦN A — không cần model.
    formula_ok = test_formula_sweep()

    print(f"\nLoading model: {model_path}")
    model = onnx.load(model_path, load_external_data=True)
    session = make_session(model.SerializeToString())
    inp_name = session.get_inputs()[0].name

    # PHẦN B — temporal trên video (crop khớp runtime).
    detector = FaceDetector(model_path=os.path.join(root, "models", "det_2.5g.onnx"))
    video_path = os.path.join(root, "data", "IMG_9945.MOV")
    video_ok = test_video_flip(session, inp_name, detector, video_path)

    # PHẦN C — diagnostic per-image.
    print("\n" + "=" * 70)
    print("PHẦN C — DIAGNOSTIC per-image")
    print("=" * 70)
    candidates = [
        "data/AFW_261068_1_13.jpg",
        "data/AFW_815038_2_6.jpg",
        "data/image.png",
        "data/test.jpg",
        "data/lanmarka_68.jpg",
    ]
    for rel in candidates:
        run_on_image(session, inp_name, os.path.join(root, rel))

    # Tổng kết.
    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(f"  A. Công thức không lật dấu [0,90°]: {'PASS' if formula_ok else 'FAIL'}")
    print(f"  B. Video không có flip góc lớn:     {'PASS' if video_ok else 'FAIL'}")
    if formula_ok and not video_ok:
        print("  ⇒ Công thức ổn nhưng video vẫn flip ⇒ thủ phạm là MODEL (#1).")


if __name__ == "__main__":
    main()
