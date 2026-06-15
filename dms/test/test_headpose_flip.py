"""
Test độc lập: chẩn đoán flip yaw khi head pose > ~85°.

CHẠY:
    cd D:\\Workspace\\driver_monitoring\\dms
    python -m test.test_headpose_flip

MỤC ĐÍCH:
    Chạy model resnet50.onnx trực tiếp lên các ảnh profile, in ra:
      - Ma trận xoay R (3x3) từ model
      - ‖RᵀR − I‖  → độ lệch khỏi ma trận chính giao (≈0 = R tốt, >0.01 = R hỏng)
      - det(R)      → ≈+1 = ma trận xoay hợp lệ; ≈−1 = reflection (luôn flip dấu)
      - sy          → gần 0 nghĩa là sắp gimbal lock (yaw ≈ ±90°)
      - Euler angles [pitch, yaw, roll] (độ)

CÁCH ĐỌC KẾT QUẢ (xem README ở cuối file):
    - ‖RᵀR − I‖ > 0.01     → Nguyên nhân 3: model OOD, R không còn ∈ SO(3)
    - det(R) ≈ −1          → R là reflection → luôn flip dấu
    - sy < 0.1             → Nguyên nhân 1: sắp gimbal lock, arctan2 không ổn định
    - R tốt nhưng Euler flip → Nguyên nhân 1 thuần (công thức phân rã)
"""

import os
import sys

import numpy as np
import cv2
import onnx

# Cho phép chạy cả `python test/test_headpose_flip.py` lẫn `python -m test.test_headpose_flip`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.onnx_providers import make_session  # noqa: E402
from core.stages.head_pose_stage import _rotation_matrix_to_euler  # noqa: E402

# ImageNet normalization (giống hệt HeadPoseStage._preprocess)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_crop: np.ndarray):
    """Resize cạnh ngắn → 224 + CenterCrop 224 (giống HeadPoseStage)."""
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
    """Trả về các chỉ số chẩn đoán R."""
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det_R = float(np.linalg.det(R))
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    euler_deg = np.degrees(_rotation_matrix_to_euler(R))
    return ortho_err, det_R, sy, euler_deg


def _verdict(ortho_err: float, det_R: float, sy: float) -> list[str]:
    """Gợi ý nguyên nhân từ các chỉ số."""
    notes = []
    if det_R < 0:
        notes.append("det(R)<0 → REFLECTION (R lật, luôn flip dấu). Nguyên nhân 3 nặng.")
    if ortho_err > 0.01:
        notes.append(f"‖RᵀR−I‖={ortho_err:.4f} > 0.01 → R hỏng, model OOD. Nguyên nhân 3.")
    if sy < 0.1:
        notes.append(f"sy={sy:.4f} < 0.1 → GẦN GIMBAL LOCK (yaw≈±90°). Nguyên nhân 1.")
    if ortho_err < 0.01 and det_R > 0 and sy >= 0.1:
        notes.append("R hợp lệ + sy đủ lớn → flip (nếu có) do công thức Euler/ngữ cảnh. Nguyên nhân 1/2.")
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
    R = np.asarray(session.run(None, {inp_name: tensor})[0], dtype=np.float64).reshape(3, 3)
    ortho_err, det_R, sy, euler_deg = diagnose(R)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"R (3x3) =")
    for row in R:
        print("  [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    print(f"‖RᵀR − I‖ = {ortho_err:.6f}   (≈0 = R hợp lệ)")
    print(f"det(R)    = {det_R:+.6f}     (≈+1 = ma trận xoay đúng, ≈−1 = reflection)")
    print(f"sy        = {sy:.6f}     (≈0 = GẦN GIMBAL LOCK)")
    pitch_d, yaw_d, roll_d = -float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])
    print(f"Euler [pitch,yaw,roll] = [{pitch_d:+.2f}, {yaw_d:+.2f}, {roll_d:+.2f}] độ")
    for note in _verdict(ortho_err, det_R, sy):
        print(f"  → {note}")

    if save_crop:
        out = path.replace(".jpg", "_crop.png").replace(".png", "_crop.png")
        cv2.imwrite(out, crop)
        print(f"→ đã lưu crop: {out}  (mở xem để kiểm mũi có bị cắt mất không)")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "resnet50.onnx")
    if not os.path.exists(model_path):
        print(f"[ERR] Không thấy model: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = onnx.load(model_path, load_external_data=True)
    session = make_session(model.SerializeToString())
    inp_name = session.get_inputs()[0].name

    candidates = [
        "data/AFW_261068_1_13.jpg",
        "data/AFW_815038_2_6.jpg",
        "data/image.png",
        "data/test.jpg",
        "data/lanmarka_68.jpg",
    ]
    for rel in candidates:
        path = os.path.join(root, rel)
        run_on_image(session, inp_name, path)


if __name__ == "__main__":
    main()
