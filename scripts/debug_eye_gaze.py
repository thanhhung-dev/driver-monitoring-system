"""
Debug script cho EyeGaze trên 1 ảnh tĩnh.

Mục đích: soi root-cause khi gaze output sai (vd. pitch=-0.165 rad khi mặt
profile, đáng lẽ ~0). Pipeline thật chạy 30 FPS với cv2.imshow nên không vẽ
matplotlib được — script này tách riêng để dùng matplotlib + log chi tiết.

Tận dụng tối đa module có sẵn:
  • EyeGazeEstimation._preprocess + _infer + _crop  (không reload model = ONNX)
  • FaceDetector + FaceMap3DMMDetector              (cùng pipeline thật)
  • Convention pitch/yaw → 3D vector giống Pipeline._pitchyaw_to_vec
    (x = -cos·sin(yaw), y = sin(pitch), z = -cos·cos(yaw))

Chỉ thêm phần matplotlib:
  • plot_gaze_3d        : 3D quiver + info box (pitch/yaw/x/y/z/magnitude)
  • plot_gaze_on_crop   : 2D arrow trên eye-crop 160×96 đã preprocess
  • plot_gaze_on_frame  : arrow vẽ trên full frame tại eye center

Cách chạy (từ thư mục gốc repository):
    python scripts/debug_eye_gaze.py data/image.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (đăng ký projection='3d')

# ── Thêm repository root vào sys.path để import được package nội bộ ───────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.face.detector import FaceDetector  # noqa: E402
from features.gaze.estimator import EyeGazeEstimation, _eye_indices  # noqa: E402
from features.landmarks.detector import FaceMap3DMMDetector  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# PHẦN 4 — Convert pitch/yaw → 3D vector (giống Pipeline._pitchyaw_to_vec)
# ════════════════════════════════════════════════════════════════════════
def pitchyaw_to_vec(pitch: float, yaw: float) -> np.ndarray:
    """Camera convention: +Z hướng vào subject, +Y up, +X right.

    Giống hệt Pipeline._pitchyaw_to_vec để kết quả debug khớp pipeline thật.
    """
    x = -np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════
# PHẦN 2 — Draw 2D gaze arrow (đơn giản, dùng cho eye crop)
# ════════════════════════════════════════════════════════════════════════
def draw_gaze_arrow(
    image: np.ndarray,
    eye_pos: np.ndarray,
    pitchyaw: np.ndarray,
    length: float | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ arrow 2D từ pitch/yaw. Convention khớp pipeline:
       dx = -sin(yaw)  (+yaw = trái → mũi tên đi sang trái ảnh, dx âm)
       dy = +sin(pitch) ở screen-space ảnh đã chuẩn hoá (+pitch = up).
       OpenCV +y down ⇒ đảo dấu khi vẽ.
    """
    out = image.copy()
    if out.ndim == 2 or out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    H, W = out.shape[:2]
    if length is None:
        length = 0.35 * float(min(H, W))

    pitch, yaw = float(pitchyaw[0]), float(pitchyaw[1])
    dx = -np.sin(yaw)
    dy = -np.sin(pitch)  # +pitch = nhìn lên ⇒ y pixel giảm

    norm = np.hypot(dx, dy)
    if norm > 1e-6:
        dx, dy = dx / norm, dy / norm
    dx *= length
    dy *= length

    start = tuple(np.round(eye_pos).astype(np.int32))
    end = (
        int(np.clip(round(eye_pos[0] + dx), 0, W - 1)),
        int(np.clip(round(eye_pos[1] + dy), 0, H - 1)),
    )
    cv2.arrowedLine(out, start, end, color, thickness, cv2.LINE_AA, tipLength=0.2)
    return out


# ════════════════════════════════════════════════════════════════════════
# PHẦN 5 — Matplotlib visualizations
# ════════════════════════════════════════════════════════════════════════
def plot_gaze_3d(pitchyaw: np.ndarray, gaze_vector: np.ndarray, title: str = ""):
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter([0], [0], [0], color="red", s=100, label="Eye Center", zorder=5)
    ax1.quiver(
        0, 0, 0,
        gaze_vector[0], gaze_vector[1], gaze_vector[2],
        color="blue", arrow_length_ratio=0.2, linewidth=3, label="Gaze Direction",
    )
    ax1.quiver(0, 0, 0, 0.5, 0, 0, color="red", arrow_length_ratio=0.2, alpha=0.3, label="X (Right)")
    ax1.quiver(0, 0, 0, 0, 0.5, 0, color="green", arrow_length_ratio=0.2, alpha=0.3, label="Y (Up)")
    ax1.quiver(0, 0, 0, 0, 0, 0.5, color="blue", arrow_length_ratio=0.2, alpha=0.3, label="Z (Forward)")
    ax1.set_xlabel("X (Right)"); ax1.set_ylabel("Y (Up)"); ax1.set_zlabel("Z (Fwd)")
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"3D Gaze Vector {title}", fontsize=12, weight="bold")

    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    pitch_deg = np.degrees(pitchyaw[0])
    yaw_deg = np.degrees(pitchyaw[1])
    info = (
        f"GAZE ANGLES:\n────────────────────\n"
        f"Pitch (θ): {pitch_deg:+.2f}°\n"
        f"Yaw   (φ): {yaw_deg:+.2f}°\n\n"
        f"3D VECTOR (camera frame):\n────────────────────\n"
        f"X: {gaze_vector[0]:+.4f}  (Right/Left)\n"
        f"Y: {gaze_vector[1]:+.4f}  (Up/Down)\n"
        f"Z: {gaze_vector[2]:+.4f}  (Forward)\n\n"
        f"MAGNITUDE: {np.linalg.norm(gaze_vector):.4f}\n"
    )
    ax2.text(
        0.05, 0.5, info, fontsize=11, family="monospace",
        transform=ax2.transAxes, verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )
    plt.tight_layout()


def plot_gaze_on_crop(eye_crop_gray01: np.ndarray, pitchyaw: np.ndarray, title: str = ""):
    """eye_crop_gray01: ảnh 96×160 float32 [0,1] (output của _preprocess)."""
    vis = cv2.cvtColor((eye_crop_gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    iris_center = np.array([vis.shape[1] / 2.0, vis.shape[0] / 2.0], dtype=np.float32)
    out = draw_gaze_arrow(vis, iris_center, pitchyaw)
    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Gaze on eye-crop {title}\n"
        f"Pitch: {np.degrees(pitchyaw[0]):+.2f}° | Yaw: {np.degrees(pitchyaw[1]):+.2f}°",
        fontsize=12, weight="bold",
    )
    plt.axis("off")
    plt.tight_layout()


def plot_gaze_on_frame(
    frame_bgr: np.ndarray,
    eye_center_l: np.ndarray | None,
    eye_center_r: np.ndarray | None,
    gaze_l: np.ndarray | None,
    gaze_r: np.ndarray | None,
):
    """Vẽ arrow ngay tại eye center trên full frame để so sánh với pipeline."""
    out = frame_bgr.copy()
    if eye_center_l is not None and gaze_l is not None:
        out = draw_gaze_arrow(out, eye_center_l, gaze_l, length=80, color=(0, 255, 0))
    if eye_center_r is not None and gaze_r is not None:
        out = draw_gaze_arrow(out, eye_center_r, gaze_r, length=80, color=(0, 200, 255))
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title("Gaze arrows on full frame (L=green, R=orange)", fontsize=12, weight="bold")
    plt.axis("off")
    plt.tight_layout()


# ════════════════════════════════════════════════════════════════════════
# MAIN — Full pipeline: detect face → landmarks → gaze → visualize
# ════════════════════════════════════════════════════════════════════════
def main(image_path: str):
    print("=" * 60)
    print(f"Debug EyeGaze on: {image_path}")
    print("=" * 60)

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    print(f"Frame shape: {frame.shape}")

    # 1) Face detection (tái sử dụng module pipeline)
    print("\n[1/4] Face detection...")
    detector = FaceDetector(model_path="models/det_2.5g.onnx")
    det, _kpss = detector.detect(frame)
    if det is None or len(det) == 0:
        print("✗ Không phát hiện mặt — không thể chạy gaze.")
        return
    x1, y1, x2, y2 = det[0][:4].astype(int)
    bbox = (x1, y1, x2, y2)
    print(f"✓ Bbox: {bbox}")

    # 2) Landmarks 68-point + head pose
    print("\n[2/4] Landmarks + head pose...")
    facemap = FaceMap3DMMDetector()
    out = facemap.detect(frame, bbox)
    if out is None:
        print("✗ Không lấy được landmarks.")
        return
    landmarks_list, (pitch_h, yaw_h, roll_h) = out
    landmarks = np.asarray(landmarks_list, dtype=np.float32)
    print(f"✓ Landmarks: {landmarks.shape}")
    print(
        f"✓ Head pose: pitch={np.degrees(pitch_h):+6.2f}°  "
        f"yaw={np.degrees(yaw_h):+6.2f}°  roll={np.degrees(roll_h):+6.2f}°"
    )

    # 3) Eye gaze (raw inference, không qua EMA của pipeline)
    print("\n[3/4] Eye gaze inference...")
    eye_gaze = EyeGazeEstimation(smooth_alpha=1.0)  # alpha=1 ⇒ raw, không smooth
    gaze_l, gaze_r, center_l, center_r = eye_gaze.detect(frame, landmarks)

    def _fmt(g):
        if g is None:
            return "None"
        return f"pitch={np.degrees(g[0]):+6.2f}°  yaw={np.degrees(g[1]):+6.2f}°"

    print(f"✓ Gaze L : {_fmt(gaze_l)}  @ center {center_l}")
    print(f"✓ Gaze R : {_fmt(gaze_r)}  @ center {center_r}")

    # 4) Visualize
    print("\n[4/4] Visualizations...")
    left_idx, right_idx = _eye_indices(len(landmarks))

    # Lấy lại eye crops đã preprocess (để vẽ matplotlib)
    crop_l_bgr, _ = EyeGazeEstimation._crop(frame, landmarks, left_idx)
    crop_r_bgr, _ = EyeGazeEstimation._crop(frame, landmarks, right_idx)
    if crop_l_bgr is not None and gaze_l is not None:
        proc_l = EyeGazeEstimation._preprocess(crop_l_bgr, flip=False)[0]  # (96,160)
        plot_gaze_on_crop(proc_l, gaze_l, title="(LEFT eye)")
        vec_l = pitchyaw_to_vec(float(gaze_l[0]), float(gaze_l[1]))
        print(f"  L vec: x={vec_l[0]:+.4f}  y={vec_l[1]:+.4f}  z={vec_l[2]:+.4f}")
        plot_gaze_3d(gaze_l, vec_l, title="(LEFT eye)")

    if crop_r_bgr is not None and gaze_r is not None:
        proc_r = EyeGazeEstimation._preprocess(crop_r_bgr, flip=True)[0]
        # gaze_r đã được negate yaw trong eye_gaze.detect; nhưng crop đang ở
        # left-eye space (flipped) ⇒ để arrow trên crop đúng phải dùng giá trị
        # raw model (chưa negate). Đơn giản hơn: hiển thị gaze_r đã unflipped
        # với cùng crop chưa flip → tạo lại proc_r không flip.
        proc_r_noflip = EyeGazeEstimation._preprocess(crop_r_bgr, flip=False)[0]
        plot_gaze_on_crop(proc_r_noflip, gaze_r, title="(RIGHT eye)")
        vec_r = pitchyaw_to_vec(float(gaze_r[0]), float(gaze_r[1]))
        print(f"  R vec: x={vec_r[0]:+.4f}  y={vec_r[1]:+.4f}  z={vec_r[2]:+.4f}")
        plot_gaze_3d(gaze_r, vec_r, title="(RIGHT eye)")

    plot_gaze_on_frame(frame, center_l, center_r, gaze_l, gaze_r)

    print("\n" + "=" * 60)
    print("✓ Done. Đóng cửa sổ matplotlib để thoát.")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug EyeGaze trên 1 ảnh.")
    parser.add_argument(
        "image", nargs="?", default="data/image.png",
        help="Đường dẫn ảnh (mặc định: data/image.png — chạy từ repository root).",
    )
    args = parser.parse_args()

    # Auto-chdir về repository root để model path tương đối luôn chính xác.
    os.chdir(_PROJECT_ROOT)
    main(args.image)
