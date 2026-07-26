"""
Telemetry helpers — Qualcomm-style debug overlay metrics.

Cung cấp:
  - compute_head_loc_mm(landmarks, image_shape) → (X, Y, Z) mm via solvePnP
  - compute_eye_loc_mm(eye_center_px, image_shape, z_mm) → (X, Y, Z) mm back-project
  - eye_openness_percent(landmarks, indices) → 0–100 % từ EAR
  - BlinkRateTracker — đếm blink/sec qua sliding window
  - classify_gaze_zone(pitch_deg, yaw_deg) → tên zone

Mục tiêu: tách logic toán học khỏi gaze_stage để overlay debug gọn gàng.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from utils.helpers import (
    _FACE_MODEL_3D,
    estimate_head_pose_pnp,
    landmarks68_to_pnp_points,
)


# ── 3D positions ────────────────────────────────────────────────────────

def compute_head_loc_mm(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float, float] | None:
    """
    Trả về (X, Y, Z) head trong frame camera (đơn vị mm).
    Cần ≥ 68 landmarks (lấy 6 điểm chuẩn cho solvePnP).
    """
    if landmarks is None or len(landmarks) < 68:
        return None
    try:
        pnp_pts = landmarks68_to_pnp_points(landmarks)
        rvec, tvec, _, _, _ = estimate_head_pose_pnp(pnp_pts, image_shape)
        if tvec is None:
            return None
        return float(tvec[0]), float(tvec[1]), float(tvec[2])
    except Exception:
        return None


def compute_eye_loc_mm(
    eye_center_px: np.ndarray,
    image_shape: tuple[int, int],
    z_mm: float,
) -> tuple[float, float, float] | None:
    """
    Back-project pixel coords → 3D mm dùng pinhole model giả định Z = head_z.
    focal ≈ image width (cùng convention với estimate_head_pose_pnp).
    """
    if eye_center_px is None or z_mm <= 0:
        return None
    h, w = image_shape[:2]
    f = float(w)
    cx, cy = w * 0.5, h * 0.5
    px, py = float(eye_center_px[0]), float(eye_center_px[1])
    x = (px - cx) * z_mm / f
    y = (py - cy) * z_mm / f
    return x, y, z_mm


# ── Eye openness ────────────────────────────────────────────────────────

# EAR chuẩn: mở ~0.30, nhắm ~0.10. Map tuyến tính sang 0-100%.
_EAR_OPEN = 0.30
_EAR_CLOSED = 0.10


def eye_openness_percent(
    landmarks: np.ndarray,
    indices: list[int],
) -> float | None:
    """EAR → % openness (0=nhắm, 100=mở to)."""
    if len(indices) < 6 or landmarks is None:
        return None
    p = landmarks[indices, :2].astype(np.float32)
    v1 = float(np.linalg.norm(p[1] - p[5]))
    v2 = float(np.linalg.norm(p[2] - p[4]))
    h  = float(np.linalg.norm(p[0] - p[3]))
    if h < 1e-3:
        return None
    ear = (v1 + v2) / (2.0 * h)
    pct = (ear - _EAR_CLOSED) / (_EAR_OPEN - _EAR_CLOSED) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


# ── Blink rate tracker ──────────────────────────────────────────────────

class BlinkRateTracker:
    """
    Đếm blink event (open→close→open) và tính blink/sec qua sliding window.
    Threshold % để xác định mắt nhắm: < 30%.
    """

    def __init__(self, window_sec: float = 30.0, closed_threshold: float = 30.0):
        self.window_sec = window_sec
        self.closed_threshold = closed_threshold
        self._closed = False
        self._events: deque[float] = deque()

    def update(self, openness_pct: float | None) -> float:
        """Update với % openness của 1 mắt (avg L/R). Trả về blink/sec."""
        now = time.time()
        if openness_pct is not None:
            is_closed = openness_pct < self.closed_threshold
            # Cạnh xuống open→closed = bắt đầu 1 blink (đếm khi mở lại)
            if self._closed and not is_closed:
                self._events.append(now)
            self._closed = is_closed

        # Loại bỏ event cũ ngoài cửa sổ
        cutoff = now - self.window_sec
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

        return len(self._events) / self.window_sec


# ── Gaze zone classification ────────────────────────────────────────────

def classify_gaze_zone(pitch_deg: float, yaw_deg: float) -> str:
    """
    Phân loại gaze direction thành cabin zone (Qualcomm-style).

    Camera mount giả định: chính diện tài xế.
    Yaw: + = nhìn trái (driver side), - = nhìn phải (passenger side).
    Pitch: + = nhìn lên, - = nhìn xuống.
    """
    if abs(yaw_deg) > 30:
        return "DRIVER_SIDE_WINDOW" if yaw_deg > 0 else "PASSENGER_SIDE_WINDOW"
    if pitch_deg > 20:
        return "REAR_VIEW_MIRROR"
    if pitch_deg < -20:
        return "INSTRUMENT_CLUSTER"
    if abs(yaw_deg) > 15:
        return "SIDE_MIRROR_LEFT" if yaw_deg > 0 else "SIDE_MIRROR_RIGHT"
    return "WINDSHIELD_CENTER"
