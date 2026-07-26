"""
Head-pose real-time recorder — capture toàn bộ dynamic state ra CSV mỗi frame.

MỤC ĐÍCH:
    Bug flip yaw > 87° chỉ xuất hiện ở real-time (phụ thuộc chuỗi frame +
    state machine extreme_pose_mode). Test tĩnh không thể tái hiện.
    Module này ghi lại mọi tín hiệu liên quan tại thời điểm process() chạy,
    để sau đó reproduce offline và phân tích nhánh logic nào đang thắng.

CÁCH DÙNG:
    set DMS_HEADPOSE_DEBUG=1   (bật cả log console + recorder)
    python main.py
    → làm lại thao tác gây flip
    → tắt (Ctrl+C), mở logs/headpose_trace.csv

MỘT DÒNG CSV = MỘT FRAME có đủ điều kiện (frame nào skip do interval hoặc
không có bbox sẽ KHÔNG xuất hiện — đó cũng là thông tin quan trọng: để xem
frame nào bị skip, so sánh frame_number trong CSV với frame thực).

CÁC CỘT:
    frame, t_ms            : thời gian & số thứ tự frame
    flipped, extreme, lost : cờ state machine + camera
    R00..R22               : 9 phần tử ma trận R thô từ model
    ortho_err, det_R, sy   : chỉ số chẩn đoán R (‖RᵀR−I‖, det, sy)
    raw_yaw, raw_pitch, raw_roll : Euler thô TRƯỚC mọi sửa đổi
    kp_sign                : dấu từ SCRFD keypoint (None/±1.0)
    forced_profile         : có ép bão hòa 88° không
    out_yaw, out_pitch, out_roll : góc CUỐI CÙNG hiển thị
    prev_yaw               : yaw frame trước (để tính jump)
    jump                   : |out_yaw - prev_yaw| (độ)
    branch                 : nhánh logic nào đã chạy (xem _BRANCH_*)
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

# === DEBUG switch =====================================================
# Cùng env var với log console trong head_pose_stage.py.
# Dùng .strip() để chống trailing whitespace do shell (cmd thường thêm
# space sau giá trị: `set X=1 ` → "1 ").
_ENABLED = os.environ.get("DMS_HEADPOSE_DEBUG", "0").strip() == "1"
# ======================================================================

_CSV_PATH = os.path.join("logs", "headpose_trace.csv")

# Tên các nhánh logic (lưu vào cột `branch` để filter nhanh trong Excel/df)
_BRANCH_RAW_OK        = "raw_ok"          # không sửa gì, nhận thô
_BRANCH_KP_FIX_SIGN   = "kp_fix_sign"     # keypoint sửa DẤU, giữ độ lớn
_BRANCH_KP_FORCED     = "kp_forced_prof"  # keypoint ép bão hòa 88°
_BRANCH_HEUR_FLIP     = "heur_mirror_flip"  # heuristic phát hiện flip
_BRANCH_HEUR_REJECT   = "heur_reject"     # heuristic từ chối, giữ cũ
_BRANCH_CLAMP_REJECT  = "clamp_reject"    # |yaw|>90 hoặc |pitch|>90 → bỏ
_BRANCH_SKIP_INTERVAL = "skip_interval"   # skip do interval
_BRANCH_NO_BBOX       = "no_bbox"

_COLUMNS = [
    "frame", "t_ms",
    "flipped", "extreme", "lost",
    "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22",
    "ortho_err", "det_R", "sy",
    "raw_yaw", "raw_pitch", "raw_roll",
    "kp_sign", "forced_profile", "branch",
    "out_yaw", "out_pitch", "out_roll",
    "prev_yaw", "jump",
    # Chất lượng ảnh đầu vào — quan trọng để bắt tương quan sáng↔flip:
    "brightness",      # mean intensity crop (0-255); thấp = tối
    "contrast",        # std intensity crop; thấp = phẳng (model khó rút feature)
    "eye_span_px",     # khoảng cách 2 mắt SCRFD; nhỏ = profile/mất mắt
    "nose_in_bbox",    # 1 nếu mũi trong bbox, 0 nếu tràn ra ngoài
]


class HeadPoseRecorder:
    """CSV recorder cho head-pose stage. Thread-safe đơn giản (GIL)."""

    def __init__(self, path: str = _CSV_PATH):
        self.enabled = _ENABLED
        self._path = Path(path)
        self._fh = None
        self._writer = None
        self._t0 = None
        if not self.enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Mở file ghi đè, ghi header.
        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_COLUMNS)
        self._writer.writeheader()
        self._t0 = time.monotonic()

    def _row(self) -> dict:
        return {c: "" for c in _COLUMNS}

    def log_skip(self, frame_number: int, reason: str) -> None:
        """Ghi 1 dòng cho frame bị skip (no bbox hoặc interval)."""
        if not self.enabled or self._writer is None:
            return
        row = self._row()
        row["frame"] = frame_number
        row["t_ms"] = int((time.monotonic() - self._t0) * 1000)
        row["branch"] = reason
        self._writer.writerow(row)
        self._fh.flush()

    def log_reject(
        self,
        frame_number: int,
        flipped: bool,
        extreme: bool,
        lost: bool,
        R: np.ndarray,
        raw_yaw: float,
        raw_pitch: float,
        raw_roll: float,
        kp_sign,
        branch: str,
        last_head_pose,
        brightness=None,
        contrast=None,
        eye_span_px=None,
        nose_in_bbox=None,
    ) -> None:
        """Ghi 1 dòng cho frame bị reject (clamp hoặc heuristic) — giữ giá trị cũ.

        out_yaw/out_pitch/out_roll = giá trị cũ (đang được tái dùng).
        An toàn khi last_head_pose = None (out_*=0).
        """
        out = last_head_pose if last_head_pose is not None else (0.0, 0.0, 0.0)
        self.log(
            frame_number, flipped, extreme, lost, R,
            raw_yaw, raw_pitch, raw_roll, kp_sign, False, branch,
            out[0], out[1], out[2],
            last_head_pose[0] if last_head_pose is not None else None,
            brightness=brightness,
            contrast=contrast,
            eye_span_px=eye_span_px,
            nose_in_bbox=nose_in_bbox,
        )

    def log(
        self,
        frame_number: int,
        flipped: bool,
        extreme: bool,
        lost: bool,
        R: np.ndarray,
        raw_yaw: float,
        raw_pitch: float,
        raw_roll: float,
        kp_sign,
        forced_profile: bool,
        branch: str,
        out_yaw: float,
        out_pitch: float,
        out_roll: float,
        prev_yaw,
        brightness=None,
        contrast=None,
        eye_span_px=None,
        nose_in_bbox=None,
    ) -> None:
        if not self.enabled or self._writer is None or R is None:
            return
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
        det_R = float(np.linalg.det(R))
        sy_val = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        jump = abs(out_yaw - prev_yaw) if prev_yaw is not None else ""

        row = self._row()
        row["frame"] = frame_number
        row["t_ms"] = int((time.monotonic() - self._t0) * 1000)
        row["flipped"] = int(flipped)
        row["extreme"] = int(extreme)
        row["lost"] = int(lost)
        for i in range(3):
            for j in range(3):
                row[f"R{i}{j}"] = f"{R[i, j]:.5f}"
        row["ortho_err"] = f"{ortho_err:.5f}"
        row["det_R"] = f"{det_R:+.5f}"
        row["sy"] = f"{sy_val:.5f}"
        row["raw_yaw"] = f"{raw_yaw:+.2f}"
        row["raw_pitch"] = f"{raw_pitch:+.2f}"
        row["raw_roll"] = f"{raw_roll:+.2f}"
        row["kp_sign"] = "" if kp_sign is None else f"{kp_sign:+.1f}"
        row["forced_profile"] = int(forced_profile)
        row["branch"] = branch
        row["out_yaw"] = f"{out_yaw:+.2f}"
        row["out_pitch"] = f"{out_pitch:+.2f}"
        row["out_roll"] = f"{out_roll:+.2f}"
        row["prev_yaw"] = "" if prev_yaw is None else f"{prev_yaw:+.2f}"
        row["jump"] = "" if jump == "" else f"{jump:.2f}"
        row["brightness"] = "" if brightness is None else f"{brightness:.1f}"
        row["contrast"] = "" if contrast is None else f"{contrast:.1f}"
        row["eye_span_px"] = "" if eye_span_px is None else f"{eye_span_px:.1f}"
        row["nose_in_bbox"] = "" if nose_in_bbox is None else int(nose_in_bbox)
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# Singleton — toàn bộ pipeline chia sẻ 1 recorder.
RECORDER = HeadPoseRecorder()
