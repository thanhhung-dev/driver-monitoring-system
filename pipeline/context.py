from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameContext:
    """Immutable container for per-frame pipeline data.

    Each stage receives a FrameContext, processes it, and returns a new
    FrameContext with additional fields populated. The frozen=True constraint
    prevents accidental mutation of intermediate results.

    Note: `frame` is a mutable ndarray -- frozen only protects the reference,
    not the array contents. This is intentional: visualizer writes in-place.
    """

    frame: np.ndarray
    frame_number: int = 0
    bbox: tuple[int, int, int, int] | None = None
    face_kpss: np.ndarray | None = None
    face_detections: np.ndarray | None = None
    face_keypoints: np.ndarray | None = None
    face_detection_fresh: bool = False
    is_driver: bool | None = None
    driver_name: str | None = None
    out_of_position: bool = False
    driver_roi: tuple[float, float, float, float] | None = None
    identity_similarity: float | None = None
    face_similarities: np.ndarray | None = None
    driver_face_index: int | None = None
    landmarks: np.ndarray | None = None
    facemap_pose: tuple[float, float, float] | None = None
    head_pose: tuple[float, float, float] | None = None
    head_rotation_matrix: np.ndarray | None = None
    gaze_l: np.ndarray | None = None
    gaze_r: np.ndarray | None = None
    gaze_vec_world: np.ndarray | None = None
    eye_center_l: np.ndarray | None = None
    eye_center_r: np.ndarray | None = None
    attribs: dict[str, float] | None = None
    driver_state: Any | None = None
    frame_flipped: bool = False
    face_lost_extreme_pose: bool = False
    extreme_pose_mode: bool = False
    # Head pose ở full profile (~90°): model + keypoint đều mất tín hiệu.
    # Khi True → tầng attention/gaze nên dùng hướng dự đoán (latch_sign) thay
    # vì giá trị số (vì số ở vùng này là rác).
    full_profile_locked: bool = False
    gaze_render_data: dict | None = None

    # Analysis layer results
    drowsiness_score: float | None = None
    distraction_score: float | None = None
    risk_score: float | None = None
    risk_severity: str | None = None
