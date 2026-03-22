import os
import mediapipe as _mp  # MUST be first — initializes _mp.tasks.python submodules
import numpy as np
import cv2

from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from typing import Optional, List, Tuple


class FaceMeshDetector:
    """MediaPipe FaceLandmarker for facial landmarks (v0.10+ API)."""

    # ── Key landmark indices ──────────────────────────────────────────────────
    LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246]
    RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]
    LEFT_EYE_CENTER = 159
    RIGHT_EYE_CENTER = 386
    NOSE_TIP = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # ── Iris landmark indices (MediaPipe Iris) ────────────────────────────────
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # ── Connection sets ──────────────────────────────────────────────────────
    _CONN_LEFT_EYE       = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
    _CONN_RIGHT_EYE      = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
    _CONN_LEFT_EYEBROW  = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
    _CONN_RIGHT_EYEBROW = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
    _CONN_LIPS          = FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    _CONN_NOSE          = FaceLandmarksConnections.FACE_LANDMARKS_NOSE
    _CONN_LEFT_IRIS     = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
    _CONN_RIGHT_IRIS    = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe FaceLandmarker.

        Args:
            model_path: Relative path to face_landmarker.task model.
            num_faces: Maximum number of faces to detect.
            min_face_detection_confidence: Minimum face detection confidence.
            min_face_presence_confidence: Minimum face presence confidence.
            min_tracking_confidence: Minimum landmark tracking confidence.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_abs = os.path.join(base_dir, model_path)

        if not os.path.exists(model_abs):
            raise FileNotFoundError(f"Model not found: {model_abs}")

        options = FaceLandmarkerOptions(
            base_options=_mp.tasks.BaseOptions(model_asset_path=model_abs),
            num_faces=num_faces,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=RunningMode.IMAGE,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Detect face mesh landmarks from a BGR image.

        Args:
            image: BGR image (HxWx3 numpy array).

        Returns:
            List of 478 (x, y) pixel landmark coordinates,
            or None if no face detected.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)

        result: FaceLandmarkerResult = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        h, w = image.shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _make_norm_landmarks(
        self, landmarks: List[Tuple[int, int]], img_h: int, img_w: int
    ):
        """Build NormalizedLandmark list from pixel coords with visibility/presence.

        Args:
            landmarks: Pixel (x, y) coordinates on the image.
            img_h: Image height for normalization.
            img_w: Image width for normalization.
        """

        class _NormLm:
            visibility = 1.0
            presence = 1.0
            def __init__(self, x_norm, y_norm):
                self.x = x_norm
                self.y = y_norm

        return [_NormLm(x / img_w, y / img_h) for (x, y) in landmarks]

    def draw_landmarks(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw all 478 face mesh landmark points (green dots)."""
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        return image

    def draw_full_mesh(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Draw eye mesh only: left eye + right eye connections.

        Args:
            image: BGR image.
            landmarks: List of (x, y) pixel landmark coordinates.

        Returns:
            Image with eye mesh drawn.
        """
        img_h, img_w = image.shape[:2]
        norm_landmarks = self._make_norm_landmarks(landmarks, img_h, img_w)
        GREEN = (0, 255, 0)
        CYAN = (0, 255, 255)

        # Vẽ mắt trái
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=norm_landmarks,
            connections=self._CONN_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=GREEN, thickness=1
            ),
            is_drawing_landmarks=False,
        )

        # Vẽ mắt phải
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=norm_landmarks,
            connections=self._CONN_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=GREEN, thickness=1
            ),
            is_drawing_landmarks=False,
        )

        # Vẽ iris trái
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=norm_landmarks,
            connections=self._CONN_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=CYAN, thickness=1
            ),
            is_drawing_landmarks=False,
        )

        # Vẽ iris phải
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=norm_landmarks,
            connections=self._CONN_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=CYAN, thickness=1
            ),
            is_drawing_landmarks=False,
        )

        for idx in self.LEFT_EYE:
            cv2.circle(image, landmarks[idx], 3, GREEN, -1)
        for idx in self.RIGHT_EYE:
            cv2.circle(image, landmarks[idx], 3, (255, 0, 0), -1)

        for idx in self.LEFT_IRIS:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1)
        cv2.circle(image, landmarks[self.LEFT_IRIS_CENTER], 3, CYAN, -1)
        for idx in self.RIGHT_IRIS:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1)
        cv2.circle(image, landmarks[self.RIGHT_IRIS_CENTER], 3, CYAN, -1)

        return image

    def draw_eye_mesh(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw eye landmarks — left (green), right (blue)."""
        for idx in self.LEFT_EYE:
            cv2.circle(image, landmarks[idx], 2, (0, 255, 0), -1)
        for idx in self.RIGHT_EYE:
            cv2.circle(image, landmarks[idx], 2, (255, 0, 0), -1)
        cv2.circle(image, landmarks[self.LEFT_EYE_CENTER], 4, (0, 255, 0), -1)
        cv2.circle(image, landmarks[self.RIGHT_EYE_CENTER], 4, (255, 0, 0), -1)
        return image

    def get_face_roi(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get tight bounding box (x_min, y_min, x_max, y_max)."""
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    def get_eye_center(
        self, landmarks: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get eye centers: ((left_x, left_y), (right_x, right_y))."""
        return landmarks[self.LEFT_EYE_CENTER], landmarks[self.RIGHT_EYE_CENTER]

    def get_eye_landmarks(
        self, landmarks: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Get 8-point eye landmark lists."""
        return [landmarks[i] for i in self.LEFT_EYE], [landmarks[i] for i in self.RIGHT_EYE]

    def get_iris_center(
        self, landmarks: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get iris centers: ((left_x, left_y), (right_x, right_y))."""
        return landmarks[self.LEFT_IRIS_CENTER], landmarks[self.RIGHT_IRIS_CENTER]

    def get_iris_landmarks(
        self, landmarks: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Get 5-point iris landmark lists (center + 4 surrounding)."""
        return [landmarks[i] for i in self.LEFT_IRIS], [landmarks[i] for i in self.RIGHT_IRIS]

    def close(self):
        """Release FaceLandmarker resources."""
        self.detector.close()
