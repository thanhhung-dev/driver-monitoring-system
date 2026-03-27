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

    EYE_BROW_LEFT = [70, 63, 105, 66, 107]
    EYE_BROW_RIGHT = [336, 296, 334, 293, 300]

    # Upper / lower eyelid splits (proper MediaPipe indices)
    LEFT_EYE_UPPER = [133, 158, 160, 33]
    LEFT_EYE_LOWER = [133, 153, 144, 33]
    RIGHT_EYE_UPPER = [362, 385, 387, 263]
    RIGHT_EYE_LOWER = [362, 380, 373, 263]
    NOSE_TIP = [4, 5, 6, 8]
    NOSE_VECTANGLE = [98,2,327]

    # Lips contours (MediaPipe)
    LIPS_UPPER_OUTER = [61, 40, 37, 267, 270, 291]
    LIPS_LOWER_OUTER = [61, 91, 84, 314, 321, 291]
    LIPS_UPPER_INNER = [78, 82, 13, 312, 308]
    LIPS_LOWER_INNER = [78, 87, 14, 317, 308]

    # ── Iris landmark indices (MediaPipe Iris) ────────────────────────────────
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473


    CHIN = 152

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
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.4,
            running_mode=RunningMode.IMAGE,
        )
        self.detector = FaceLandmarker.create_from_options(options)
        self._prev_landmarks: Optional[np.ndarray] = None
        self._smooth_alpha = 0.4  # lower = smoother (less jitter)
        self._miss_count = 0
        self._max_miss = 3  # keep previous landmarks for up to 3 missed frames

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
            self._miss_count += 1
            if self._prev_landmarks is not None and self._miss_count <= self._max_miss:
                return [(int(x), int(y)) for x, y in self._prev_landmarks]
            self._prev_landmarks = None
            return None

        self._miss_count = 0
        h, w = image.shape[:2]
        raw = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])

        if self._prev_landmarks is not None and self._prev_landmarks.shape == raw.shape:
            smoothed = self._smooth_alpha * raw + (1 - self._smooth_alpha) * self._prev_landmarks
        else:
            smoothed = raw

        self._prev_landmarks = smoothed
        return [(int(x), int(y)) for x, y in smoothed]

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
        CYAN = (255, 255, 0)

        # Vẽ mắt trái
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=norm_landmarks,
            connections=self._CONN_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=CYAN, thickness=1
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
                color=CYAN, thickness=1
            ),
            is_drawing_landmarks=False,
        )
        for idx in self.LEFT_EYE:
            cv2.circle(image, landmarks[idx], 3, (255,255,0), -1)
        for idx in self.RIGHT_EYE:
            cv2.circle(image, landmarks[idx], 3, (255, 255, 0), -1)


        return image

    def draw_eye_mesh(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw eye landmarks — upper (green) / lower (red) with lines."""
        CYAN = (255, 255, 0)

        for upper, lower in [
            (self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER),
            (self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER),
        ]:
            # Upper eyelid: circles + line
            upper_pts = np.array([landmarks[i] for i in upper], dtype=np.int32)
            cv2.polylines(image, [upper_pts], isClosed=False, color=CYAN, thickness=1)
            for idx in upper:
                cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)

            # Lower eyelid: circles + line
            lower_pts = np.array([landmarks[i] for i in lower], dtype=np.int32)
            cv2.polylines(image, [lower_pts], isClosed=False, color=CYAN, thickness=1)
            for idx in lower:
                cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)

        return image
    

    @staticmethod
    def _ear(upper_pts: List[Tuple[int, int]], lower_pts: List[Tuple[int, int]]) -> float:
        """Compute Eye Aspect Ratio from upper/lower eyelid points."""
        vertical = 0.0
        count = min(len(upper_pts), len(lower_pts))
        for i in range(count):
            vertical += np.linalg.norm(np.array(upper_pts[i]) - np.array(lower_pts[i]))
        vertical /= max(count, 1)
        horizontal = np.linalg.norm(np.array(upper_pts[0]) - np.array(upper_pts[-1]))
        return vertical / max(horizontal, 1e-6)

    def compute_ear(self, landmarks: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute EAR for left and right eyes. Returns (left_ear, right_ear)."""
        left_upper = [landmarks[i] for i in self.LEFT_EYE_UPPER]
        left_lower = [landmarks[i] for i in self.LEFT_EYE_LOWER]
        right_upper = [landmarks[i] for i in self.RIGHT_EYE_UPPER]
        right_lower = [landmarks[i] for i in self.RIGHT_EYE_LOWER]
        return self._ear(left_upper, left_lower), self._ear(right_upper, right_lower)

    def get_eye_landmarks(
        self, landmarks: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Get 8-point eye landmark lists."""
        return [landmarks[i] for i in self.LEFT_EYE], [landmarks[i] for i in self.RIGHT_EYE]
    def close(self):
        """Release FaceLandmarker resources."""
        self.detector.close()


    def draw_nose(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw nose landmarks."""
        CYAN = (255, 255, 0)
        for idx in self.NOSE_TIP:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)
        nose_pts = np.array([landmarks[i] for i in self.NOSE_TIP], dtype=np.int32)
        cv2.polylines(image, [nose_pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        # Line from point 4 down to 98 and 327 only
        for idx in [98, 327]:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)
            cv2.line(image, landmarks[4], landmarks[idx], CYAN, 1, lineType=cv2.LINE_AA)
        cv2.circle(image, landmarks[2], 2, CYAN, -1, lineType=cv2.LINE_AA)

        # Connect 98 -> 2 -> 327
        bottom_pts = np.array([landmarks[98], landmarks[2], landmarks[327]], dtype=np.int32)
        cv2.polylines(image, [bottom_pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        return image

    def draw_eyebrow_left(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw left eyebrow landmarks with line."""
        CYAN = (255, 255, 0)
        for idx in self.EYE_BROW_LEFT:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)
        pts = np.array([landmarks[i] for i in self.EYE_BROW_LEFT], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        return image

    def draw_eyebrow_right(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw right eyebrow landmarks with line."""
        CYAN = (255, 255, 0)
        for idx in self.EYE_BROW_RIGHT:
            cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)
        pts = np.array([landmarks[i] for i in self.EYE_BROW_RIGHT], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        return image

    def draw_chin(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw chin landmark point."""
        CYAN = (255, 255, 0)
        cv2.circle(image, landmarks[self.CHIN], 3, CYAN, -1, lineType=cv2.LINE_AA)
        return image

    def draw_lips(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw lips — upper outer, lower outer, upper inner, lower inner."""
        CYAN = (255, 255, 0)
        for contour in [self.LIPS_UPPER_OUTER, self.LIPS_LOWER_OUTER,
                        self.LIPS_UPPER_INNER, self.LIPS_LOWER_INNER]:
            for idx in contour:
                cv2.circle(image, landmarks[idx], 2, CYAN, -1, lineType=cv2.LINE_AA)
            pts = np.array([landmarks[i] for i in contour], dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        return image
