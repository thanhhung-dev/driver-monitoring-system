# dms/input/preprocessor.py
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Pre-processing pipeline for raw BGR frames.
    Converts to grayscale and enhances contrast via CLAHE.
    Target: ~3ms per frame to maintain 15 FPS.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8),
    ):
        """
        Args:
            clahe_clip_limit: Threshold for contrast limiting.
            clahe_tile_grid_size: Size of grid for histogram equalization.
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        logger.info(
            "Preprocessor initialized | CLAHE clipLimit=%.1f tileGrid=%s",
            clahe_clip_limit,
            clahe_tile_grid_size,
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pre-processing pipeline:
          1. Grayscale conversion
          2. CLAHE contrast enhancement

        Args:
            frame: Raw BGR frame, shape (H, W, 3), dtype uint8.

        Returns:
            Grayscale + contrast-enhanced frame, shape (H, W), dtype uint8.
        """
        t0 = time.perf_counter()

        gray = self._to_grayscale(frame)
        enhanced = self._apply_clahe(gray)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("preprocess() took %.2f ms", elapsed_ms)

        return enhanced

    def resize(
        self,
        frame: np.ndarray,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """
        Resize frame to target dimensions.

        Args:
            frame: Any numpy frame (grayscale or BGR).
            width: Target width in pixels (default 640).
            height: Target height in pixels (default 480).

        Returns:
            Resized frame.
        """
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to a grayscale frame."""
        return self.clahe.apply(gray)