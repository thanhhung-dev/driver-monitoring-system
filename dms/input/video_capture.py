import time
import os

import cv2
import yaml

from utils.logger import setup_logger


class CameraNotFoundError(Exception):
    """Raised when the camera cannot be opened or is not connected."""
class FPSCounter:
    """
    Measures real-time FPS over a sliding window of recent frames.

    Args:
        window: Number of frames used for the rolling average.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record the timestamp of a successfully captured frame."""
        self._timestamps.append(time.time())
        if len(self._timestamps) > self.window:
            self._timestamps.pop(0)

    def get_fps(self) -> float:
        """Return the current rolling FPS. Returns 0.0 if not enough data."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return 0.0 if elapsed == 0 else (len(self._timestamps) - 1) / elapsed



class VideoCapture:
    """
    Manages a continuous video stream from a camera or IR device.

    Follows the pipeline entry-point:
        Video Input  →  Detection  →  Analysis  →  Action

    Usage::

        cap = VideoCapture()
        cap.open()          # raises CameraNotFoundError if not connected
        ret, frame = cap.read_frame()
        cap.release()
    """

    def __init__(self, config_path: str = "dms/config.yaml") -> None:
        self.logger = setup_logger("VideoCapture", config_path)
        self._config = self._load_config(config_path)
        self._cap: cv2.VideoCapture | None = None
        self._is_video_file = False
        self.fps_counter = FPSCounter()



    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def open(self) -> None:
        """
        Open the camera device and apply resolution / FPS settings.

        Raises:
            CameraNotFoundError: If the device cannot be opened (AC #5).
        """
        cam_cfg = self._config["camera"]
        source = cam_cfg.get("source")
        device_id: int = cam_cfg["device_id"]
        width: int = cam_cfg["resolution"]["width"]
        height: int = cam_cfg["resolution"]["height"]
        target_fps: int = cam_cfg["fps"]

        # Use video file if source is set, otherwise use camera device
        self._is_video_file = bool(source)
        input_source = source if source else device_id
        self.logger.info(f"Opening video source: {input_source} ...")
        self._cap = cv2.VideoCapture(input_source)

        if not self._cap.isOpened():
            raise CameraNotFoundError(
                f"[ERROR-CAM-001] Cannot open source: {input_source}. "
                "Check file path or hardware connection."
            )

        if not source:
            # Apply settings only for camera (best-effort; hardware may override)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        self.logger.info(
            f"Source opened successfully: {input_source}"
        )

    def read_frame(self):
        """
        Capture one frame from the camera.

        Returns:
            (success: bool, frame: np.ndarray | None)

        Raises:
            CameraNotFoundError: If called before open().
        """
        if self._cap is None or not self._cap.isOpened():
            raise CameraNotFoundError(
                "[ERROR-CAM-002] Camera is not open. Call open() first."
            )

        ret, frame = self._cap.read()
        if not ret and self._is_video_file:
            # Loop video from the beginning
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        if ret:
            self.fps_counter.tick()
        return ret, frame

    def get_fps(self) -> float:
        """Return the measured real-time FPS."""
        return self.fps_counter.get_fps()

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self.logger.info("Camera released.")