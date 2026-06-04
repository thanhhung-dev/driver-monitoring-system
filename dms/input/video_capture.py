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

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, config_path: str = "dms/config.yaml") -> None:
        self.logger = setup_logger("VideoCapture", config_path)
        self._config = self._load_config(config_path)
        self._cap: cv2.VideoCapture | None = None
        self._is_video_file = False
        self._is_image_file = False
        self._image_frame = None
        self.fps_counter = FPSCounter()
        # Pacing cho video file: giữ tốc độ phát đúng FPS gốc.
        self._frame_interval: float = 0.0   # seconds per frame
        self._next_frame_t: float = 0.0     # mốc thời gian dự kiến cho frame kế



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

        # Detect source type: image file / video file / camera
        if isinstance(source, str) and source.lower().endswith(self.IMAGE_EXTS):
            self._is_image_file = True
            self._is_video_file = False
            if not os.path.exists(source):
                raise CameraNotFoundError(
                    f"[ERROR-CAM-001] Image file not found: {source}"
                )
            self._image_frame = cv2.imread(source)
            if self._image_frame is None:
                raise CameraNotFoundError(
                    f"[ERROR-CAM-001] Cannot read image: {source}"
                )
            self.logger.info(f"Image source loaded: {source} (shape={self._image_frame.shape})")
            return

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
        else:
            # Video file: đọc FPS gốc để pacing đúng tốc độ thực.
            native_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if native_fps > 1.0:
                self._frame_interval = 1.0 / native_fps
                self.logger.info(f"Video native FPS: {native_fps:.2f} → pacing on")
            else:
                self.logger.info("Video FPS unknown — no pacing")

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
        # Image mode: trả lại cùng 1 frame mỗi lần để debug vẽ overlay
        if self._is_image_file:
            if self._image_frame is None:
                raise CameraNotFoundError(
                    "[ERROR-CAM-002] Image is not loaded. Call open() first."
                )
            self.fps_counter.tick()
            return True, self._image_frame.copy()

        if self._cap is None or not self._cap.isOpened():
            raise CameraNotFoundError(
                "[ERROR-CAM-002] Camera is not open. Call open() first."
            )

        ret, frame = self._cap.read()
        if not ret and self._is_video_file:
            # Loop video from the beginning
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            self._next_frame_t = 0.0   # reset pacing khi tua lại đầu video
        if ret:
            self.fps_counter.tick()
            # ── Pacing cho video file ─────────────────────────────────────
            # Pipeline chạy nhanh hơn FPS gốc → sleep cho khớp realtime.
            # Dùng schedule absolute (không drift theo thời gian xử lý).
            if self._is_video_file and self._frame_interval > 0.0:
                now = time.time()
                if self._next_frame_t == 0.0:
                    self._next_frame_t = now + self._frame_interval
                else:
                    delay = self._next_frame_t - now
                    if delay > 0:
                        time.sleep(delay)
                    self._next_frame_t += self._frame_interval
                    # Pipeline chậm hơn FPS gốc nhiều → reset schedule
                    # để không tích lũy "nợ" thời gian.
                    if self._next_frame_t < time.time() - self._frame_interval:
                        self._next_frame_t = time.time() + self._frame_interval
        return ret, frame

    def get_fps(self) -> float:
        """Return the measured real-time FPS."""
        return self.fps_counter.get_fps()

    def release(self) -> None:
        """Release the camera resource."""
        if self._is_image_file:
            self._image_frame = None
            self.logger.info("Image source released.")
            return
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self.logger.info("Camera released.")