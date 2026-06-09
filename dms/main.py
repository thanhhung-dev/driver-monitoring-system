import sys
import yaml

from utils.logger import setup_logger
from input.video_capture import VideoCapture
from detection.face_detector import FaceDetector
from detection.facemap_3dmm import FaceMap3DMMDetector
from detection.face_attrib_detector import FaceAttribDetector
from analysis.drowsiness_analyzer import DrowsinessAnalyzer
from core.pipeline import DMSPipeline
from core.event_bus import EventBus
from core.head_pose_feedback import HeadPoseFeedback
from core.visualizer import Visualizer
from detection.eye_gaze import EyeGazeEstimation
from core.stages import (
    CaptureStage,
    DetectStage,
    LandmarkStage,
    HeadPoseStage,
    GazeStage,
    AttribStage,
    DrowsinessStage,
    DebugStage,
    VizStage,
)
from utils.gaze_debug_helper import GazeDebugLogger

CONFIG_PATH = "config.yaml"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    logger = setup_logger("Main")
    logger.info("Initializing Driver Monitoring System...")

    config = _load_config(CONFIG_PATH)
    model_cfg = config.get("models", {})

    # 1. Input
    capture = VideoCapture(config_path=CONFIG_PATH)

    # 2. AI Models
    face_detector = FaceDetector(model_path="models/det_2.5g.onnx")
    facemap = FaceMap3DMMDetector() if model_cfg.get("facemap", True) else None
    attrib_detector = FaceAttribDetector() if model_cfg.get("attrib", True) else None
    eye_gaze = EyeGazeEstimation() if model_cfg.get("eye_gaze", False) else None

    # Shared cross-frame channel: HeadPoseStage ghi head pose, DetectStage đọc
    # ở frame kế tiếp để bật extreme_pose_mode khi yaw quá lớn.
    head_pose_feedback = HeadPoseFeedback()

    head_pose = None
    if model_cfg.get("head_pose", False):
        head_pose = HeadPoseStage(
            model_path="models/resnet50.onnx",
            feedback=head_pose_feedback,
        )

    # 3. Logic & UI
    analyzer = DrowsinessAnalyzer() if model_cfg.get("analyzer", True) else None
    visualizer = Visualizer() if model_cfg.get("visualizer", True) else None
    gaze_debug = GazeDebugLogger(
        log_every_n=10,
        enabled=config.get("debug", {}).get("gaze_logging", False),
    )

    logger.info(
        f"Models enabled → facemap={facemap is not None}, "
        f"attrib={attrib_detector is not None}, head_pose={head_pose is not None}, "
        f"analyzer={analyzer is not None}, visualizer={visualizer is not None}"
    )

    # 4. Build stage chain
    stages = [s for s in [
        CaptureStage(capture),
        DetectStage(face_detector, interval=5, feedback=head_pose_feedback),
        LandmarkStage(facemap),
        head_pose,
        GazeStage(eye_gaze, visualizer, debug_logger=gaze_debug),
        AttribStage(attrib_detector),
        DrowsinessStage(analyzer),
        DebugStage(logger, config),
        VizStage(visualizer, capture),
    ] if s is not None]

    event_bus = EventBus()
    threaded = config.get("system", {}).get("threaded", False)
    pipeline = DMSPipeline(stages, event_bus=event_bus, threaded=threaded)

    if threaded:
        logger.info("Running in THREADED mode (capture + inference on separate threads)")
    else:
        logger.info("Running in SEQUENTIAL mode")

    # 5. Run
    try:
        pipeline.start(capture)
    except KeyboardInterrupt:
        logger.info("User stopped the system.")
    finally:
        logger.info("System shutdown")


if __name__ == "__main__":
    main()
