import yaml

from utils.logger import setup_logger
from input.video_capture import VideoCapture
from detection.face_detector import FaceDetector
from detection.facemap_3dmm import FaceMap3DMMDetector
from detection.face_attrib_detector import FaceAttribDetector
from analysis.drowsiness_analyzer import DrowsinessAnalyzer
from analysis.distraction_analyzer import DistractionAnalyzer
from analysis.risk_engine import RiskEngine
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
    DistractionStage,
    RiskStage,
    DebugStage,
    VizStage,
)
from action.alert_manager import AlertManager
from action.event_logger import EventLogger
from storage.db_manager import DBManager
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
    analysis_cfg = config.get("analysis", {})
    action_cfg = config.get("action", {})

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

    # 3. Analysis Layer
    analyzer = DrowsinessAnalyzer() if model_cfg.get("analyzer", True) else None

    distraction_cfg = analysis_cfg.get("distraction", {})
    distraction_analyzer = None
    if distraction_cfg.get("enabled", True):
        distraction_analyzer = DistractionAnalyzer(
            gaze_threshold=distraction_cfg.get("gaze_threshold", 0.4),
            yaw_threshold=distraction_cfg.get("yaw_threshold", 25.0),
            lookaway_duration_sec=distraction_cfg.get("lookaway_duration_sec", 2.0),
        )

    risk_cfg = analysis_cfg.get("risk", {})
    risk_engine = RiskEngine(
        w_drowsiness=risk_cfg.get("w_drowsiness", 0.45),
        w_distraction=risk_cfg.get("w_distraction", 0.35),
        w_activity=risk_cfg.get("w_activity", 0.20),
        threshold_warning=risk_cfg.get("threshold_warning", 0.4),
        threshold_critical=risk_cfg.get("threshold_critical", 0.7),
    )

    # 4. Action Layer
    event_bus = EventBus()

    alert_cfg = action_cfg.get("alerts", {})
    alert_manager = None
    if alert_cfg.get("enabled", True):
        alert_manager = AlertManager(
            cooldown_sec=alert_cfg.get("cooldown_sec", 5.0),
        )
        event_bus.subscribe("risk", alert_manager.on_risk_event)

    logger_cfg = action_cfg.get("logger", {})
    event_logger = None
    if logger_cfg.get("enabled", True):
        db_manager = DBManager(db_path=logger_cfg.get("db_path", "data/dms.db"))
        event_logger = EventLogger(
            db_manager=db_manager,
            log_file=logger_cfg.get("log_file", "events.jsonl"),
            min_severity_to_log=logger_cfg.get("min_severity", "medium"),
        )
        event_bus.subscribe("risk", event_logger.on_risk_event)
        event_logger.start()

    # 5. UI
    visualizer = Visualizer() if model_cfg.get("visualizer", True) else None
    gaze_debug = GazeDebugLogger(
        log_every_n=10,
        enabled=config.get("debug", {}).get("gaze_logging", False),
    )

    logger.info(
        f"Models enabled → facemap={facemap is not None}, "
        f"attrib={attrib_detector is not None}, head_pose={head_pose is not None}, "
        f"analyzer={analyzer is not None}, distraction={distraction_analyzer is not None}, "
        f"visualizer={visualizer is not None}"
    )

    # 6. Build stage chain
    stages = [s for s in [
        CaptureStage(capture),
        DetectStage(face_detector, interval=5, feedback=head_pose_feedback),
        LandmarkStage(facemap),
        head_pose,
        GazeStage(eye_gaze, visualizer, debug_logger=gaze_debug),
        AttribStage(attrib_detector),
        DrowsinessStage(analyzer),
        DistractionStage(distraction_analyzer) if distraction_analyzer else None,
        RiskStage(risk_engine, event_bus=event_bus),
        DebugStage(logger, config),
        VizStage(visualizer, capture),
    ] if s is not None]

    threaded = config.get("system", {}).get("threaded", False)
    pipeline = DMSPipeline(stages, event_bus=event_bus, threaded=threaded)

    if threaded:
        logger.info("Running in THREADED mode (capture + inference on separate threads)")
    else:
        logger.info("Running in SEQUENTIAL mode")

    # 7. Run
    try:
        pipeline.start(capture)
    except KeyboardInterrupt:
        logger.info("User stopped the system.")
    finally:
        if event_logger:
            event_logger.stop()
        logger.info("System shutdown")


if __name__ == "__main__":
    main()
