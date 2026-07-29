from pathlib import Path

import cv2
import numpy as np

from alerting.alert_manager import AlertManager
from app.application import Application
from features.distraction.analyzer import DistractionAnalyzer
from features.distraction.stage import DistractionStage
from features.drowsiness.analyzer import DrowsinessAnalyzer
from features.drowsiness.stage import DrowsinessStage
from features.face.attribute_detector import FaceAttribDetector
from features.face.attribute_stage import AttribStage
from features.face.detector import FaceDetector
from features.face.identity_stage import DriverIdentityStage
from features.face.recognizer import ArcFaceRecognizer
from features.face.stage import DetectStage
from features.gaze.debug import GazeDebugLogger
from features.gaze.estimator import EyeGazeEstimation
from features.gaze.stage import GazeStage
from features.head_pose.feedback import HeadPoseFeedback
from features.head_pose.stage import HeadPoseStage
from features.landmarks.detector import FaceMap3DMMDetector
from features.landmarks.stage import LandmarkStage
from features.risk.engine import RiskEngine
from features.risk.stage import RiskStage
from infrastructure.camera import VideoCapture
from infrastructure.capture_stage import CaptureStage
from infrastructure.database import DBManager
from infrastructure.event_logger import EventLogger
from pipeline.event_bus import EventBus
from pipeline.runner import DMSPipeline
from presentation.opencv.debug_stage import DebugStage
from presentation.opencv.stage import VizStage
from presentation.opencv.visualizer import Visualizer
from utils.logger import load_yaml_config, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: str | Path) -> dict:
    return load_yaml_config(str(path))


def create_application(config_path: str | Path = CONFIG_PATH) -> Application:
    """Construct the application and its ordered processing pipeline."""
    config_path = Path(config_path).resolve()
    logger = setup_logger("Main", str(config_path))
    logger.info("Initializing Driver Monitoring System...")

    config = _load_config(config_path)
    model_cfg = config.get("models", {})
    analysis_cfg = config.get("analysis", {})
    action_cfg = config.get("action", {})

    # 1. Input
    capture = VideoCapture(config_path=str(config_path))

    # 2. AI Models
    face_detector = FaceDetector(model_path=str(PROJECT_ROOT / "models/det_2.5g.onnx"))
    identity_stage = None
    if model_cfg.get("driver_identity", False):
        identity_cfg = config.get("identity", {})
        model_path = PROJECT_ROOT / identity_cfg.get("model_path", "models/w600k_mbf.onnx")
        recognizer = ArcFaceRecognizer(str(model_path))

        drivers_dir = identity_cfg.get("drivers_dir")
        if drivers_dir:
            dir_path = PROJECT_ROOT / drivers_dir
            driver_embeddings_map = {}
            if dir_path.is_dir():
                for driver_folder in dir_path.iterdir():
                    if not driver_folder.is_dir():
                        continue
                    driver_name = driver_folder.name
                    embeddings = []
                    for img_file in driver_folder.glob("*.*"):
                        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                            continue
                        img = cv2.imread(str(img_file))
                        if img is None:
                            continue
                        dets, kpss = face_detector.detect(img)
                        if kpss is not None and len(kpss) > 0:
                            # Take the largest face if multiple
                            best_face_idx = 0
                            if len(dets) > 1:
                                areas = [(d[2]-d[0])*(d[3]-d[1]) for d in dets]
                                best_face_idx = int(np.argmax(areas))
                            emb = recognizer.get_embedding(img, kpss[best_face_idx])
                            embeddings.append(emb)
                    if embeddings:
                        driver_embeddings_map[driver_name] = np.vstack(embeddings)
                
                if not driver_embeddings_map:
                    logger.warning(f"No valid faces found in drivers directory: {drivers_dir}")
            else:
                logger.warning(f"Drivers directory not found: {drivers_dir}")
        else:
            raise ValueError("Driver identity requires identity.drivers_dir")

        identity_stage = DriverIdentityStage(
            recognizer=recognizer,
            driver_embeddings=driver_embeddings_map,
            similarity_threshold=identity_cfg.get("similarity_threshold", 0.4),
            driver_roi=identity_cfg.get("driver_roi", [0.45, 0.0, 1.0, 1.0]),
        )
    facemap = FaceMap3DMMDetector() if model_cfg.get("facemap", True) else None
    attrib_detector = FaceAttribDetector() if model_cfg.get("attrib", True) else None
    eye_gaze = EyeGazeEstimation() if model_cfg.get("eye_gaze", False) else None

    # Shared cross-frame channel: HeadPoseStage ghi head pose, DetectStage đọc
    # ở frame kế tiếp để bật extreme_pose_mode khi yaw quá lớn.
    head_pose_feedback = HeadPoseFeedback()

    head_pose = None
    if model_cfg.get("head_pose", False):
        head_pose = HeadPoseStage(
            model_path=str(PROJECT_ROOT / "models/resnet50.onnx"),
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
        db_path = Path(logger_cfg.get("db_path", "data/dms.db"))
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        db_manager = DBManager(db_path=str(db_path))
        event_logger = EventLogger(
            db_manager=db_manager,
            log_dir=str(PROJECT_ROOT / "logs"),
            log_file=logger_cfg.get("log_file", "events.jsonl"),
            min_severity_to_log=logger_cfg.get("min_severity", "medium"),
        )
        event_bus.subscribe("risk", event_logger.on_risk_event)

    # 5. UI
    visualizer = Visualizer() if model_cfg.get("visualizer", True) else None
    gaze_debug = GazeDebugLogger(
        log_every_n=10,
        enabled=config.get("debug", {}).get("gaze_logging", False),
    )

    logger.info(
        f"Models enabled → facemap={facemap is not None}, "
        f"driver_identity={identity_stage is not None}, "
        f"attrib={attrib_detector is not None}, head_pose={head_pose is not None}, "
        f"analyzer={analyzer is not None}, distraction={distraction_analyzer is not None}, "
        f"visualizer={visualizer is not None}"
    )

    # 6. Build stage chain
    stages = [s for s in [
        CaptureStage(capture),
        DetectStage(face_detector, interval=5, feedback=head_pose_feedback),
        identity_stage,
        LandmarkStage(facemap),
        head_pose,
        GazeStage(eye_gaze, visualizer, debug_logger=gaze_debug),
        AttribStage(attrib_detector),
        DrowsinessStage(analyzer),
        DistractionStage(distraction_analyzer) if distraction_analyzer else None,
        RiskStage(risk_engine, event_bus=event_bus),
        DebugStage(logger, config),
        VizStage(
            visualizer, 
            capture
        ),
    ] if s is not None]

    threaded = config.get("system", {}).get("threaded", False)
    pipeline = DMSPipeline(stages, event_bus=event_bus, threaded=threaded)

    if threaded:
        logger.info("Running in THREADED mode (capture + inference on separate threads)")
    else:
        logger.info("Running in SEQUENTIAL mode")

    return Application(
        pipeline=pipeline,
        capture=capture,
        logger=logger,
        event_logger=event_logger,
    )
