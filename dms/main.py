import sys
import yaml
import torch
from utils.logger import setup_logger
from input.video_capture import VideoCapture
from detection.face_detector import FaceDetector
from detection.facemap_3dmm import FaceMap3DMMDetector
from detection.face_attrib_detector import FaceAttribDetector
from detection.mobilenetv2 import mobilenet_v2
from detection.common import load_filtered_state_dict
from analysis.drowsiness_analyzer import DrowsinessAnalyzer
from core.pipeline import DMSPipeline
from core.visualizer import Visualizer
from detection.eye_gaze import EyeGazeEstimation

CONFIG_PATH = "config.yaml"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    logger = setup_logger("Main")
    logger.info("Initializing Driver Monitoring System...")

    config = _load_config(CONFIG_PATH)
    model_cfg = config.get("models", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo đầu vào
    capture = VideoCapture(config_path=CONFIG_PATH)

    # 2. Khởi tạo các AI Models theo config (đặt = None để pipeline skip)
    face_detector = FaceDetector(model_path="models/det_2.5g.onnx")  # bắt buộc

    facemap = FaceMap3DMMDetector() if model_cfg.get("facemap", True) else None
    attrib_detector = FaceAttribDetector() if model_cfg.get("attrib", True) else None
    eye_gaze = EyeGazeEstimation() if model_cfg.get("eye_gaze", False) else None

    head_pose = None
    if model_cfg.get("head_pose", False):
        head_pose = mobilenet_v2(pretrained=False, num_classes=6)
        state_dict = torch.load(
            "models/mobilenetv2.pt", map_location=device, weights_only=True
        )
        load_filtered_state_dict(head_pose, state_dict)
        head_pose.to(device)
        head_pose.eval()

    # 3. Logic và UI
    analyzer = DrowsinessAnalyzer() if model_cfg.get("analyzer", True) else None
    visualizer = Visualizer() if model_cfg.get("visualizer", True) else None

    logger.info(
        f"Models enabled → facemap={facemap is not None}, "
        f"attrib={attrib_detector is not None}, head_pose={head_pose is not None}, "
        f"analyzer={analyzer is not None}, visualizer={visualizer is not None}"
    )

    # 4. Lắp ráp tất cả vào Pipeline (Đường ống xử lý)
    pipeline = DMSPipeline(
        capture=capture,
        detector=face_detector,
        facemap=facemap,
        eye_gaze=eye_gaze,
        attrib_detector=attrib_detector,
        head_pose=head_pose,
        analyzer=analyzer,
        visualizer=visualizer,
        device=device,
    )

    # 5. Chạy hệ thống
    try:
        pipeline.start()
    except KeyboardInterrupt:
        logger.info("Người dùng đã dừng chương trình.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()