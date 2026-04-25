import sys
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

def main():
    logger = setup_logger("Main")
    logger.info("Initializing Driver Monitoring System...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo đầu vào
    capture = VideoCapture(config_path="config.yaml")

    # 2. Khởi tạo các AI Models
    face_detector = FaceDetector(model_path="models/det_2.5g.onnx")
    facemap = FaceMap3DMMDetector()
    attrib_detector = FaceAttribDetector()

    head_pose = mobilenet_v2(pretrained=False, num_classes=6)
    state_dict = torch.load("models/mobilenetv2.pt", map_location=device, weights_only=True)
    load_filtered_state_dict(head_pose, state_dict)
    head_pose.to(device)
    head_pose.eval()

    # 3. Khởi tạo Logic và UI
    analyzer = DrowsinessAnalyzer()
    visualizer = Visualizer()

    # 4. Lắp ráp tất cả vào Pipeline (Đường ống xử lý)
    pipeline = DMSPipeline(
        capture=capture,
        detector=face_detector,
        facemap=facemap,
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
