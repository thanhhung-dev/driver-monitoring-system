import sys
import os
import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from detection.face_detector import FaceDetector
from detection.facemap_3dmm import FaceMap3DMMDetector
from detection.mobilenetv2 import mobilenet_v2
sys.path.insert(0, os.path.dirname(__file__))

from input.video_capture import VideoCapture, CameraNotFoundError
from analysis.drowsiness_analyzer import DrowsinessAnalyzer, DriverState, STATE_COLORS
from utils.logger import setup_logger
from utils.helpers import draw_bbox, draw_axis, expand_bbox
from utils.general import compute_euler_angles_from_rotation_matrices
from detection.common import load_filtered_state_dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
MIN_FPS = 15
OUTPUT_PATH = "output.mp4"

# Preprocessing transform (ImageNet normalization)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main() -> None:
    logger = setup_logger("Main", CONFIG_PATH)
    logger.info("Driver Monitoring System starting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load analysis config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    analysis_cfg = cfg.get("analysis", {})
    cam_fps = cfg["camera"]["fps"]

    analyzer = DrowsinessAnalyzer(
        fps=cam_fps,
        **{k: v for k, v in analysis_cfg.items() if v is not None},
    )
    logger.info("Drowsiness analyzer initialized")

    capture = VideoCapture(config_path=CONFIG_PATH)
    detector = FaceDetector(
        model_path="models/det_2.5g.onnx",
        input_size=(320, 320),
        conf_thres=0.5
    )
    facemap_detector = FaceMap3DMMDetector()
    logger.info("FaceMap 3DMM landmark detector loaded")
    # Load MobileNetV2 head pose model directly
    head_pose = mobilenet_v2(pretrained=False, num_classes=6)
    state_dict = torch.load("models/mobilenetv2.pt", map_location=device, weights_only=True)
    load_filtered_state_dict(head_pose, state_dict)
    head_pose.to(device)
    head_pose.eval()
    logger.info("MobileNetV2 head pose model loaded")
    video_writer = None

    try:
        capture.open()
        logger.info("Capture loop started. Press 'q' to quit.")

        with torch.no_grad():
            while True:
                ret, frame = capture.read_frame()
                if not ret:
                    break
                if not capture._is_video_file:
                    frame = cv2.flip(frame, 1)

                # Initialize video writer on first frame
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 15, (w, h))
                    logger.info(f"Saving output to {OUTPUT_PATH}")

                det, _ = detector.detect(frame)

                # Handle empty detections
                if det is None or len(det) == 0:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                else:
                    for box in det:
                        x1, y1, x2, y2 = box[:4].astype(int)
                        bbox_width = x2 - x1
                        draw_bbox(frame, (x1, y1, x2, y2))

                        # Pad bbox slightly for better landmark fit
                        pad = int(0.05 * bbox_width)
                        h_frame, w_frame = frame.shape[:2]
                        lx1 = max(0, x1 - pad)
                        ly1 = max(0, y1 - pad)
                        lx2 = min(w_frame, x2 + pad)
                        ly2 = min(h_frame, y2 + pad)

                        # FaceMap 3DMM landmarks (68-point, Qualcomm)
                        facemap_lmks = facemap_detector.detect(frame, (lx1, ly1, lx2, ly2))
                        if facemap_lmks:
                            facemap_detector.draw_full_mesh(frame, facemap_lmks)
            
                        # Expand bbox for better head pose estimation
                        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2)
                        h, w = frame.shape[:2]
                        ex2 = min(ex2, w)
                        ey2 = min(ey2, h)

                        # Crop & preprocess face
                        face_crop = frame[ey1:ey2, ex1:ex2]
                        pitch, yaw, roll = 0.0, 0.0, 0.0
                        if face_crop.size > 0:
                            image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            image = preprocess(image).unsqueeze(0).to(device)

                            rotation_matrix = head_pose(image)
                            euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                            pitch = float(euler[:, 0].cpu())
                            yaw = float(euler[:, 1].cpu())
                            roll = float(euler[:, 2].cpu())

                            draw_axis(frame, yaw, pitch, roll, [x1, y1, x2, y2])

                        # ── Analysis pipeline ──
                        if facemap_lmks:
                            state = analyzer.update(facemap_lmks, pitch, yaw)
                            color = STATE_COLORS[state]

                            # HUD overlay
                            cv2.putText(frame, f"State: {state.value}", (20, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(frame, f"EAR: {analyzer.ear:.2f}", (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                            cv2.putText(frame, f"MAR: {analyzer.mar:.2f}", (20, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                            cv2.putText(frame, f"PERCLOS: {analyzer.perclos:.1f}%", (20, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                            cv2.putText(frame, f"Score: {analyzer.drowsy_score:.2f}", (20, 210),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                fps = capture.get_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                video_writer.write(frame)
                cv2.imshow("Driver Monitoring", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if fps < MIN_FPS:
                    logger.warning(f"FPS dropped below requirement: {fps:.2f}")

    except CameraNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    finally:
        if video_writer:
            video_writer.release()
            logger.info(f"Output saved to {OUTPUT_PATH}")
        capture.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown")


if __name__ == "__main__":
    main()