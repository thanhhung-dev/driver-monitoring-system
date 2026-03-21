import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from detection.face_detector import FaceDetector
from detection.face_mesh import FaceMeshDetector
from detection.mobilenetv2 import mobilenet_v2

sys.path.insert(0, os.path.dirname(__file__))

from input.video_capture import VideoCapture, CameraNotFoundError
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

    capture = VideoCapture(config_path=CONFIG_PATH)
    detector = FaceDetector(
        model_path="models/det_2.5g.onnx",
        input_size=(320, 320),
        conf_thres=0.5
    )
    mesh_detector = FaceMeshDetector()

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

                        # Face Mesh (MediaPipe) — detect on cropped face, draw on full frame
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            landmarks = mesh_detector.detect(face_crop)
                            if landmarks:
                                # Convert crop coords back to full-frame coords
                                offset_landmarks = [(x + x1, y + y1) for (x, y) in landmarks]
                                frame = mesh_detector.draw_full_mesh(frame, offset_landmarks)

                        # Expand bbox for better head pose estimation
                        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2)
                        h, w = frame.shape[:2]
                        ex2 = min(ex2, w)
                        ey2 = min(ey2, h)

                        # Crop & preprocess face
                        face_crop = frame[ey1:ey2, ex1:ex2]
                        if face_crop.size > 0:
                            image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            image = preprocess(image).unsqueeze(0).to(device)

                            rotation_matrix = head_pose(image)
                            euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                            pitch = float(euler[:, 0].cpu())
                            yaw = float(euler[:, 1].cpu())
                            roll = float(euler[:, 2].cpu())

                            draw_axis(frame, yaw, pitch, roll, [x1, y1, x2, y2])

                            cv2.putText(
                                frame,
                                f"Pitch: {pitch:.1f}",
                                (x1, y1 - 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"Yaw: {yaw:.1f}",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"Roll: {roll:.1f}",
                                (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )

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

                if cv2.waitKey(1) == 27:
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