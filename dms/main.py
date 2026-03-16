import sys
import os
import cv2
from detection.face_detector import FaceDetector
from detection.head_pose_estimator import HeadPoseEstimator

sys.path.insert(0, os.path.dirname(__file__))

from input.video_capture import VideoCapture, CameraNotFoundError
from utils.logger import setup_logger
from utils.helpers import draw_bbox, draw_axis

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
MIN_FPS = 15


def main() -> None:
    logger = setup_logger("Main", CONFIG_PATH)
    logger.info("Driver Monitoring System starting")

    capture = VideoCapture(config_path=CONFIG_PATH)
    detector = FaceDetector(
        model_path="models/det_2.5g.onnx",
        input_size=(320, 320),
        conf_thres=0.5
    )
    # Head pose estimation using MobileNetV3 Small
    head_pose = HeadPoseEstimator(
        weights_path="models/mobilenetv3_small.pt"
    )

    try:
        capture.open()
        logger.info("Capture loop started. Press 'q' to quit.")

        while True:
            ret, frame = capture.read_frame()
            if not ret:
                break
            frame = cv2.flip(frame,1)
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
                    draw_bbox(frame, (x1, y1, x2, y2))

                    # Crop face for head pose estimation
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        result = head_pose.estimate_pose(face_crop)
                        if result:
                            # Draw 3D axis
                            draw_axis(frame, result.yaw, result.pitch, result.roll, [x1, y1, x2, y2])

                            # Draw head pose info
                            cv2.putText(
                                frame,
                                f"Pitch: {result.pitch:.1f}",
                                (x1, y1 - 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"Yaw: {result.yaw:.1f}",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"Roll: {result.roll:.1f}",
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

            cv2.imshow("Driver Monitoring", frame)

            if cv2.waitKey(1) == 27:
                break

            if fps < MIN_FPS:
                logger.warning(f"FPS dropped below requirement: {fps:.2f}")

    except CameraNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    finally:                                
        capture.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown")


if __name__ == "__main__":
    main()