import cv2
from detection.face_detector import FaceDetector

sys.path.insert(0, os.path.dirname(__file__))
 
from input.video_capture import VideoCapture, CameraNotFoundError
from utils.logger import setup_logger

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
MIN_FPS = 15  # NFR1 requirement


def main() -> None:
    logger = setup_logger("Main", CONFIG_PATH)
    logger.info("Driver Monitoring System starting")
 
    capture = VideoCapture(config_path=CONFIG_PATH)
    detector = FaceDetector("models/det_2.5g.onnx")
    try:
        capture.open()
    except CameraNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Capture loop started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

    result = detector.detect(frame)
    boxes = output[0]
    
    for box in boxes:
        draw_bbox(frame, box[:4].astype(int))
    
    # FPS monitor
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
cap.release()
cv2.destroyAllWindows()
logger.info("System shutdown")
if __name__ == "__main__":
    main()
