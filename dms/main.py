import cv2
from detection.face_detector import FaceDetector

detector = FaceDetector("models/det_2.5g.onnx")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)

    print(result)

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
