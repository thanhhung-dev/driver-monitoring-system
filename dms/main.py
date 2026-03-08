import cv2
from detection.face_detector import FaceDetector

detector = FaceDetector("models/det_2.5g.onnx")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)
    boxes = output[0]
    for box in boxes:
        draw_bbox(frame, box[:4].astype(int))
    print(result)

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
