# core/visualizer.py
import cv2

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_warning = (0, 0, 255)
        self.color_normal = (0, 255, 0)

    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), self.font, 1, self.color_normal, 2)

    def draw_no_face_warning(self, frame):
        cv2.putText(frame, "No face detected", (20, 80), self.font, 1, self.color_warning, 2)

    def draw_face_info(self, frame, bbox, landmarks, driver_state):
        # Vẽ bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_normal, 2)
        
        # Vẽ lưới 3DMM
        if landmarks is not None:
            # Code vẽ điểm mốc của bạn ở đây
            pass
            
        
        return frame

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)