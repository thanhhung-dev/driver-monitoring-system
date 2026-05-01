# core/visualizer.py
import cv2
import numpy as np
from utils import facial_constants as fc
from utils.helpers import draw_bbox
from utils.helpers import draw_axis
from typing import Tuple, List

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_warning = (0, 0, 255)
        self.color_normal = (0, 255, 0)


    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), self.font, 1, self.color_normal, 2)

    def draw_no_face_warning(self, frame):
        cv2.putText(frame, "No face detected", (20, 80), self.font, 1, self.color_warning, 2)

    def draw_face_info(self, frame, bbox, landmarks, driver_state, head_pose=None):
        """Vẽ bbox + trục head-pose (nếu có).

        Args:
            frame:        Ảnh BGR cần vẽ lên (in-place).
            bbox:         (x1, y1, x2, y2).
            landmarks:    List 68 (x, y) hoặc None — landmark mesh đã được vẽ
                          trực tiếp trong pipeline qua FaceMap3DMMDetector.draw_full_mesh.
            driver_state: Trạng thái driver (chưa dùng, để mở rộng cảnh báo).
            head_pose:    Tuple (yaw, pitch, roll) độ, hoặc None để không vẽ trục.
        """
        # Bounding box với corner-accent
        draw_bbox(frame, bbox, self.color_normal)

        # 3D head-pose axes (chỉ vẽ khi có giá trị yaw/pitch/roll)
        if head_pose is not None:
            yaw, pitch, roll = head_pose
            draw_axis(frame, yaw, pitch, roll, list(bbox))

        return frame
    
    def draw_full_mesh(self, image, landmarks, color=(255,255,0), radius=2):

        if landmarks is None:
            return image

        scale = 2
        h, w = image.shape[:2]
        big = cv2.resize(image, (w * scale, h * scale))
        CYAN = (255, 255, 0)

        def pt(i):
            x, y = landmarks[i]
            return (int(x * scale), int(y * scale))

        def pts(idxs):
            return np.array([pt(i) for i in idxs], dtype=np.int32)

        cv2.polylines(big, [pts(fc.LEFT_EYE_INDICES)], True, CYAN, 2, cv2.LINE_AA)
        for i in fc.LEFT_EYE_INDICES:
            cv2.circle(big, pt(i), radius*scale, color, -1, cv2.LINE_AA)
        cv2.polylines(big, [pts(fc.RIGHT_EYE_INDICES)], True, CYAN, 2, cv2.LINE_AA)
        for i in fc.RIGHT_EYE_INDICES:
            cv2.circle(big, pt(i), radius*scale, color, -1, cv2.LINE_AA)
            cv2.polylines(big, [pts(fc.EYE_BROW_LEFT)], False, CYAN, 2, cv2.LINE_AA)
        for i in fc.EYE_BROW_LEFT:
            cv2.circle(big, pt(i), radius*scale, color, -1, cv2.LINE_AA)

        cv2.polylines(big, [pts(fc.EYE_BROW_RIGHT)], False, CYAN, 2, cv2.LINE_AA)
        for i in fc.EYE_BROW_RIGHT:
            cv2.circle(big, pt(i), radius*scale, color, -1, cv2.LINE_AA)

        # Ve mui
        cv2.polylines(big, [pts([27, 28, 29, 30])], False, CYAN, 2, cv2.LINE_AA)

        for idx in fc.NOISE_INDICES:
            cv2.circle(big, pt(idx), radius*scale, color, -1, lineType=cv2.LINE_AA)


        # Ve Mui Tam Giac
        v_shape = np.array([pt(31), pt(30), pt(35)], dtype=np.int32)
        cv2.polylines(big, [v_shape], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        w_shape = np.array([pt(31), pt(33), pt(35)], dtype=np.int32)
        cv2.polylines(big, [w_shape], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        for idx in fc.NOISE_TRIANGLE_INDICES:
            cv2.circle(big, pt(idx), radius*scale, color, -1, lineType=cv2.LINE_AA)


        # Ve moi ngoai
        outer_lips_pts = np.array([pt(i) for i in fc.OUTER_LIPS_INDICES], dtype=np.int32)
        cv2.polylines(big, [outer_lips_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        for idx in fc.OUTER_LIPS_INDICES:
            cv2.circle(big, pt(idx), radius*scale, color, -1, lineType=cv2.LINE_AA)


        # Ve moi trong
        inner_lips_pts = np.array([pt(i) for i in fc.INNER_LIPS_INDICES], dtype=np.int32)
        cv2.polylines(big, [inner_lips_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        for idx in fc.INNER_LIPS_INDICES:
            cv2.circle(big, pt(idx), radius*scale, color, -1, lineType=cv2.LINE_AA)


        # Ve Long May Trai
        brow_pts = np.array([pt(i) for i in fc.EYE_BROW_LEFT], dtype=np.int32)
        cv2.polylines(big, [brow_pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)

        for idx in fc.EYE_BROW_LEFT:
            cv2.circle(big, pt(idx), radius*scale, color, -1, lineType=cv2.LINE_AA)
        image[:] = cv2.resize(big, (w, h))

        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)