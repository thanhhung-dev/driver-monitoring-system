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
    

    def draw_gaze(
        self,image_in: np.ndarray,
        eye_pos: np.ndarray,
        pitchyaw: np.ndarray,
        length: float | None = None,
        thickness: int = 2,
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """
        Draw gaze angle on given image with specified eye positions.

        - Adaptive arrow length based on image size if length is None.
        - Normalizes direction vector so arrow length is consistent regardless of angle magnitude.
        - Keeps the arrow endpoint within image bounds.

        Parameters
        ----------
        image_in
            Input image (grayscale or BGR).
        eye_pos
            Eye position coordinates [x, y].
        pitchyaw
            Gaze angles [pitch, yaw].
        length
            Length of the gaze arrow. If None, computed as 0.35 * min(H, W).
        thickness
            Thickness of the gaze arrow.
        color
            Color of the gaze arrow in RGB format.

        Returns
        -------
        output_image : np.ndarray
            Image with gaze arrow drawn.
        """
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

        H, W = image_out.shape[:2]
        # Adaptive length based on image size
        if length is None:
            length = 0.35 * float(min(H, W))

        # 2D direction from pitch/yaw
        dx = -np.sin(float(pitchyaw[1]))
        dy = np.sin(float(pitchyaw[0]))

        # Normalize to unit length to keep arrow length consistent
        norm = np.hypot(dx, dy)
        if norm > 1e-6:
            dx /= norm
            dy /= norm

        dx *= length
        dy *= length

        start_pt = np.round(eye_pos).astype(np.int32)
        end_pt = np.array([eye_pos[0] + dx, eye_pos[1] + dy], dtype=np.float32)

        # Keep endpoint inside image bounds
        end_pt[0] = np.clip(end_pt[0], 0, W - 1)
        end_pt[1] = np.clip(end_pt[1], 0, H - 1)
        end_pt_int = tuple(np.round(end_pt).astype(np.int32))

        cv2.arrowedLine(
            image_out,
            tuple(start_pt),
            end_pt_int,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        return image_out

    def draw_gaze_3d(
        self,
        image: np.ndarray,
        eye_pos: np.ndarray,
        v_world: np.ndarray,
        length: float | None = None,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws a 3D gaze vector projected onto the 2D image.
        """
        H, W = image.shape[:2]
        if length is None:
            length = 0.35 * float(min(H, W))

        # In our world space: +X is right, +Y is down, +Z is into screen
        # So dx and dy are simply the X and Y components of the vector.
        dx = v_world[0] * length
        dy = v_world[1] * length

        start_pt = tuple(np.round(eye_pos).astype(int))
        end_pt = (int(round(eye_pos[0] + dx)), int(round(eye_pos[1] + dy)))

        # Clamp to image bounds
        end_pt = (
            max(0, min(W - 1, end_pt[0])),
            max(0, min(H - 1, end_pt[1]))
        )

        cv2.arrowedLine(image, start_pt, end_pt, color, thickness, cv2.LINE_AA, tipLength=0.2)
        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)
