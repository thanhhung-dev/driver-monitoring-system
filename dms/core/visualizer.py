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
        pitchyaw: np.ndarray,
        length_px: float = 100.0,
        thickness: int = 2,
        draw_axes: bool = True,
    ) -> np.ndarray:
        """
        Vẽ mũi tên gaze 3D bằng pinhole projection để có cảm giác chiều sâu.

        Convention (khớp Qualcomm EyeNet):
          • pitch > 0 → nhìn xuống (+y trên ảnh)
          • yaw   > 0 → nhìn sang phải của khuôn mặt (−x trên ảnh)

        Trục local tại mắt:
          • X (đỏ)  : phải
          • Y (lục) : xuống
          • Z (lam) : hướng gaze (mũi tên chính)
        """
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        H, W = image.shape[:2]
        pitch = float(pitchyaw[0])
        yaw   = float(pitchyaw[1])

        # Camera intrinsics xấp xỉ (không cần calibration)
        f  = float(max(W, H))
        cx, cy = W * 0.5, H * 0.5

        # Đặt mắt ở độ sâu z0 = 1 trong không gian camera, back-project từ pixel
        z0 = 1.0
        eye3d = np.array([
            (float(eye_pos[0]) - cx) / f * z0,
            (float(eye_pos[1]) - cy) / f * z0,
            z0,
        ], dtype=np.float64)

        # Vector gaze 3D (đơn vị) — match công thức 2D của Qualcomm
        g = np.array([
            -np.sin(yaw),
             np.sin(pitch),
             np.cos(yaw) * np.cos(pitch),
        ], dtype=np.float64)
        g /= (np.linalg.norm(g) + 1e-9)

        # Độ dài "thật" trong không gian (quy đổi từ pixel)
        L = float(length_px) / f * z0

        # Build basis trực giao tại mắt với Z = gaze, Y ≈ down, X = Y×Z
        up_world = np.array([0.0, 1.0, 0.0])      # +y trên ảnh là "down"
        x_axis = np.cross(up_world, g)
        nx = np.linalg.norm(x_axis)
        if nx < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= nx
        y_axis = np.cross(g, x_axis)

        tip_z = eye3d + g       * L
        tip_x = eye3d + x_axis  * L * 0.5
        tip_y = eye3d + y_axis  * L * 0.5

        def project(p):
            if p[2] <= 1e-6:
                return None
            return (
                int(round(f * p[0] / p[2] + cx)),
                int(round(f * p[1] / p[2] + cy)),
            )

        o  = project(eye3d)
        pz = project(tip_z)
        if o is None or pz is None:
            return image

        if draw_axes:
            px = project(tip_x)
            py = project(tip_y)
            if px is not None:
                cv2.line(image, o, px, (0, 0, 255), thickness, cv2.LINE_AA)   # X đỏ
            if py is not None:
                cv2.line(image, o, py, (0, 255, 0), thickness, cv2.LINE_AA)   # Y lục

        # Mũi tên gaze chính — dày hơn + chấm đầu để nhấn 3D
        cv2.arrowedLine(image, o, pz, (255, 128, 0),
                        thickness + 1, cv2.LINE_AA, tipLength=0.25)             # Z lam-cam
        cv2.circle(image, o, max(3, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, pz, max(3, thickness),    (255, 200, 100), -1, cv2.LINE_AA)
        return image

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)
