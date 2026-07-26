import os
import cv2
import numpy as np
import onnxruntime
import torch
from typing import Optional, Tuple, List
import onnx
from features.landmarks import constants as fc
from utils.onnx_providers import make_session
from onnx.external_data_helper import load_external_data_for_model

class FaceMap3DMMDetector:
    """Qualcomm AI Hub Facial-Landmark-Detection (facemap_3dmm) ONNX wrapper.

    Outputs 68 3D facial landmarks from a 128x128 face crop.
    The ONNX model outputs 265 3DMM parameters which are decoded
    into 2D landmark coordinates using pre-computed basis matrices.
    """
    def __init__(
        self,
        model_dir: str = "models/face-lanmark-detection",
        bbox_pad_ratio_x: float = 0.5,  
        bbox_pad_ratio_y: float = 0.10,  
    ) -> None:
        self.bbox_pad_ratio_x = bbox_pad_ratio_x
        self.bbox_pad_ratio_y = bbox_pad_ratio_y
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "facemap_3dmm.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")


        onnx_model = onnx.load(onnx_path, load_external_data=False)
        load_external_data_for_model(onnx_model, model_dir)
        model_bytes = onnx_model.SerializeToString()

        self.session = make_session(model_bytes)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = int(shape[2])
        else:
            self.input_size = fc.INPUT_SIZE

        # Load 3DMM basis matrices (numpy float32 — nhanh hơn torch CPU vì không
        # tạo autograd graph + tránh overhead chuyển đổi mỗi frame).
        self.mean_face = np.load(os.path.join(model_dir, "meanFace.npy")).reshape(
            3 * fc.VERTEX_NUM, 1
        ).astype(np.float32)
        self.basis_id = np.load(os.path.join(model_dir, "shapeBasis.npy")).reshape(
            3 * fc.VERTEX_NUM, fc.ALPHA_ID_SIZE
        ).astype(np.float32)
        self.basis_exp = np.load(os.path.join(model_dir, "blendShape.npy")).reshape(
            3 * fc.VERTEX_NUM, fc.ALPHA_EXP_SIZE
        ).astype(np.float32)

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize to model input size, convert BGR->RGB, normalize to [0,1], NCHW."""
        resized = cv2.resize(
            face_crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    def _project_landmark(self, output: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Decode 265 3DMM params into 68 (x, y) landmark coordinates and head pose.

        Triển khai bằng numpy thay vì torch CPU → giảm overhead 5-10× trên Windows
        (không tạo autograd graph, không round-trip torch↔numpy)."""
        out = np.asarray(output, dtype=np.float32)
        ratio = self.input_size / 128.0

        alpha_id = out[0:219] * 3.0
        alpha_exp = out[219:258] * 0.5 + 0.5

        # Pose angles (radians)
        pitch = float(out[258]) * (np.pi / 2)
        yaw   = float(out[259]) * (np.pi / 2)
        roll  = float(out[260]) * (np.pi / 2)

        tX = float(out[261]) * 60.0 * ratio
        tY = float(out[262]) * 60.0 * ratio
        tZ = 500.0
        f  = (float(out[263]) * 150.0 + 450.0) * ratio

        P = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float32)

        cp, sp = np.cos(-pitch), np.sin(-pitch)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)

        cy, sy = np.cos(-yaw), np.sin(-yaw)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)

        cr, sr = np.cos(-roll), np.sin(-roll)
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        r_matrix = ry @ rx @ P @ rz

        shape = (
            self.mean_face
            + self.basis_id @ alpha_id.reshape(-1, 1)
            + self.basis_exp @ alpha_exp.reshape(-1, 1)
        ).reshape(fc.VERTEX_NUM, 3)

        vertices = shape @ r_matrix.T
        vertices[:, 0] += tX
        vertices[:, 1] += tY
        vertices[:, 2] += tZ
        landmarks_2d = vertices[:, 0:2] * f / tZ

        return landmarks_2d, (pitch, yaw, roll)

    def detect(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks_5: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[List[Tuple[int, int]], Tuple[float, float, float]]]:
        """Detect 68 facial landmarks and head pose given a frame and face bounding box.

        Args:
            frame: Full BGR image.
            bbox: Face bounding box as (x1, y1, x2, y2). Đã được detector pad sẵn.
            landmarks_5: Không sử dụng (giữ để tương thích interface).

        Returns:
            Tuple containing:
              - List of 68 (x, y) pixel coordinates in full-frame space.
              - Tuple of (pitch, yaw, roll) in radians.
            Returns None if the crop is invalid.
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        px = bw * self.bbox_pad_ratio_x
        py = bh * self.bbox_pad_ratio_y
        x1 = x1 - px
        y1 = y1 - py
        x2 = x2 + px
        y2 = y2 + py

        # Clamp bbox vào trong frame
        ix1 = max(int(round(x1)), 0)
        iy1 = max(int(round(y1)), 0)
        ix2 = min(int(round(x2)), w_frame)
        iy2 = min(int(round(y2)), h_frame)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        face_crop = frame[iy1:iy2, ix1:ix2]
        crop_h, crop_w = face_crop.shape[:2]
        blob = self._preprocess(face_crop)
        output = self.session.run(None, {self.input_name: blob})[0][0]  # (265,)
        landmark, head_pose = self._project_landmark(output)
        proj = float(self.input_size)
        landmark[:, 0] = (landmark[:, 0] + proj / 2) * crop_w / proj + ix1
        landmark[:, 1] = (landmark[:, 1] + proj / 2) * crop_h / proj + iy1

        landmarks_list = [(int(x), int(y)) for x, y in landmark]
        return landmarks_list, head_pose
