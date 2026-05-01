import os
import cv2
import numpy as np
import onnxruntime
import torch
from typing import Optional, Tuple, List
import onnx
from utils import facial_constants as fc
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
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "facemap_3dmm.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Model uses external data file (model.data) — must load from directory

        onnx_model = onnx.load(onnx_path, load_external_data=False)
        load_external_data_for_model(onnx_model, model_dir)
        model_bytes = onnx_model.SerializeToString()

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = onnxruntime.InferenceSession(
            model_bytes,
            sess_options=sess_options,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        # Auto-detect spatial input size from the model (NCHW: [N, C, H, W]).
        shape = input_meta.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = int(shape[2])
        else:
            self.input_size = fc.INPUT_SIZE

        # Load 3DMM basis matrices
        self.mean_face = torch.from_numpy(
            np.load(os.path.join(model_dir, "meanFace.npy")).reshape(
                3 * fc.VERTEX_NUM, 1
            )
        )
        self.basis_id = torch.from_numpy(
            np.load(os.path.join(model_dir, "shapeBasis.npy")).reshape(
                3 * fc.VERTEX_NUM, fc.ALPHA_ID_SIZE
            )
        )
        self.basis_exp = torch.from_numpy(
            np.load(os.path.join(model_dir, "blendShape.npy")).reshape(
                3 * fc.VERTEX_NUM, fc.ALPHA_EXP_SIZE
            )
        )

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize to model input size, convert BGR->RGB, normalize to [0,1], NCHW."""
        resized = cv2.resize(
            face_crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    def _project_landmark(self, output: np.ndarray) -> torch.Tensor:
        """Decode 265 3DMM params into 68 (x, y) landmark coordinates."""
        out = torch.from_numpy(output)

        # Calculate scale ratio relative to baseline 128x128
        ratio = self.input_size / 128.0

        # 1. Scaling coefficients (chuẩn theo Qualcomm facemap_3dmm)
        alpha_id = out[0:219] * 3.0
        alpha_exp = out[219:258] * 0.5 + 0.5

        # Pose angles (radians)
        pitch = out[258] * (np.pi / 2)
        yaw   = out[259] * (np.pi / 2)
        roll  = out[260] * (np.pi / 2)

        tX = out[261] * 60.0 * ratio
        tY = out[262] * 60.0 * ratio
        tZ = 500.0
        f  = (out[263] * 150.0 + 450.0) * ratio

        # 2. Ma trận xoay và lật trục
        # P lật trục Y và Z để khớp với hệ tọa độ ảnh (Y hướng xuống)
        # Tương đương xoay quanh trục X một góc -pi (theo reference Qualcomm).
        P = torch.tensor([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=torch.float32)

        # Reference Qualcomm dùng GÓC ÂM cho cả pitch / yaw / roll
        cp, sp = torch.cos(-pitch), torch.sin(-pitch)
        rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

        cy, sy = torch.cos(-yaw), torch.sin(-yaw)
        ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        cr, sr = torch.cos(-roll), torch.sin(-roll)
        rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

        # Thứ tự nhân chuẩn của model này: Yaw * Pitch * P * Roll
        r_matrix = torch.mm(ry, torch.mm(rx, torch.mm(P, rz)))

        # 3. Reconstruct 3D Shape
        shape = (
            self.mean_face
            + torch.mm(self.basis_id, alpha_id.view(-1, 1))
            + torch.mm(self.basis_exp, alpha_exp.view(-1, 1))
        ).view(fc.VERTEX_NUM, 3)

        # 4. Transform: V' = R * V + T
        vertices = torch.mm(shape, r_matrix.t())

        # Tịnh tiến theo đúng reference Qualcomm (KHÔNG cộng thêm offset nào)
        vertices[:, 0] += tX
        vertices[:, 1] += tY
        vertices[:, 2] += tZ

        # 5. Weak Perspective Projection
        # f/tZ đóng vai trò là scale factor. f khoảng 450-600, tZ=500 -> scale ~1.0-1.2
        landmarks_2d = vertices[:, 0:2] * f / tZ

        return landmarks_2d

    def detect(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Detect 68 facial landmarks given a frame and face bounding box.

        Args:
            frame: Full BGR image.
            bbox: Face bounding box as (x1, y1, x2, y2).

        Returns:
            List of 68 (x, y) pixel coordinates in full-frame space,
            or None if the crop is invalid.
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1)
        half = side / 2.0

        sx1 = int(round(cx - half))
        sy1 = int(round(cy - half))
        sx2 = sx1 + int(round(side))
        sy2 = sy1 + int(round(side))

        # Vùng giao với frame
        ix1, iy1 = max(sx1, 0), max(sy1, 0)
        ix2, iy2 = min(sx2, w_frame), min(sy2, h_frame)
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        # Tạo canvas vuông và copy phần ảnh thực vào đúng offset
        crop_size = sx2 - sx1
        face_crop = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)
        face_crop[iy1 - sy1: iy2 - sy1, ix1 - sx1: ix2 - sx1] = frame[iy1:iy2, ix1:ix2]

        blob = self._preprocess(face_crop)
        output = self.session.run(None, {self.input_name: blob})[0][0]  # (265,)

        landmark = self._project_landmark(output)

        # Transform từ canonical space (centered at 0,0) về frame space.
        # Reference Qualcomm: cộng (input_size / 2) rồi scale theo (crop_size / input_size).
        # Vì pipeline preprocess đã resize crop_size -> input_size, nên dùng đúng input_size
        # của model (vd: 224) thay vì hardcode 128 mới đúng tỉ lệ.
        proj = float(self.input_size)
        landmark[:, 0] = (landmark[:, 0] + proj / 2) * crop_size / proj + sx1
        landmark[:, 1] = (landmark[:, 1] + proj / 2) * crop_size / proj + sy1

        return [(int(lm[0].item()), int(lm[1].item())) for lm in landmark]

        
