import os
import cv2
import numpy as np
import onnxruntime
import torch
from typing import Optional, Tuple, List
import onnx
from onnx.external_data_helper import load_external_data_for_model

class FaceMap3DMMDetector:
    """Qualcomm AI Hub Facial-Landmark-Detection (facemap_3dmm) ONNX wrapper.

    Outputs 68 3D facial landmarks from a 128x128 face crop.
    The ONNX model outputs 265 3DMM parameters which are decoded
    into 2D landmark coordinates using pre-computed basis matrices.
    """

    INPUT_SIZE = 224           # network feeding size (depends on ONNX file)
    PROJECTION_SIZE = 128      # canonical 3DMM space for landmark projection
    VERTEX_NUM = 68
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    NOISE_INDICES = [27,28,29,30]
    NOISE_TRIANGLE_INDICES = [31,33,35]
    EYE_BROW_LEFT = [17,19,21]
    EYE_BROW_RIGHT = [22,24,26]
    OUTER_LIPS_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    INNER_LIPS_INDICES = [60, 61, 62, 63, 64, 65, 66, 67]
    ALPHA_ID_SIZE = 219
    ALPHA_EXP_SIZE = 39
    CHIN_END = 8
    def __init__(
        self,
        model_dir: str = "models/face-lanmark-detection",
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "dms_3dmm.onnx")
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
            self.input_size = self.INPUT_SIZE

        # Load 3DMM basis matrices
        self.mean_face = torch.from_numpy(
            np.load(os.path.join(model_dir, "meanFace.npy")).reshape(
                3 * self.VERTEX_NUM, 1
            )
        )
        self.basis_id = torch.from_numpy(
            np.load(os.path.join(model_dir, "shapeBasis.npy")).reshape(
                3 * self.VERTEX_NUM, self.ALPHA_ID_SIZE
            )
        )
        self.basis_exp = torch.from_numpy(
            np.load(os.path.join(model_dir, "blendShape.npy")).reshape(
                3 * self.VERTEX_NUM, self.ALPHA_EXP_SIZE
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

        alpha_id = out[0:219] * 3
        alpha_exp = out[219:258] * 0.5 + 0.5
        pitch = out[258] * np.pi / 2
        yaw = out[259] * np.pi / 2
        roll = out[260] * np.pi / 2
        tX = out[261] * 60
        tY = out[262] * 60
        tZ = 500
        f = out[263] * 150 + 450

        p_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(-torch.tensor(np.pi)), -torch.sin(-torch.tensor(np.pi))],
                [0, torch.sin(-torch.tensor(np.pi)), torch.cos(-torch.tensor(np.pi))],
            ]
        )

        roll_matrix = torch.tensor(
            [
                [torch.cos(-roll), -torch.sin(-roll), 0],
                [torch.sin(-roll), torch.cos(-roll), 0],
                [0, 0, 1],
            ]
        )

        yaw_matrix = torch.tensor(
            [
                [torch.cos(-yaw), 0, torch.sin(-yaw)],
                [0, 1, 0],
                [-torch.sin(-yaw), 0, torch.cos(-yaw)],
            ]
        )

        pitch_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(-pitch), -torch.sin(-pitch)],
                [0, torch.sin(-pitch), torch.cos(-pitch)],
            ]
        )

        r_matrix = torch.mm(
            yaw_matrix, torch.mm(pitch_matrix, torch.mm(p_matrix, roll_matrix))
        )

        vertices = torch.mm(
            (
                self.mean_face
                + torch.mm(self.basis_id, alpha_id.view(self.ALPHA_ID_SIZE, 1))
                + torch.mm(self.basis_exp, alpha_exp.view(self.ALPHA_EXP_SIZE, 1))
            ).view([self.VERTEX_NUM, 3]),
            r_matrix.transpose(0, 1),
        )

        vertices[:, 0] += tX
        vertices[:, 1] += tY
        vertices[:, 2] += tZ

        f_vec = torch.tensor([f, f]).float()
        return vertices[:, 0:2] * f_vec / tZ

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

        # ------------------------------------------------------------------
        # Mở rộng bbox thành HÌNH VUÔNG quanh tâm để tránh méo aspect ratio
        # khi resize về kích thước cố định của model. Nếu phần mở rộng vượt
        # khỏi biên frame, ta pad bằng pixel đen để giữ tỉ lệ 1:1.
        # ------------------------------------------------------------------
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

        # Transform từ canonical 3DMM space (PROJECTION_SIZE) về frame space.
        # Vì crop đã vuông nên crop_w == crop_h == crop_size, không bị méo.
        proj = self.PROJECTION_SIZE
        landmark[:, 0] = (landmark[:, 0] + proj / 2) * crop_size / proj + sx1
        landmark[:, 1] = (landmark[:, 1] + proj / 2) * crop_size / proj + sy1

        return [(int(lm[0].item()), int(lm[1].item())) for lm in landmark]
    
    def draw_full_mesh(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (255, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        """Draw eye mesh only: left eye + right eye connections.

        Args:
            image: BGR image.
            landmarks: List of (x, y) pixel landmark coordinates.

        Returns:
            Image with eye mesh drawn.
        """
        CYAN = (255, 255, 0)

        # Vẽ mắt trái
        left_pts = np.array([landmarks[i] for i in self.LEFT_EYE_INDICES], dtype=np.int32)
        cv2.polylines(image, [left_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.LEFT_EYE_INDICES:
            x, y = landmarks[idx]
            cv2.circle(image, (x, y), radius,  color, -1, lineType=cv2.LINE_AA)

        # Vẽ mắt phải
        right_pts = np.array([landmarks[i] for i in self.RIGHT_EYE_INDICES], dtype=np.int32)
        cv2.polylines(image, [right_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.RIGHT_EYE_INDICES:
            x, y = landmarks[idx]
            cv2.circle(image, (x, y), radius,  color, -1, lineType=cv2.LINE_AA)
        # Ve mui
        noise_pts = np.array([landmarks[i] for i in self.NOISE_INDICES], dtype=np.int32)
        cv2.polylines(image, [noise_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.NOISE_INDICES:
            x,y = landmarks[idx]
            cv2.circle(image, (x,y), radius, color, -1, lineType=cv2.LINE_AA)

        #Ve Mui Tam Giac 
        # draw bottom noise
        v_shape = np.array([landmarks[31], landmarks[30], landmarks[35]], dtype=np.int32)
        cv2.polylines(image, [v_shape], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        w_shape = np.array([landmarks[31], landmarks[33], landmarks[35]], dtype=np.int32)
        cv2.polylines(image, [w_shape], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.NOISE_TRIANGLE_INDICES:
            x,y = landmarks[idx]
            cv2.circle(image, (x,y), radius, color, -1 , lineType=cv2.LINE_AA)

        # Ve moi ngoai
        outer_lips_pts = np.array([landmarks[i] for i in self.OUTER_LIPS_INDICES], dtype=np.int32)
        cv2.polylines(image, [outer_lips_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.OUTER_LIPS_INDICES:
            x, y = landmarks[idx]
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

        # Ve moi trong
        inner_lips_pts = np.array([landmarks[i] for i in self.INNER_LIPS_INDICES], dtype=np.int32)
        cv2.polylines(image, [inner_lips_pts], isClosed=True, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.INNER_LIPS_INDICES:
            x, y = landmarks[idx]
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        #Ve Long May Trai
        brow_pts = np.array([landmarks[i] for i in self.EYE_BROW_LEFT], dtype=np.int32)
        cv2.polylines(image, [brow_pts], isClosed= False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.EYE_BROW_LEFT:
            x,y = landmarks[idx]
            cv2.circle(image, [x,y], radius, color, -1 ,lineType=cv2.LINE_AA)
            
        #Ve Long May Phai
        brow_pts = np.array([landmarks[i] for i in self.EYE_BROW_RIGHT], dtype=np.int32)
        cv2.polylines(image, [brow_pts], isClosed= False, color=CYAN, thickness=1, lineType=cv2.LINE_AA)
        for idx in self.EYE_BROW_RIGHT:
            x,y = landmarks[idx]
            cv2.circle(image, [x,y], radius, color, -1 ,lineType=cv2.LINE_AA)
        # draw chin
        cv2.circle(image, landmarks[self.CHIN_END], radius, color, -1, lineType=cv2.LINE_AA)
        
