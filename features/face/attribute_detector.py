import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import onnx
from onnx.external_data_helper import load_external_data_for_model

from utils.onnx_providers import make_session

ATTRIB_NAMES = [
    "left_eye_open",
    "right_eye_open",
    "glasses",
    "mask",
    "sunglasses",
]

class FaceAttribDetector:
    """Qualcomm AI Hub Facial-Attribute-Detection ONNX wrapper.

    Detects facial attributes (eye openness, glasses, mask, sunglasses)
    from a 224x224 face crop.
    """

    INPUT_SIZE = 224

    def __init__(
        self,
        model_dir: str = "models/Facial-Attribute-Detection",
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        onnx_model = onnx.load(onnx_path, load_external_data=False)

        # The ONNX file may reference an external data filename
        # (e.g. "dms_3dmm.onnx.data") that differs from the actual file
        # on disk ("model.data"). Rewrite the "location" entry of every
        # tensor's external_data so it points to the real file.
        data_candidates = ["model.data", "model.onnx.data"]
        actual_data_file = next(
            (name for name in data_candidates
             if os.path.exists(os.path.join(model_dir, name))),
            None,
        )
        if actual_data_file is not None:
            for tensor in onnx_model.graph.initializer:
                if tensor.HasField("data_location") and tensor.data_location == tensor.EXTERNAL:
                    for entry in tensor.external_data:
                        if entry.key == "location":
                            entry.value = actual_data_file

        load_external_data_for_model(onnx_model, model_dir)
        model_bytes = onnx_model.SerializeToString()

        self.session = make_session(model_bytes)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        # Auto-detect spatial input size from the model (NCHW: [N, C, H, W]).
        shape = input_meta.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = int(shape[2])
        else:
            self.input_size = self.INPUT_SIZE

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize to model input size, convert BGR->RGB, normalize to [0,1], NCHW."""
        resized = cv2.resize(
            face_crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    def detect(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Dict[str, float]]:
        """Detect facial attributes given a frame and face bounding box.

        Args:
            frame: Full BGR image.
            bbox: Face bounding box as (x1, y1, x2, y2).

        Returns:
            Dict mapping attribute name to probability [0, 1],
            or None if the crop is invalid.
        """
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        blob = self._preprocess(face_crop)
        output = self.session.run(None, {self.input_name: blob})[0]  # (1, 5)
        probs = output[0]

        return {name: float(probs[i]) for i, name in enumerate(ATTRIB_NAMES)}
