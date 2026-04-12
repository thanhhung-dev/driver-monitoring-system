import os
import cv2
import numpy as np
import onnxruntime
from typing import Dict, Optional, Tuple


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
    from a 128x128 face crop.
    """

    INPUT_SIZE = 128

    def __init__(
        self,
        model_dir: str = "models/Facial-Attribute-Detection",
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        import onnx
        from onnx.external_data_helper import load_external_data_for_model

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
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize to 128x128, convert BGR->RGB, normalize to [0,1], NCHW."""
        resized = cv2.resize(
            face_crop,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 128, 128)

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
