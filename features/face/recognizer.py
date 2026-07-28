"""ArcFace ONNX wrapper for extracting normalized face embeddings."""

import numpy as np

from utils.helpers import face_alignment
from utils.onnx_providers import make_session


class ArcFaceRecognizer:
    """Extract 512-dimensional embeddings from SCRFD-aligned faces."""

    INPUT_SIZE = 112

    def __init__(self, model_path: str) -> None:
        self._session = make_session(model_path)
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._output_name = self._session.get_outputs()[0].name

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("ArcFace embedding contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("ArcFace returned a zero-length embedding")
        return vector / norm

    @staticmethod
    def similarity(first: np.ndarray, second: np.ndarray) -> float:
        first_normalized = ArcFaceRecognizer.normalize_embedding(first)
        second_normalized = ArcFaceRecognizer.normalize_embedding(second)
        return float(np.dot(first_normalized, second_normalized))

    def get_embedding(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Align a face from five SCRFD keypoints and return its embedding."""
        aligned, _ = face_alignment(
            frame,
            np.asarray(keypoints, dtype=np.float32),
            image_size=self.INPUT_SIZE,
        )
        rgb = aligned[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
        output = self._session.run(
            [self._output_name],
            {self._input_name: blob},
        )[0]
        return self.normalize_embedding(output)
