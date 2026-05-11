import os
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx.external_data_helper import load_external_data_for_model
from utils.onnx_providers import make_session

# ── Landmark index (auto-detect format) ───────────────────────────────────
# Trả về (left_eye_idx, right_eye_idx) theo perspective của SUBJECT (người).
#   - dlib 68-point:  36–41 = subject's RIGHT eye, 42–47 = subject's LEFT eye
#   - MediaPipe 468:  33/133/… nằm bên trái ảnh = subject's RIGHT eye;
#                     362/263/… nằm bên phải ảnh = subject's LEFT eye
# Quy ước trên đảm bảo flip ngang chỉ áp dụng cho mắt PHẢI (subject) —
# đúng như input mà model EyeGaze (Qualcomm) yêu cầu (trained trên left eye).
_IDX = {
    468: ([362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144]),
    68:  ([42, 43, 44, 45, 46, 47],       [36, 37, 38, 39, 40, 41]),
    5:   ([1],                             [0]),
}

def _eye_indices(n: int):
    for threshold in (468, 68, 5):
        if n >= threshold:
            return _IDX[threshold]
    return _IDX[5]


class EyeGazeEstimation:
    """
    Qualcomm AI Hub — EyeGaze (EyeNet) ONNX wrapper.

    Input  : grayscale eye crop, 96 × 160 (H × W), float32 [0, 1], NCHW (1,1,96,160)
    Output : [pitch, yaw] in radians, shape (2,)
    """

    # Qualcomm spec: H=96, W=160
    INPUT_H = 96
    INPUT_W = 160

    def __init__(
        self,
        model_dir: str = "models/eye-gaze-dectecion",
        smooth_alpha: float = 0.35,
    ) -> None:
        # EMA smoothing: gaze_out = alpha * raw + (1-alpha) * prev.
        # alpha nhỏ => mượt hơn nhưng trễ; 0.3–0.5 là vùng cân bằng tốt.
        self.smooth_alpha = float(smooth_alpha)
        self._prev_gaze_l: np.ndarray | None = None
        self._prev_gaze_r: np.ndarray | None = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "eyegaze.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX Model not found:  {onnx_path}")
        onnx_model = onnx.load(onnx_path, load_external_data=False)

        data_candidates = ["eyegaze.data"]
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

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = onnxruntime.InferenceSession(
            model_bytes,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name   = self.session.get_inputs()[0].name
        # Model có 3 output: [heatmaps, landmarks, gaze_pitchyaw]
        # Lấy hết tên output để run trả về đầy đủ; gaze nằm ở index [2].
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ── Qualcomm preprocess ───────────────────────────────────────────────

    @classmethod
    def _preprocess(cls, bgr_crop: np.ndarray, flip: bool = False) -> np.ndarray:
        """
        Qualcomm EyeGaze preprocess (mirror chính xác app.py upstream):
          BGR → grayscale → resize (W=160, H=96) → equalizeHist
          → [0,1] float32 → fliplr nếu mắt phải → (1, 96, 160) CHW (rank-3, không batch).
        """
        if bgr_crop.ndim == 3:
            gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_crop
        gray = cv2.resize(gray, (cls.INPUT_W, cls.INPUT_H))
        gray = cv2.equalizeHist(gray)
        tensor = gray.astype(np.float32) / 255.0           # [0, 1]
        if flip:
            tensor = np.fliplr(tensor).copy()
        return tensor[np.newaxis, :, :]                    # (1, 96, 160)

    # ── Crop helper ───────────────────────────────────────────────────────

    @staticmethod
    def _crop(
        frame: np.ndarray,
        landmarks: np.ndarray,
        indices: list[int],
        scale: float = 2.2,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Crop vùng mắt với tỉ lệ 5:3 (= INPUT_W:INPUT_H = 160:96) để khớp đúng
        format input của EyeGaze model. Mở rộng đủ rộng để bao gồm mí mắt + vùng
        xung quanh — model cần ngữ cảnh này để xác định iris position so với eye
        boundary (thiếu ngữ cảnh -> output không nhạy với eye motion -> arrow
        nhìn như đang follow head).

        scale: hệ số nhân kích thước bbox eye-corner để có crop rộng hơn.
        """
        h, w = frame.shape[:2]
        pts = landmarks[indices, :2].astype(np.float32)
        center = pts.mean(axis=0)

        # Lấy khoảng cách 2 corner ngang (eye width) làm chuẩn.
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        eye_w = max(float(x_max - x_min), 4.0)

        # Crop width = scale * eye_w; height giữ aspect 5:3 ⇒ h = w * 96/160.
        crop_w = max(eye_w * scale, 24.0)
        crop_h = crop_w * (EyeGazeEstimation.INPUT_H / EyeGazeEstimation.INPUT_W)

        cx, cy = float(center[0]), float(center[1])
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x2 = int(round(cx + crop_w / 2.0))
        y2 = int(round(cy + crop_h / 2.0))

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if x2 - x1 < 8 or y2 - y1 < 6:
            return None, center
        return frame[y1:y2, x1:x2], center

    # ── Inference ─────────────────────────────────────────────────────────

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        Chạy ONNX session, trả về [pitch, yaw] radians shape (2,).
        Model có 3 outputs: [heatmaps, landmarks, gaze_pitchyaw] — lấy index cuối.
        """
        outs = self.session.run(self.output_names, {self.input_name: tensor})
        gaze = np.asarray(outs[-1], dtype=np.float32).reshape(-1)
        return gaze[:2]   # [pitch, yaw]

    # ── Public API ────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,          # (N, 2|3) pixel coords
    ) -> tuple[
        np.ndarray | None,              # gaze_left  [pitch, yaw] rad
        np.ndarray | None,              # gaze_right [pitch, yaw] rad
        np.ndarray | None,              # eye_center_left  (2,) px
        np.ndarray | None,              # eye_center_right (2,) px
    ]:
        if landmarks is None or len(landmarks) == 0:
            return None, None, None, None

        landmarks = np.asarray(landmarks, dtype=np.float32)
        left_idx, right_idx = _eye_indices(len(landmarks))
        gaze_l = gaze_r = center_l = center_r = None
        a = self.smooth_alpha

        # ── Mắt trái ──────────────────────────────────────────────────────
        crop_l, center_l = self._crop(frame, landmarks, left_idx)
        if crop_l is not None:
            raw_l = self._infer(self._preprocess(crop_l, flip=False))
            if self._prev_gaze_l is None:
                gaze_l = raw_l
            else:
                gaze_l = a * raw_l + (1.0 - a) * self._prev_gaze_l
            self._prev_gaze_l = gaze_l

        # ── Mắt phải — flip về left-eye space trước khi inference ─────────
        crop_r, center_r = self._crop(frame, landmarks, right_idx)
        if crop_r is not None:
            raw_r = self._infer(self._preprocess(crop_r, flip=True))
            # Sau inference: negate yaw để convert ngược lại về world coords
            # (flip ngang ↔ yaw đổi dấu, pitch giữ nguyên)
            raw_r = raw_r * np.array([1.0, -1.0], dtype=np.float32)
            if self._prev_gaze_r is None:
                gaze_r = raw_r
            else:
                gaze_r = a * raw_r + (1.0 - a) * self._prev_gaze_r
            self._prev_gaze_r = gaze_r

        return gaze_l, gaze_r, center_l, center_r