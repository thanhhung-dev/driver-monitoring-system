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

    # ── Foreshortening / validity thresholds ──────────────────────────────
    # Khi đầu quay nghiêng, mắt phía xa camera bị "thu lại" → eye_w giảm
    # mạnh → crop bị méo → gaze output rối loạn.
    # 1) MIN_EYE_WIDTH_PX  : ngưỡng tuyệt đối (px) → quá nhỏ thì bỏ luôn.
    # 2) EYE_WIDTH_RATIO_THRESHOLD : ngưỡng tương đối giữa 2 mắt →
    #    mắt nhỏ hơn nhiều so với mắt còn lại = bị foreshorten → bỏ.
    MIN_EYE_WIDTH_PX = 10.0
    EYE_WIDTH_RATIO_THRESHOLD = 0.65   # tightened from 0.55 — bắt được góc 50–60°

    # Adaptive EMA: khi raw nhảy mạnh so với prev (ví dụ landmark giật) →
    # smoothing aggressive hơn để tránh arrow flicker.
    MAX_GAZE_JUMP_RAD = 0.7    # ~40°
    LOW_ALPHA = 0.15

    # Outlier rejection giữa 2 mắt: nếu 2 mắt cho gaze quá lệch nhau
    # (>~28°) → một mắt chắc chắn sai (foreshortened) → drop mắt có
    # eye_w nhỏ hơn (mắt phía xa camera).
    MAX_EYE_DISAGREEMENT_RAD = 0.5   # ~28°

    def __init__(
        self,
        model_dir: str = "models/eye-gaze-dectecion",
        smooth_alpha: float = 0.5,
        ear_closed_threshold: float = 0.18,
    ) -> None:
        # EMA smoothing: gaze_out = alpha * raw + (1-alpha) * prev.
        # alpha nhỏ => mượt hơn nhưng trễ; 0.3–0.5 là vùng cân bằng tốt.
        self.smooth_alpha = float(smooth_alpha)
        # EAR (Eye Aspect Ratio) threshold để xác định mắt nhắm.
        # Khi EAR < ngưỡng này → coi như nhắm mắt → gaze được set về [0, 0]
        # (nhìn thẳng vào tâm mắt) thay vì đoán bừa từ crop bị méo.
        self.ear_closed_threshold = float(ear_closed_threshold)
        self._prev_gaze_l: np.ndarray | None = None
        self._prev_gaze_r: np.ndarray | None = None
        # Alternate eyes mỗi frame để giảm 2 ONNX call → 1 call/frame.
        # Mỗi mắt vẫn refresh ~mỗi 2 frame (≈30 Hz @ 60 FPS source);
        # smoothing EMA che được sự "lệch" 1 frame giữa 2 mắt.
        self._alternate_eyes = True
        self._eye_turn = 0   # 0 = trái, 1 = phải
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

        # Model EyeGaze có nhánh heatmap [1,3,34,48,80] (~391k voxels) khá nặng,
        # nên DirectML/CUDA vẫn nhanh hơn CPU (~20 vs ~26 ms/inference đo thực tế).
        # Dùng provider tốt nhất (CUDA → DirectML → CoreML → CPU).
        self.session = make_session(model_bytes)

        self.input_name   = self.session.get_inputs()[0].name
        # Model có 3 output: [heatmaps (391k voxels), landmarks, gaze_pitchyaw].
        # Pipeline chỉ dùng gaze → chỉ yêu cầu đúng output đó để ORT có thể
        # prune nhánh heatmap (op nặng nhất) khi chạy → nhanh hơn đáng kể.
        all_outputs = [o.name for o in self.session.get_outputs()]
        gaze_name = next((n for n in all_outputs if "gaze" in n.lower()), all_outputs[-1])
        self.output_names = [gaze_name]

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
        return tensor              

    # ── EAR (eye-closed detection) ────────────────────────────────────────

    @staticmethod
    def _eye_aspect_ratio(landmarks: np.ndarray, indices: list[int]) -> float | None:
        """
        Tính Eye Aspect Ratio (Soukupová & Čech, 2016) từ 6 điểm mắt theo
        convention: [outer, top1, top2, inner, bottom2, bottom1].

        EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)

        Mắt mở: EAR ≈ 0.25–0.35.
        Mắt nhắm: EAR < ~0.18.

        Trả về None nếu không đủ điểm (vd. landmarks 5 điểm).
        """
        if len(indices) < 6:
            return None
        p = landmarks[indices, :2].astype(np.float32)
        v1 = float(np.linalg.norm(p[1] - p[5]))
        v2 = float(np.linalg.norm(p[2] - p[4]))
        h  = float(np.linalg.norm(p[0] - p[3]))
        if h < 1e-3:
            return None
        return (v1 + v2) / (2.0 * h)

    # ── Eye width (foreshortening detection) ──────────────────────────────

    @staticmethod
    def _eye_width(landmarks: np.ndarray, indices: list[int]) -> float:
        """
        Khoảng cách 2 corner ngang của mắt (px). Dùng để phát hiện
        foreshortening: mắt bị thu lại khi head quay nghiêng.
        Convention 6-point: indices[0] = outer corner, indices[3] = inner.
        """
        if len(indices) >= 4:
            pts = landmarks[indices, :2].astype(np.float32)
            return float(np.linalg.norm(pts[0] - pts[3]))
        if len(indices) >= 2:
            pts = landmarks[indices, :2].astype(np.float32)
            return float(np.linalg.norm(pts[0] - pts[1]))
        return 0.0

    # ── Adaptive EMA ──────────────────────────────────────────────────────

    def _adaptive_alpha(self, raw: np.ndarray, prev: np.ndarray | None) -> float:
        """Hạ alpha (smoothing mạnh hơn) khi raw nhảy đột ngột so với prev."""
        if prev is None:
            return self.smooth_alpha
        diff = float(np.linalg.norm(raw - prev))
        if diff > self.MAX_GAZE_JUMP_RAD:
            return self.LOW_ALPHA
        return self.smooth_alpha

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
        # Model expects rank-3 input: (1, H, W) — CHW without batch dim.
        batch = tensor[np.newaxis, :, :]
        outs = self.session.run(self.output_names, {self.input_name: batch})
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

        # ── Validity gating dựa trên eye width ────────────────────────────
        # Khi head quay nghiêng, mắt phía xa camera bị foreshortened →
        # eye_w giảm mạnh. Nếu nhỏ hơn ngưỡng tuyệt đối hoặc nhỏ hơn nhiều
        # so với mắt còn lại thì BỎ mắt đó (return None, clear prev cache).
        ew_l = self._eye_width(landmarks, left_idx)
        ew_r = self._eye_width(landmarks, right_idx)
        max_ew = max(ew_l, ew_r, 1.0)
        valid_l = (ew_l >= self.MIN_EYE_WIDTH_PX
                   and (ew_l / max_ew) >= self.EYE_WIDTH_RATIO_THRESHOLD)
        valid_r = (ew_r >= self.MIN_EYE_WIDTH_PX
                   and (ew_r / max_ew) >= self.EYE_WIDTH_RATIO_THRESHOLD)

        # Quyết định frame này chạy mắt nào (luân phiên để giảm overhead ONNX).
        # Nếu alternate_eyes=False: chạy cả 2 mắt như cũ.
        do_left  = (not self._alternate_eyes) or self._eye_turn == 0
        do_right = (not self._alternate_eyes) or self._eye_turn == 1
        if self._alternate_eyes:
            self._eye_turn ^= 1

        # ── Mắt trái ──────────────────────────────────────────────────────
        crop_l, center_l = self._crop(frame, landmarks, left_idx)
        if not valid_l:
            # Foreshortened → drop hoàn toàn + clear cache (tránh smoothing
            # qua các sự kiện gián đoạn khi head xoay nhanh).
            self._prev_gaze_l = None
            gaze_l = None
        elif do_left:
            ear_l = self._eye_aspect_ratio(landmarks, left_idx)
            if ear_l is not None and ear_l < self.ear_closed_threshold:
                # Nhắm mắt → set gaze về tâm mắt (pitch=yaw=0, vector hướng thẳng).
                # Bỏ qua inference cho hợp lý + giữ smoothing đồng bộ.
                gaze_l = np.zeros(2, dtype=np.float32)
                self._prev_gaze_l = gaze_l
            elif crop_l is not None:
                raw_l = self._infer(self._preprocess(crop_l, flip=False))
                alpha = self._adaptive_alpha(raw_l, self._prev_gaze_l)
                if self._prev_gaze_l is None:
                    gaze_l = raw_l
                else:
                    gaze_l = alpha * raw_l + (1.0 - alpha) * self._prev_gaze_l
                self._prev_gaze_l = gaze_l
        else:
            gaze_l = self._prev_gaze_l   # dùng giá trị cache frame trước

        # ── Mắt phải ──────────────────────────────────────────────────────
        crop_r, center_r = self._crop(frame, landmarks, right_idx)
        if not valid_r:
            self._prev_gaze_r = None
            gaze_r = None
        elif do_right:
            ear_r = self._eye_aspect_ratio(landmarks, right_idx)
            if ear_r is not None and ear_r < self.ear_closed_threshold:
                gaze_r = np.zeros(2, dtype=np.float32)
                self._prev_gaze_r = gaze_r
            elif crop_r is not None:
                raw_r = self._infer(self._preprocess(crop_r, flip=True))
                raw_r[1] = -raw_r[1]
                alpha = self._adaptive_alpha(raw_r, self._prev_gaze_r)
                if self._prev_gaze_r is None:
                    gaze_r = raw_r
                else:
                    gaze_r = alpha * raw_r + (1.0 - alpha) * self._prev_gaze_r
                self._prev_gaze_r = gaze_r
        else:
            gaze_r = self._prev_gaze_r   # dùng giá trị cache frame trước

        # ── Outlier rejection giữa 2 mắt ──────────────────────────────────
        # Nếu cả 2 mắt còn sống nhưng cho gaze mâu thuẫn nặng → drop mắt
        # có eye_w nhỏ hơn (mắt bị foreshortened, kém tin cậy).
        if gaze_l is not None and gaze_r is not None:
            disagree = float(np.linalg.norm(gaze_l - gaze_r))
            if disagree > self.MAX_EYE_DISAGREEMENT_RAD:
                if ew_l < ew_r:
                    gaze_l = None
                    self._prev_gaze_l = None
                else:
                    gaze_r = None
                    self._prev_gaze_r = None

        return gaze_l, gaze_r, center_l, center_r