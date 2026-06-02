import cv2
import numpy as np

from core.frame_context import FrameContext
from utils.helpers import expand_bbox
from utils.onnx_providers import make_session

import onnx


class HeadPoseStage:
    """Head pose estimation using SixDRepNet ONNX with interval-based skipping.

    Input  : BGR crop 224×224, float32 NCHW, ImageNet normalized
    Output : 3×3 rotation matrix → Euler angles (yaw, pitch, roll) degrees
    """

    # ImageNet normalization
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_dir: str, interval: int = 5) -> None:
        self._interval = interval
        self._counter = 0
        self._last_head_pose = None

        # Load ONNX model with external data (resolves paths relative to onnx file)
        onnx_path = f"{model_dir}/pose_estimator.onnx"
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        model_bytes = onnx_model.SerializeToString()
        self._session = make_session(model_bytes)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "head_pose"

    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """BGR crop → ImageNet-normalized NCHW float32."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        normalized = (resized - self._MEAN) / self._STD
        return normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1,3,224,224)

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.bbox is None:
            return ctx

        self._counter += 1
        if self._counter < self._interval:
            if self._last_head_pose is not None:
                return FrameContext(
                    frame=ctx.frame,
                    frame_number=ctx.frame_number,
                    bbox=ctx.bbox,
                    face_kpss=ctx.face_kpss,
                    landmarks=ctx.landmarks,
                    facemap_pose=ctx.facemap_pose,
                    head_pose=self._last_head_pose,
                )
            return ctx

        self._counter = 0
        x1, y1, x2, y2 = ctx.bbox
        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2)

        h, w = ctx.frame.shape[:2]
        ex1 = max(0, ex1)
        ey1 = max(0, ey1)
        ex2 = min(ex2, w)
        ey2 = min(ey2, h)

        head_crop = ctx.frame[ey1:ey2, ex1:ex2]
        if head_crop.size == 0:
            return ctx

        tensor = self._preprocess(head_crop)
        outs = self._session.run(None, {self._input_name: tensor})
        R = np.asarray(outs[0], dtype=np.float64).reshape(3, 3)

        # Euler angles — negate pitch & yaw to match Qualcomm convention:
        #   pitch + = up, yaw + = right, roll + = clockwise
        # Uses clamp for numerical stability (avoids NaN from sqrt of tiny negatives)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2).clip(min=1e-12)
        pitch_d = -float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw_d   = -float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll_d  = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

        # Clamp to plausible human head range ±90°
        if abs(pitch_d) > 90 or abs(yaw_d) > 90:
            return FrameContext(
                frame=ctx.frame,
                frame_number=ctx.frame_number,
                bbox=ctx.bbox,
                face_kpss=ctx.face_kpss,
                landmarks=ctx.landmarks,
                facemap_pose=ctx.facemap_pose,
                head_pose=self._last_head_pose,
            )

        head_pose_angles = (yaw_d, pitch_d, roll_d)
        self._last_head_pose = head_pose_angles

        return FrameContext(
            frame=ctx.frame,
            frame_number=ctx.frame_number,
            bbox=ctx.bbox,
            face_kpss=ctx.face_kpss,
            landmarks=ctx.landmarks,
            facemap_pose=ctx.facemap_pose,
            head_pose=head_pose_angles,
        )
