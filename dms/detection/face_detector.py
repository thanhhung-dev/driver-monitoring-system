import logging
from typing import Tuple

import cv2
import numpy as np

from utils.onnx_providers import make_session


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetector:
    """
    InsightFace SCRFD ONNX detector (e.g. det_2.5g.onnx).

    Anchor-based, 3 FPN strides (8, 16, 32), 2 anchors per location,
    RGB input normalized with mean=127.5, std=128.
    """

    DEFAULT_INPUT_SHAPE = (480, 640)  # (H, W)

    # SCRFD constants
    _FEAT_STRIDE_FPN = [8, 16, 32]
    _NUM_ANCHORS = 2
    _INPUT_MEAN = 127.5
    _INPUT_STD = 128.0

    def __init__(
        self,
        model_path: str,
        metadata_path: str | None = None,  # kept for backward compatibility (unused)
        conf_thres: float = 0.5,
        iou_thres: float = 0.4,
        input_shape: Tuple[int, int] | None = None,
    ) -> None:
        """Initialize the SCRFD face detector.

        Args:
            model_path:    Path to det_2.5g.onnx (or any SCRFD ONNX).
            metadata_path: Unused (kept for backward compat).
            conf_thres:    Confidence threshold.
            iou_thres:     IoU threshold for NMS.
            input_shape:   (H, W) override; defaults to DEFAULT_INPUT_SHAPE.
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        h, w = input_shape if input_shape is not None else self.DEFAULT_INPUT_SHAPE
        self.input_h = int(h)
        self.input_w = int(w)

        self._initialize_model(model_path)
        self._anchor_cache: dict[tuple, np.ndarray] = {}

    # --------------------------------------------------------------------- #
    # Model loading
    # --------------------------------------------------------------------- #
    def _initialize_model(self, model_path: str) -> None:
        try:
            self.session = make_session(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            # Heuristic: SCRFD has 9 outputs (3 strides × {score, bbox, kps})
            if len(self.output_names) != 9:
                logging.warning(
                    f"Expected 9 outputs for SCRFD, got {len(self.output_names)}. "
                    "Detection may not work correctly."
                )

            logging.info(
                f"Loaded SCRFD model from {model_path} "
                f"(providers={self.session.get_providers()}, input={self.input_h}x{self.input_w})"
            )
        except Exception as e:
            logging.error(f"Failed to load SCRFD model: {e}")
            raise

    # --------------------------------------------------------------------- #
    # Pre-processing (letterbox + normalize)
    # --------------------------------------------------------------------- #
    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.input_h, self.input_w, 3), dtype=image.dtype)
        top = (self.input_h - new_h) // 2
        left = (self.input_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w, :] = resized
        return canvas, scale, top, left

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        letter, scale, top, left = self._letterbox(image)
        rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - self._INPUT_MEAN) / self._INPUT_STD
        blob = np.transpose(blob, (2, 0, 1))[None, ...]  # NCHW
        return blob, scale, top, left

    # --------------------------------------------------------------------- #
    # Anchor centers cache
    # --------------------------------------------------------------------- #
    def _get_anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        ax, ay = np.meshgrid(np.arange(w), np.arange(h))
        centers = np.stack([ax, ay], axis=-1).astype(np.float32) * stride
        centers = centers.reshape(-1, 2)
        if self._NUM_ANCHORS > 1:
            centers = np.repeat(centers, self._NUM_ANCHORS, axis=0)
        self._anchor_cache[key] = centers
        return centers

    # --------------------------------------------------------------------- #
    # NMS
    # --------------------------------------------------------------------- #
    @staticmethod
    def _nms(dets: np.ndarray, iou_thres: float) -> list[int]:
        if dets.shape[0] == 0:
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= iou_thres)[0] + 1]
        return keep

    # --------------------------------------------------------------------- #
    # Detect
    # --------------------------------------------------------------------- #
    def detect(
        self, image: np.ndarray, max_num: int = 0, metric: str = "max"
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        if image is None or image.size == 0:
            return np.empty((0, 5), dtype=np.float32), None

        h_orig, w_orig = image.shape[:2]
        blob, scale, pad_top, pad_left = self._preprocess(image)

        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # SCRFD output ordering (InsightFace convention):
        #   first  N: scores  (per stride)
        #   middle N: bbox    (per stride, in stride units)
        #   last   N: kps     (per stride, in stride units)
        n = len(self._FEAT_STRIDE_FPN)
        scores_list = outputs[0:n]
        bboxes_list = outputs[n:2 * n]
        kpss_list = outputs[2 * n:3 * n]

        all_scores, all_bboxes, all_kpss = [], [], []

        for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
            scores = scores_list[idx]
            bbox_preds = bboxes_list[idx] * stride
            kps_preds = kpss_list[idx] * stride

            # Flatten (1, A*H*W, C) or (A*H*W, C) → (N, C)
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            kps_preds = kps_preds.reshape(-1, 10)

            # Derive feature map size
            feat_h = self.input_h // stride
            feat_w = self.input_w // stride
            anchor_centers = self._get_anchor_centers(feat_h, feat_w, stride)

            # Safety: if shapes mismatch (dynamic), recompute anchors from count
            if anchor_centers.shape[0] != scores.shape[0]:
                # Recompute using actual count
                total = scores.shape[0]
                feat_size = total // self._NUM_ANCHORS
                # assume square-ish feat: use input_h / stride
                feat_h = self.input_h // stride
                feat_w = feat_size // feat_h
                anchor_centers = self._get_anchor_centers(feat_h, feat_w, stride)

            keep = scores >= self.conf_thres
            if not np.any(keep):
                continue

            anchors_k = anchor_centers[keep]
            bboxes = _distance2bbox(anchors_k, bbox_preds[keep])
            kpss = _distance2kps(anchors_k, kps_preds[keep]).reshape(-1, 5, 2)

            all_scores.append(scores[keep])
            all_bboxes.append(bboxes)
            all_kpss.append(kpss)

        if not all_scores:
            return np.empty((0, 5), dtype=np.float32), None

        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        kpss = np.concatenate(all_kpss)

        # Undo letterbox
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_left) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_top) / scale
        kpss[..., 0] = (kpss[..., 0] - pad_left) / scale
        kpss[..., 1] = (kpss[..., 1] - pad_top) / scale

        # Clip
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w_orig - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h_orig - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w_orig - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h_orig - 1)

        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)

        # NMS
        order = scores.argsort()[::-1]
        dets, kpss = dets[order], kpss[order]
        keep = self._nms(dets, self.iou_thres)
        dets, kpss = dets[keep], kpss[keep]

        if 0 < max_num < dets.shape[0]:
            area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            cy, cx = h_orig // 2, w_orig // 2
            offsets = np.vstack([
                (dets[:, 0] + dets[:, 2]) / 2 - cx,
                (dets[:, 1] + dets[:, 3]) / 2 - cy,
            ])
            offset_dist_sq = np.sum(offsets ** 2, axis=0)
            values = area if metric == "max" else (area - offset_dist_sq * 2.0)
            bindex = np.argsort(values)[::-1][:max_num]
            dets, kpss = dets[bindex], kpss[bindex]

        return dets, (kpss if kpss.shape[0] > 0 else None)
