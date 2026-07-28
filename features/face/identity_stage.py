"""Select the enrolled driver and gate driver-only monitoring stages."""

import dataclasses

import numpy as np

from features.face.recognizer import ArcFaceRecognizer
from pipeline.context import FrameContext


class DriverIdentityStage:
    """Compare every SCRFD face with the enrolled driver's embedding."""

    def __init__(
        self,
        recognizer: ArcFaceRecognizer,
        driver_embeddings: dict[str, np.ndarray],
        similarity_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        driver_roi: list[float] | None = None,
    ) -> None:
        self._recognizer = recognizer
        
        self._driver_embeddings = {}
        for name, emb in driver_embeddings.items():
            if emb.ndim == 1:
                self._driver_embeddings[name] = [ArcFaceRecognizer.normalize_embedding(emb)]
            else:
                self._driver_embeddings[name] = [ArcFaceRecognizer.normalize_embedding(e) for e in emb]
            
        self._threshold = float(similarity_threshold)
        self._iou_threshold = float(iou_threshold)
        self._driver_roi = driver_roi
        self._last_index: int | None = None
        self._last_is_driver = False
        self._last_driver_name: str | None = None
        self._last_similarity: float | None = None
        self._last_similarities: np.ndarray | None = None
        self._last_driver_bbox: tuple[int, int, int, int] | None = None

    @property
    def name(self) -> str:
        return "driver_identity"

    @staticmethod
    def _selection(ctx: FrameContext, index: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
        detection = ctx.face_detections[index]
        bbox = tuple(int(value) for value in detection[:4])
        return bbox, ctx.face_keypoints[index]
        
    @staticmethod
    def _bbox_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _is_in_roi(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> bool:
        if not self._driver_roi:
            return True
        h, w = frame_shape[:2]
        x_min, y_min, x_max, y_max = self._driver_roi
        roi_x1, roi_y1 = w * x_min, h * y_min
        roi_x2, roi_y2 = w * x_max, h * y_max
        
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        
        return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2

    def process(self, ctx: FrameContext) -> FrameContext:
        detections = ctx.face_detections
        keypoints = ctx.face_keypoints
        has_faces = (
            detections is not None
            and keypoints is not None
            and len(detections) > 0
            and len(keypoints) == len(detections)
        )
        if not has_faces:
            self._last_index = None
            self._last_is_driver = False
            self._last_driver_name = None
            self._last_similarity = None
            return dataclasses.replace(
                ctx,
                bbox=None,
                face_kpss=None,
                is_driver=False,
                driver_name=None,
                out_of_position=self._last_is_driver, # If we had a driver but now no faces, they are OOP
                driver_roi=self._driver_roi,
                identity_similarity=None,
                face_similarities=None,
                driver_face_index=None,
            )

        if ctx.face_detection_fresh or self._last_similarities is None:
            similarities = []
            max_iou = 0.0
            best_iou_index = -1
            best_driver_names = []
            for i, face_keypoints in enumerate(keypoints):
                bbox = tuple(int(value) for value in detections[i][:4])
                
                # Check if face is in ROI before computing ArcFace
                if not self._is_in_roi(bbox, ctx.frame.shape):
                    similarities.append(0.0)
                    best_driver_names.append(None)
                    continue
                    
                embedding = self._recognizer.get_embedding(ctx.frame, face_keypoints)
                # Compute max similarity across all reference embeddings of all drivers
                best_sim = -1.0
                best_name = None
                
                for name, refs in self._driver_embeddings.items():
                    sims = [self._recognizer.similarity(embedding, ref) for ref in refs]
                    sim = max(sims)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                        
                similarities.append(best_sim)
                best_driver_names.append(best_name)
                # Check spatial tracking (IoU) if we had a driver in the previous frame
                if self._last_driver_bbox is not None:
                    iou = self._bbox_iou(bbox, self._last_driver_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_iou_index = i
                        
            best_sim_index = int(np.argmax(similarities))
            best_sim = float(similarities[best_sim_index])
            best_sim_name = best_driver_names[best_sim_index]
            
            was_driver = self._last_is_driver
            
            # Decision logic: Identity first, then Spatial Tracking fallback
            if best_sim >= self._threshold:
                best_index = best_sim_index
                self._last_is_driver = True
                self._last_driver_name = best_sim_name
            elif max_iou >= self._iou_threshold:
                best_index = best_iou_index
                self._last_is_driver = True
                # Keep the same driver name from previous frame due to spatial tracking
            else:
                best_index = best_sim_index
                self._last_is_driver = False
                self._last_driver_name = None
            self._last_similarities = np.asarray(similarities, dtype=np.float32)
            self._last_similarity = float(similarities[best_index])
            self._last_index = best_index if self._last_is_driver else None
            
            # OOP is when we had a driver, but now we don't (lost them or they moved out of ROI)
            out_of_position = was_driver and not self._last_is_driver
        else:
            # If not fresh detection, maintain the OOP state based on whether we currently have a driver
            out_of_position = not self._last_is_driver if self._last_index is None else False

        if self._last_index is None or self._last_index >= len(detections):
            return dataclasses.replace(
                ctx,
                bbox=None,
                face_kpss=None,
                is_driver=False,
                driver_name=None,
                out_of_position=out_of_position,
                driver_roi=self._driver_roi,
                identity_similarity=self._last_similarity,
                face_similarities=self._last_similarities,
                driver_face_index=None,
            )

        bbox, face_keypoints = self._selection(ctx, self._last_index)
        
        # Update spatial tracking bbox for the next frame
        if self._last_is_driver:
            self._last_driver_bbox = bbox
        else:
            self._last_driver_bbox = None
            
        return dataclasses.replace(
            ctx,
            bbox=bbox,
            face_kpss=face_keypoints,
            is_driver=self._last_is_driver,
            driver_name=self._last_driver_name,
            out_of_position=out_of_position,
            driver_roi=self._driver_roi,
            identity_similarity=self._last_similarity,
            face_similarities=self._last_similarities,
            driver_face_index=self._last_index,
        )
