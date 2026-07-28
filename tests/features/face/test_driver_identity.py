import dataclasses
import unittest

import numpy as np

from features.face.identity_stage import DriverIdentityStage
from features.face.recognizer import ArcFaceRecognizer
from features.landmarks.stage import LandmarkStage
from pipeline.context import FrameContext


class StubRecognizer:
    def __init__(self, embeddings: list[np.ndarray]) -> None:
        self._embeddings = iter(embeddings)

    def get_embedding(self, frame, keypoints):
        return next(self._embeddings)

    @staticmethod
    def similarity(first, second):
        return float(np.dot(first, second))


class StubLandmarkDetector:
    def __init__(self) -> None:
        self.called = False

    def detect(self, frame, bbox, keypoints):
        self.called = True
        return np.zeros((68, 2), dtype=np.float32), (0.0, 0.0, 0.0)


class ArcFaceRecognizerTest(unittest.TestCase):
    def test_normalize_embedding_returns_unit_vector(self) -> None:
        normalized = ArcFaceRecognizer.normalize_embedding(
            np.array([3.0, 4.0], dtype=np.float32)
        )

        np.testing.assert_allclose(normalized, np.array([0.6, 0.8]))

    def test_normalize_embedding_rejects_zero_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            ArcFaceRecognizer.normalize_embedding(np.zeros(512, dtype=np.float32))


class DriverIdentityStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)
        self.detections = np.array(
            [
                [10, 20, 80, 100, 0.99],
                [150, 20, 240, 120, 0.98],
            ],
            dtype=np.float32,
        )
        self.keypoints = np.zeros((2, 5, 2), dtype=np.float32)

    def test_selects_matching_driver_instead_of_first_passenger(self) -> None:
        recognizer = StubRecognizer(
            [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ]
        )
        stage = DriverIdentityStage(
            recognizer=recognizer,
            reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            similarity_threshold=0.5,
        )
        context = FrameContext(
            frame=self.frame,
            bbox=(10, 20, 80, 100),
            face_kpss=self.keypoints[0],
            face_detections=self.detections,
            face_keypoints=self.keypoints,
            face_detection_fresh=True,
        )

        result = stage.process(context)

        self.assertTrue(result.is_driver)
        self.assertEqual((150, 20, 240, 120), result.bbox)
        self.assertAlmostEqual(1.0, result.identity_similarity)

    def test_unknown_face_is_kept_for_scrfd_display_but_monitoring_is_disabled(self) -> None:
        recognizer = StubRecognizer(
            [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.2, 0.8], dtype=np.float32),
            ]
        )
        stage = DriverIdentityStage(
            recognizer=recognizer,
            reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            similarity_threshold=0.5,
        )
        context = FrameContext(
            frame=self.frame,
            bbox=(10, 20, 80, 100),
            face_kpss=self.keypoints[0],
            face_detections=self.detections,
            face_keypoints=self.keypoints,
            face_detection_fresh=True,
        )

        result = stage.process(context)

        self.assertFalse(result.is_driver)
        self.assertIsNone(result.bbox)
        self.assertIsNone(result.driver_face_index)
        np.testing.assert_allclose(result.face_similarities, [0.0, 0.2])
        self.assertAlmostEqual(0.2, result.identity_similarity)

    def test_fresh_extreme_frame_is_reidentified_instead_of_reusing_driver(self) -> None:
        recognizer = StubRecognizer(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        )
        stage = DriverIdentityStage(
            recognizer=recognizer,
            reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            similarity_threshold=0.5,
        )
        initial = FrameContext(
            frame=self.frame,
            face_detections=self.detections,
            face_keypoints=self.keypoints,
            face_detection_fresh=True,
        )
        self.assertTrue(stage.process(initial).is_driver)

        fresh_extreme = dataclasses.replace(initial, extreme_pose_mode=True)
        result = stage.process(fresh_extreme)

        self.assertFalse(result.is_driver)
        self.assertIsNone(result.bbox)

    def test_landmark_inference_is_skipped_for_non_driver(self) -> None:
        detector = StubLandmarkDetector()
        stage = LandmarkStage(detector)
        context = FrameContext(
            frame=self.frame,
            bbox=(10, 20, 80, 100),
            face_kpss=self.keypoints[0],
            is_driver=False,
        )

        result = stage.process(context)

        self.assertFalse(detector.called)
        self.assertIsNone(result.landmarks)


if __name__ == "__main__":
    unittest.main()
