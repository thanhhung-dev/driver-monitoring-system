import unittest

import numpy as np

from features.gaze.stage import GazeStage
from pipeline.context import FrameContext
from presentation.opencv.visualizer import Visualizer


class StubEyeGaze:
    def __init__(self) -> None:
        self.eye_distance = 20.0
        self._prev_gaze_l = None
        self._prev_gaze_r = None

    def detect(self, frame, landmarks):
        center_x = 100.0
        half_distance = self.eye_distance / 2.0
        gaze = np.zeros(2, dtype=np.float32)
        return (
            gaze.copy(),
            gaze.copy(),
            np.array([center_x - half_distance, 100.0], dtype=np.float32),
            np.array([center_x + half_distance, 100.0], dtype=np.float32),
        )


class GazeStageTest(unittest.TestCase):
    def test_adjust_gaze_converts_model_pitch_to_app_vertical_direction(self) -> None:
        stage = GazeStage(eye_gaze=None, visualizer=None)

        adjusted = stage._adjust_gaze(np.array([-0.3, 0.0], dtype=np.float32))
        vector = stage._pitchyaw_to_vec(adjusted)

        self.assertGreater(adjusted[0], 0.0)
        self.assertLess(vector[1], 0.0)

    def test_upward_eye_gaze_is_drawn_above_the_eye(self) -> None:
        stage = GazeStage(eye_gaze=None, visualizer=None)
        visualizer = Visualizer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        eye_position = np.array([100.0, 100.0], dtype=np.float32)
        adjusted = stage._adjust_gaze(np.array([-0.3, 0.0], dtype=np.float32))
        vector = stage._pitchyaw_to_vec(adjusted)

        visualizer.draw_gaze_3d(
            image,
            eye_position,
            vector,
            length=70,
            focal_length=200,
        )

        painted_y, _ = np.nonzero(np.any(image != 0, axis=2))
        self.assertGreater(len(painted_y), 0)
        self.assertLess(float(np.mean(painted_y)), eye_position[1])

    def test_gaze_length_is_fixed_when_face_distance_changes(self) -> None:
        eye_gaze = StubEyeGaze()
        stage = GazeStage(eye_gaze=eye_gaze, visualizer=None)
        stage._eye_open_length_scale = lambda landmarks: 1.0
        context = FrameContext(
            frame=np.zeros((200, 200, 3), dtype=np.uint8),
            landmarks=np.zeros((68, 2), dtype=np.float32),
        )

        far_result = stage.process(context)
        eye_gaze.eye_distance = 100.0
        near_result = stage.process(context)

        self.assertEqual(
            far_result.gaze_render_data["length"],
            near_result.gaze_render_data["length"],
        )
        self.assertEqual(GazeStage.EYE_GAZE_LENGTH, far_result.gaze_render_data["length"])

    def test_head_fallback_starts_at_nose_and_matches_head_arrow_length(self) -> None:
        stage = GazeStage(eye_gaze=StubEyeGaze(), visualizer=None)
        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[30] = np.array([90.0, 75.0], dtype=np.float32)
        context = FrameContext(
            frame=np.zeros((200, 200, 3), dtype=np.uint8),
            landmarks=landmarks,
            head_pose=(80.0, 0.0, 0.0),
        )

        result = stage.process(context)

        self.assertTrue(result.gaze_render_data["fallback"])
        np.testing.assert_array_equal(
            result.gaze_render_data["center_l"], landmarks[30]
        )
        self.assertIsNone(result.gaze_render_data["center_r"])
        self.assertEqual(40.0, result.gaze_render_data["length"])


if __name__ == "__main__":
    unittest.main()
