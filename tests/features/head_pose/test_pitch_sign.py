import unittest
from unittest.mock import patch

import numpy as np

from features.head_pose.stage import _rotation_matrix_to_head_pose
from utils.general import get_rotation_matrix
from utils.helpers import draw_head_direction_arrow


class PitchSignTest(unittest.TestCase):
    def test_head_pose_pitch_is_positive_when_looking_up(self):
        rotation = get_rotation_matrix(np.deg2rad(20.0), 0.0, 0.0)

        _, pitch, _ = _rotation_matrix_to_head_pose(rotation)

        self.assertAlmostEqual(pitch, 20.0)

    def test_head_pose_pitch_is_negative_when_looking_down(self):
        rotation = get_rotation_matrix(np.deg2rad(-20.0), 0.0, 0.0)

        _, pitch, _ = _rotation_matrix_to_head_pose(rotation)

        self.assertAlmostEqual(pitch, -20.0)

    def test_yaw_and_roll_are_unchanged(self):
        rotation = get_rotation_matrix(
            np.deg2rad(20.0), np.deg2rad(-15.0), np.deg2rad(10.0),
        )

        yaw, _, roll = _rotation_matrix_to_head_pose(rotation)

        self.assertAlmostEqual(yaw, -15.0)
        self.assertAlmostEqual(roll, 10.0)

    @patch("utils.helpers.cv2.arrowedLine")
    def test_positive_pitch_draws_head_direction_up(self, arrowed_line):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = np.full((68, 2), 50.0, dtype=np.float32)

        draw_head_direction_arrow(image, landmarks, yaw=0.0, pitch=20.0)

        start = arrowed_line.call_args.args[1]
        end = arrowed_line.call_args.args[2]
        self.assertLess(end[1], start[1])


if __name__ == "__main__":
    unittest.main()
