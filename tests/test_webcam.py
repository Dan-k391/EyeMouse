"""Synthetic checks for the webcam mouse's gaze features and mapping.

Run: .venv/bin/python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_gaze import EYES, FaceMapper, face_features  # noqa: E402
from webcam_mouse import CHUNK, ROLE, FaceSamples, fit_rows, validation_error  # noqa: E402

SCREEN = np.array([1440.0, 900.0])


def landmarks(gaze=(0.0, 0.0), shift=(0.0, 0.0), scale=1.0, roll=0.0):
    """A face in pixels, with eye corners and irises only. gaze is in eye widths."""
    points = np.zeros((478, 2))
    c, s = np.cos(roll), np.sin(roll)
    rotate = np.array([[c, -s], [s, c]])
    for (outer, inner, iris), centre, direction in zip(EYES, ((-60, 0), (60, 0)), (-1, 1)):
        centre = np.array(centre, float)
        points[outer] = centre + (direction*30, 0)
        points[inner] = centre - (direction*30, 0)
        points[iris] = centre + 60*np.array(gaze)
    return (points*scale) @ rotate.T + np.array([640, 360]) + shift


def head(yaw=0.0, pitch=0.0, x=0.0, y=0.0):
    matrix = np.eye(4)
    matrix[:3, 2] = yaw, pitch, 1
    matrix[:2, 3] = x, y
    return matrix


class FeatureTest(unittest.TestCase):
    def test_both_eyes_agree_on_gaze_direction(self):
        self.assertAlmostEqual(face_features(landmarks((.1, 0)), head())[0], .1)
        self.assertAlmostEqual(face_features(landmarks((-.1, 0)), head())[0], -.1)
        self.assertAlmostEqual(face_features(landmarks((0, .05)), head())[1], .05)

    def test_iris_features_ignore_face_position_size_and_roll(self):
        reference = face_features(landmarks((.1, .04)), head())[:2]
        for moved in (landmarks((.1, .04), shift=(120, -60)), landmarks((.1, .04), scale=1.7),
                      landmarks((.1, .04), roll=.3)):
            self.assertTrue(np.allclose(face_features(moved, head())[:2], reference, atol=1e-9))

    def test_collapsed_eye_is_rejected(self):
        self.assertIsNone(face_features(np.zeros((478, 2)), head()))


class MappingTest(unittest.TestCase):
    def scene(self, rng, count=200, noise=0.0):
        """Gaze moves the point on screen; a head turn moves it the opposite way when eyes hold still."""
        features, targets = [], []
        for _ in range(count):
            target = rng.uniform((80, 50), SCREEN - (80, 50))
            yaw, pitch = rng.uniform(-.15, .15), rng.uniform(-.1, .1)
            u, v = (target/SCREEN - .5)
            gaze = (.2*u - .6*yaw, .1*v - .5*pitch)
            row = face_features(landmarks(gaze), head(yaw, pitch))
            features.append(row + rng.normal(0, noise, 6)*[1, 1, 0, 0, 0, 0])
            targets.append(target)
        return np.array(features), np.array(targets)

    def test_turning_the_head_does_not_move_the_cursor(self):
        rng = np.random.default_rng(0)
        mapper = FaceMapper.fit('gaze', *self.scene(rng))
        features, targets = self.scene(rng, 50)
        errors = [np.linalg.norm(mapper.point(f) - t) for f, t in zip(features, targets)]
        self.assertLess(np.mean(errors), 60)

    def test_head_mode_ignores_the_iris(self):
        rng = np.random.default_rng(1)
        features = np.column_stack([rng.normal(0, .1, (100, 2)), rng.uniform(-.2, .2, (100, 2)), np.zeros((100, 2))])
        targets = 720 + features[:, 2:4]*[3000, 2000]
        mapper = FaceMapper.fit('head', features, targets)
        moved = features[0].copy()
        moved[:2] += .3
        self.assertTrue(np.allclose(mapper.point(moved), mapper.point(features[0])))
        self.assertLess(np.linalg.norm(mapper.point(features[0]) - targets[0]), 60)

    def test_too_few_rows(self):
        self.assertIsNone(FaceMapper.fit('gaze', np.zeros((5, 6)), np.zeros((5, 2))))

    def test_save_round_trip(self):
        rng = np.random.default_rng(2)
        mapper = FaceMapper.fit('gaze', *self.scene(rng, 40))
        again = FaceMapper.from_dict(mapper.to_dict())
        row = self.scene(rng, 1)[0][0]
        self.assertTrue(np.allclose(mapper.point(row), again.point(row)))


class SessionTest(unittest.TestCase):
    def test_lost_face_is_not_a_blink(self):
        samples = FaceSamples()
        samples.face_at = 100.0
        self.assertTrue(samples.visible(100.2))
        self.assertFalse(samples.visible(100.5))

    def test_rows_and_validation(self):
        rng = np.random.default_rng(3)
        results = []
        for target in rng.uniform((80, 50), SCREEN - (80, 50), (30, 2)):
            u, v = target/SCREEN - .5
            row = np.array([.2*u, .1*v, 0, 0, 0, 0])
            results.append((target, {ROLE: [(0.0, row + rng.normal(0, .002, 6), 0.0) for _ in range(20)]}))
        rows, targets = fit_rows(results)
        self.assertEqual(len(rows), 30*20//CHUNK)
        mapper = FaceMapper.fit('gaze', rows, targets)
        self.assertLess(validation_error(mapper, results), 60)


if __name__ == '__main__':
    unittest.main()
