"""Synthetic checks for the eye mouse: mapping, smoothing and blink clicks.

Run: .venv/bin/python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaze_mapper import BlinkClicker, EyeMapper, GazeMapper, OneEuroFilter, robust_centre  # noqa: E402

SCREEN = np.array([1440.0, 900.0])
PUPIL_SIZE = 40.0


def pupil_for(screen_point, flip=1.0):
    """A plausible nonlinear eye: pupil pixels for a gaze point on screen."""
    u, v = (np.asarray(screen_point, float)/SCREEN - .5)*2
    return np.array([160 + flip*(55*u + 6*u*u + 4*u*v), 120 + 38*v + 5*v*v - 3*u*u])


def grid():
    return [SCREEN*(.06 + .88*np.array([x, y])) for y in (0, .5, 1) for x in (0, .5, 1)] + \
           [SCREEN*(.06 + .88*np.array([x, y])) for y in (.25, .75) for x in (.25, .75)]


class MappingTest(unittest.TestCase):
    def mapper(self, flip=1.0):
        targets = grid()
        return EyeMapper.fit([pupil_for(t, flip) for t in targets], targets, PUPIL_SIZE)

    def test_recovers_points_between_targets(self):
        mapper = self.mapper()
        rng = np.random.default_rng(0)
        for point in rng.uniform([100, 80], [1340, 820], (50, 2)):
            self.assertLess(np.linalg.norm(mapper(pupil_for(point)) - point), 15)

    def test_mirrored_eye_works(self):  # the left feed is rotated 180 degrees
        mapper = self.mapper(-1.0)
        self.assertLess(np.linalg.norm(mapper(pupil_for((400, 300), -1.0)) - (400, 300)), 15)

    def test_rejects_too_few_targets_or_a_still_pupil(self):
        self.assertIsNone(EyeMapper.fit([pupil_for(t) for t in grid()[:5]], grid()[:5], PUPIL_SIZE))
        self.assertIsNone(EyeMapper.fit([(160, 120)]*13, grid(), PUPIL_SIZE))

    def test_plausibility_rejects_false_hits(self):
        mapper = self.mapper()
        self.assertTrue(mapper.plausible(pupil_for((700, 450)), PUPIL_SIZE))
        self.assertFalse(mapper.plausible(pupil_for((700, 450)), PUPIL_SIZE*.3))
        self.assertFalse(mapper.plausible((10, 10), PUPIL_SIZE))

    def test_either_eye_alone_and_fusion(self):
        gaze = GazeMapper(dict(left=self.mapper(), right=self.mapper(-1.0)))
        target = np.array([900.0, 300.0])
        both = dict(left=(pupil_for(target), PUPIL_SIZE), right=(pupil_for(target, -1.0), PUPIL_SIZE))
        self.assertLess(np.linalg.norm(gaze(both) - target), 15)
        self.assertLess(np.linalg.norm(gaze(dict(right=both['right'])) - target), 15)
        self.assertIsNone(gaze({}))

    def test_worse_eye_gets_less_weight_and_a_bad_eye_is_dropped(self):
        gaze = GazeMapper(dict(left=self.mapper(), right=self.mapper()))
        gaze.set_weights(dict(left=20.0, right=40.0))
        self.assertGreater(gaze.eyes['left'].weight, gaze.eyes['right'].weight)
        gaze.set_weights(dict(left=20.0, right=100.0))
        self.assertEqual(list(gaze.eyes), ['left'])

    def test_eye_without_validation_data_is_dropped(self):
        gaze = GazeMapper(dict(left=self.mapper(), right=self.mapper()))
        gaze.set_weights(dict(right=30.0))
        self.assertEqual(list(gaze.eyes), ['right'])

    def test_save_round_trip(self):
        gaze = GazeMapper(dict(left=self.mapper()))
        again = GazeMapper.from_dict(gaze.to_dict())
        pupil = pupil_for((300, 700))
        self.assertTrue(np.allclose(gaze.eyes['left'](pupil), again.eyes['left'](pupil)))

    def test_robust_centre_ignores_outliers(self):
        points = [(100 + i*.01, 50) for i in range(40)] + [(300, 300)]*5
        self.assertLess(np.linalg.norm(robust_centre(points) - (100.2, 50)), 1)


class FilterTest(unittest.TestCase):
    def test_steadies_a_fixation_and_follows_a_saccade(self):
        rng = np.random.default_rng(0)
        smooth = OneEuroFilter()
        out = [smooth(i/120, (700, 450) + rng.normal(0, 12, 2)) for i in range(240)]
        self.assertLess(np.std(out[120:], axis=0).max(), 3)
        start = 240
        for i in range(start, start + 30):  # 250 ms after a 600-point saccade
            last = smooth(i/120, (1300, 450) + rng.normal(0, 12, 2))
        self.assertLess(abs(last[0] - 1300), 60)


class BlinkTest(unittest.TestCase):
    def blink(self, clicker, hold, returned=(700, 450)):
        t = 0.0
        for _ in range(60):
            clicker.opened(t, (700, 450))
            t += 1/120
        result = None
        for _ in range(int(hold*120)):
            clicker.closed(t)
            t += 1/120
        for _ in range(12):
            result = clicker.opened(t, returned) or result
            t += 1/120
        return result

    def test_ordinary_blink_does_not_click(self):
        self.assertIsNone(self.blink(BlinkClicker(), .15))

    def test_long_blink_left_clicks_and_longer_right_clicks(self):
        self.assertEqual(self.blink(BlinkClicker(), .6)[0], 'left')
        self.assertEqual(self.blink(BlinkClicker(), 1.6)[0], 'right')

    def test_closing_eyes_for_a_while_does_nothing(self):
        self.assertIsNone(self.blink(BlinkClicker(), 4.0))

    def test_looking_elsewhere_afterwards_is_not_a_click(self):
        self.assertIsNone(self.blink(BlinkClicker(), .6, returned=(700, 880)))

    def test_click_lands_before_the_lids_dragged_the_cursor(self):
        clicker = BlinkClicker()
        t = 0.0
        for _ in range(60):
            clicker.opened(t, (700, 450))
            t += 1/120
        for i in range(10):  # the closing lid drags the pupil estimate down
            clicker.opened(t, (700, 450 + 8*i))
            t += 1/120
        for _ in range(60):
            clicker.closed(t)
            t += 1/120
        result = None
        for _ in range(12):
            result = clicker.opened(t, (700, 460)) or result
            t += 1/120
        self.assertEqual(result[0], 'left')
        self.assertLess(np.linalg.norm(result[1] - (700, 450)), 10)

    def test_stray_pupil_hit_does_not_split_a_blink(self):
        clicker = BlinkClicker()
        t = 0.0
        for _ in range(60):
            clicker.opened(t, (700, 450))
            t += 1/120
        for i in range(72):
            if i == 30:
                self.assertIsNone(clicker.opened(t, (700, 450)))
            else:
                clicker.closed(t)
            t += 1/120
        result = None
        for _ in range(12):
            result = clicker.opened(t, (700, 450)) or result
            t += 1/120
        self.assertEqual(result[0], 'left')


if __name__ == '__main__':
    unittest.main()
