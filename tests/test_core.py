"""Synthetic checks for the ported pupil fitter and 3D eye model.

Run: .venv\\Scripts\\python.exe -m unittest discover -s tests
"""
import math
import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye_model import EyeModelUpdater, project_circle, unproject  # noqa: E402
from pupil_fitter import PupilFitter  # noqa: E402

FOCAL = 280.0
SIZE = (320, 240)


def angle_between(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))


def gaze_normal(yaw, pitch):
    # Facing the camera (-z), rotated by yaw/pitch.
    return np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch), -math.cos(yaw)*math.cos(pitch)])


class UnprojectTest(unittest.TestCase):
    def test_round_trip(self):
        centre = np.array([3.0, -2.0, 30.0])
        normal = gaze_normal(.4, -.25)
        ellipse = project_circle(centre, normal, 2.0, FOCAL)
        candidates = unproject(ellipse, 2.0, FOCAL)
        best = min(candidates, key=lambda c: angle_between(c[1], normal))
        self.assertLess(angle_between(best[1], normal), .5)
        self.assertLess(np.linalg.norm(best[0] - centre), .1)


class EyeModelTest(unittest.TestCase):
    def test_recovers_eye_and_gaze(self):
        eye = np.array([1.5, 2.0, 35.0])
        updater = EyeModelUpdater(FOCAL, SIZE)
        rng = np.random.default_rng(1)
        while not updater.built:
            n = gaze_normal(rng.uniform(-.5, .5), rng.uniform(-.4, .4))
            el = project_circle(eye + 12*n, n, 2.0, FOCAL)
            updater.update(el)
            self.assertLess(updater.count, 400)
        self.assertLess(np.linalg.norm(updater.fitter.eye_centre - eye), 1.0)
        n = gaze_normal(.3, .2)
        result = updater.update(project_circle(eye + 12*n, n, 2.0, FOCAL))
        self.assertTrue(result['reliable'], result)
        self.assertLess(angle_between(result['gaze'], n), 2.0)
        self.assertAlmostEqual(result['pupil_radius'], 2.0, delta=.2)


class PupilFitterTest(unittest.TestCase):
    def synthetic(self, centre, axes, angle):
        rng = np.random.default_rng(0)
        img = np.full((240, 320), 150, np.uint8)
        cv2.ellipse(img, (80, 60), (60, 40), 0, 0, 360, 110, -1)  # iris-ish
        cv2.ellipse(img, (centre, axes, angle), 25, -1)
        cv2.circle(img, (int(centre[0]+8), int(centre[1]-6)), 3, 250, -1)  # glint
        img = cv2.GaussianBlur(img, (5, 5), 1.2)
        noise = rng.normal(0, 4, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def test_finds_pupil(self):
        fitter = PupilFitter(320)
        for centre, axes, angle in [((160, 120), (40, 30), 30), ((100, 150), (34, 34), 0), ((220, 90), (44, 28), 95)]:
            found = fitter.fit(self.synthetic(centre, axes, angle))
            self.assertIsNotNone(found, centre)
            self.assertLess(np.hypot(found['center'][0]-centre[0], found['center'][1]-centre[1]), 2.0)
            self.assertLess(abs(max(found['axes']) - max(axes)), 4.0)

    def test_roi_and_speed(self):
        fitter = PupilFitter(320)
        img = self.synthetic((200, 140), (36, 30), 20)
        found = fitter.fit(img, roi=(140, 80, 120, 120))
        self.assertIsNotNone(found)
        self.assertLess(np.hypot(found['center'][0]-200, found['center'][1]-140), 2.0)
        start = time.perf_counter()
        for _ in range(200):
            fitter.fit(img)
        per_frame = (time.perf_counter() - start)/200
        print(f'\nfull-frame fit: {per_frame*1000:.2f} ms')
        self.assertLess(per_frame, 1/120)

    def test_blank_frame(self):
        self.assertIsNone(PupilFitter(320).fit(np.full((240, 320), 128, np.uint8)))


if __name__ == '__main__':
    unittest.main()
