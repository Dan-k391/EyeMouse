"""Pupil-to-screen mapping, cursor smoothing and blink clicks for eyemouse.py.

No camera or OS code lives here, so all of it can be tested on synthetic data.
The headset keeps each camera fixed relative to its eye, so a polynomial from 2D
pupil position to screen position is enough; the 3D eye model is not used.
"""
from collections import deque
import math

import numpy as np

SIZE_RANGE = (.55, 1.7)  # pupil size accepted, relative to the calibrated median
RANGE_MARGIN = .35       # fraction of the calibrated pupil range accepted beyond it


def robust_centre(points, cutoff=3.0):
    """Median after dropping points further than `cutoff` MADs, for one target's samples."""
    points = np.asarray(points, float)
    median = np.median(points, axis=0)
    distance = np.linalg.norm(points - median, axis=1)
    mad = np.median(distance)
    kept = points[distance <= cutoff*max(mad, 1e-6)]
    return np.median(kept, axis=0) if len(kept) else median


def poly_terms(points):
    x, y = points[:, 0], points[:, 1]
    return np.column_stack([np.ones(len(x)), x, y, x*y, x*x, y*y])


class EyeMapper:
    """Second-order polynomial from one eye's pupil centre to screen points."""

    def __init__(self, coeffs, centre, spread, low, high, pupil_size, weight=1.0):
        self.coeffs = np.asarray(coeffs, float)
        self.centre = np.asarray(centre, float)
        self.spread = np.asarray(spread, float)
        self.low = np.asarray(low, float)
        self.high = np.asarray(high, float)
        self.pupil_size = float(pupil_size)
        self.weight = float(weight)

    @classmethod
    def fit(cls, pupils, targets, pupil_size, ridge=1e-3):
        pupils, targets = np.asarray(pupils, float), np.asarray(targets, float)
        centre, spread = pupils.mean(0), pupils.std(0)
        if len(pupils) < 6 or spread.min() < .5:
            return None  # too few targets, or the pupil barely moved
        terms = poly_terms((pupils - centre)/spread)
        penalty = ridge*len(pupils)*np.eye(terms.shape[1])
        penalty[0, 0] = 0
        coeffs = np.linalg.solve(terms.T @ terms + penalty, terms.T @ targets)
        return cls(coeffs, centre, spread, pupils.min(0), pupils.max(0), pupil_size)

    def __call__(self, pupil):
        point = (np.asarray(pupil, float)[None] - self.centre)/self.spread
        return (poly_terms(point) @ self.coeffs)[0]

    def plausible(self, pupil, size):
        """Reject lash and eyelid false hits: wrong pupil size or far outside the calibrated range."""
        margin = RANGE_MARGIN*(self.high - self.low)
        pupil = np.asarray(pupil, float)
        return (SIZE_RANGE[0]*self.pupil_size <= size <= SIZE_RANGE[1]*self.pupil_size
                and bool(np.all(pupil >= self.low - margin)) and bool(np.all(pupil <= self.high + margin)))

    def to_dict(self):
        return dict(coeffs=self.coeffs.tolist(), centre=self.centre.tolist(), spread=self.spread.tolist(),
                    low=self.low.tolist(), high=self.high.tolist(), pupil_size=self.pupil_size, weight=self.weight)


class GazeMapper:
    """Fuses both eyes, weighted by how well each one validated. Either eye alone still works."""

    def __init__(self, eyes):
        self.eyes = eyes

    def __call__(self, observations):
        """observations: {role: (pupil_xy, pupil_size)}. Returns a screen point, or None."""
        points, weights = [], []
        for role, (pupil, size) in observations.items():
            mapper = self.eyes.get(role)
            if mapper and mapper.plausible(pupil, size):
                points.append(mapper(pupil))
                weights.append(mapper.weight)
        return np.average(points, axis=0, weights=weights) if points else None

    def set_weights(self, errors):
        """errors: {role: mean validation error in points}. An eye 3x worse than the best is dropped."""
        best = min(errors.values())
        self.eyes = {role: mapper for role, mapper in self.eyes.items() if errors.get(role, math.inf) <= 3*best}
        for role, mapper in self.eyes.items():
            mapper.weight = 1/max(errors[role], 1.0)**2

    def to_dict(self):
        return {role: mapper.to_dict() for role, mapper in self.eyes.items()}

    @classmethod
    def from_dict(cls, data):
        return cls({role: EyeMapper(**values) for role, values in data.items()})


class OneEuroFilter:
    """Casiez et al. 2012: heavy smoothing while fixating, little lag during saccades."""

    def __init__(self, min_cutoff=.5, beta=.005, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self, point=None):
        self.value = None if point is None else np.asarray(point, float)
        self.speed = np.zeros(2)
        self.time = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1/(1 + 1/(2*math.pi*cutoff*dt))

    def __call__(self, now, point):
        point = np.asarray(point, float)
        if self.value is None or self.time is None or now <= self.time:
            self.value = point if self.value is None else self.value
            self.time = now
            return self.value
        dt = now - self.time
        self.time = now
        self.speed += self.alpha(self.d_cutoff, dt)*((point - self.value)/dt - self.speed)
        cutoff = self.min_cutoff + self.beta*np.linalg.norm(self.speed)
        self.value = self.value + self.alpha(cutoff, dt)*(point - self.value)
        return self.value


class BlinkClicker:
    """Turns a deliberate long blink of both eyes into a click.

    Ordinary blinks are shorter than left_hold and are ignored. The click lands where
    the cursor was just before the lids started to move, not where they dragged it.
    """

    def __init__(self, left_hold=(.35, 1.2), right_hold=(1.2, 2.5), lead=.12, reopen=.04, radius=220.0):
        self.left_hold, self.right_hold = left_hold, right_hold
        self.lead, self.reopen, self.radius = lead, reopen, radius
        self.history = deque()
        self.reset()

    def reset(self):
        self.history.clear()
        self.closed_at = None
        self.anchor = None
        self.open_since = None

    def closed(self, now):
        self.open_since = None
        if self.closed_at is None and self.history:
            self.closed_at = self.history[-1][0]
            earlier = [p for t, p in self.history if t <= self.closed_at - self.lead]
            self.anchor = earlier[-1] if earlier else self.history[0][1]

    def opened(self, now, point):
        """Feed a visible cursor point. Returns ('left'|'right', anchor) when a click completes."""
        point = np.asarray(point, float)
        if self.closed_at is None:
            self.history.append((now, point))
            while now - self.history[0][0] > .5:
                self.history.popleft()
            return None
        if self.open_since is None:  # one stray pupil hit must not end the blink
            self.open_since = now
        if now - self.open_since < self.reopen:
            return None
        held, anchor = self.open_since - self.closed_at, self.anchor
        self.closed_at = self.anchor = self.open_since = None
        self.history.clear()
        self.history.append((now, point))
        if anchor is None or np.linalg.norm(point - anchor) > self.radius:
            return None  # the eyes came back elsewhere: looking down, not a blink
        if self.left_hold[0] <= held < self.left_hold[1]:
            return 'left', anchor
        if self.right_hold[0] <= held < self.right_hold[1]:
            return 'right', anchor
        return None
