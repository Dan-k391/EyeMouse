"""Webcam gaze features and their mapping to the screen, for webcam_mouse.py.

Works on MediaPipe's 478 face landmarks and head matrix. No camera or OS code
lives here, so all of it can be tested on synthetic data.
"""
import numpy as np

# MediaPipe face mesh indices: (outer corner, inner corner, iris centre) of each eye.
EYES = ((33, 133, 468), (263, 362, 473))
# Fixed scales instead of sample spread: a head that barely moved during
# calibration must not have its noise amplified into the fit.
SCALES = np.array([.08, .05, .15, .15, 15.0, 15.0])
EYE_FEATURES = (0, 1)
HEAD_FEATURES = (2, 3, 4, 5)
MODES = dict(gaze=EYE_FEATURES + HEAD_FEATURES, head=HEAD_FEATURES)


def iris_offset(points, outer, inner, iris):
    """Iris position inside one eye, in eye widths along and across the corner-to-corner line."""
    along = points[inner] - points[outer]
    width = np.linalg.norm(along)
    if width < 1e-6:
        return None
    along = along/width
    across = np.array([-along[1], along[0]])
    offset = (points[iris] - (points[outer] + points[inner])/2)/width
    return offset @ along, offset @ across


def face_features(points, matrix):
    """[iris_x, iris_y, yaw, pitch, head_x, head_y] from landmark pixels and a 4x4 head matrix.

    The inner/outer order is mirrored between the eyes, which flips both of one eye's
    axes, so it is negated: both eyes then agree when they turn.
    """
    offsets = [iris_offset(points, *eye) for eye in EYES]
    if any(offset is None for offset in offsets):
        return None
    (x0, y0), (x1, y1) = offsets
    forward = matrix[:3, 2]
    return np.array([(x0 - x1)/2, (y0 - y1)/2, forward[0], forward[1], matrix[0, 3], matrix[1, 3]])


def face_terms(features):
    """Linear in everything, plus second-order iris terms when iris features are present."""
    terms = [np.ones(len(features)), *features.T]
    if features.shape[1] == len(SCALES):
        x, y = features[:, 0], features[:, 1]
        terms += [x*y, x*x, y*y]
    return np.column_stack(terms)


class FaceMapper:
    def __init__(self, mode, coeffs, centre):
        self.mode = mode
        self.columns = list(MODES[mode])
        self.coeffs = np.asarray(coeffs, float)
        self.centre = np.asarray(centre, float)

    @classmethod
    def fit(cls, mode, features, targets, ridge=.002):
        columns = list(MODES[mode])
        features, targets = np.asarray(features, float)[:, columns], np.asarray(targets, float)
        centre = features.mean(0)
        terms = face_terms((features - centre)/SCALES[columns])
        if len(terms) <= terms.shape[1]:
            return None
        penalty = ridge*len(terms)*np.eye(terms.shape[1])
        penalty[0, 0] = 0
        coeffs = np.linalg.solve(terms.T @ terms + penalty, terms.T @ targets)
        return cls(mode, coeffs, centre)

    def point(self, features):
        z = (np.asarray(features, float)[self.columns] - self.centre)/SCALES[self.columns]
        return (face_terms(z[None]) @ self.coeffs)[0]

    def __call__(self, observations):
        """Same call as GazeMapper, so the shared cursor loop can drive either."""
        face = observations.get('face')
        return None if face is None else self.point(face[0])

    def to_dict(self):
        return dict(mode=self.mode, coeffs=self.coeffs.tolist(), centre=self.centre.tolist())

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
