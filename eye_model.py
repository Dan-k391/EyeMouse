"""3D eye model ported from 3D-Eye-Tracker's EyeModelUpdater and singleeyefitter.

Original C++: https://github.com/YutaItoh/3D-Eye-Tracker (MIT, see
third_party/3D-Eye-Tracker/LICENSE), main/eye_model_updater.cpp by Yuta Itoh and
singleeyefitter/SingleEyeFitter.cpp by Leszek Swirski.

Upstream's runtime path is ported: collect ~30 pupil ellipses spread over the
image, unproject each into two candidate 3D circles, RANSAC-intersect the
projected gaze lines to find the eyeball centre, then intersect each new pupil
ray with the 12 mm eyeball. Upstream never calls the Ceres refinement steps, so
neither does this port. The circle unprojection solves the same cone/plane
problem as upstream's Safaee-Rad implementation, written with an eigen
decomposition instead of the closed-form cubic.

Coordinates follow upstream: camera at the origin, x right, y down, z forward,
image coordinates relative to the image centre, units of millimetres after the
model is scaled to a 12 mm eyeball radius. Metric values are only as good as
the focal length estimate.
"""
import math

import numpy as np

EYE_RADIUS_MM = 12.0


def ellipse_from_box(box, frame_size):
    """cv2 RotatedRect -> centred (cx, cy, a, b, angle_rad), as toImgCoordInv + toEllipse."""
    (x, y), (w, h), angle = box
    return (x - frame_size[0]/2, y - frame_size[1]/2, w/2, h/2, math.radians(angle))


def box_from_ellipse(ellipse, frame_size):
    cx, cy, a, b, angle = ellipse
    return ((cx + frame_size[0]/2, cy + frame_size[1]/2), (2*a, 2*b), math.degrees(angle))


def conic_matrix(ellipse):
    cx, cy, a, b, angle = ellipse
    c, s = math.cos(angle), math.sin(angle)
    # Homogeneous image point -> ellipse-aligned coordinates.
    to_local = np.array([[c, s, -(c*cx + s*cy)],
                         [-s, c, s*cx - c*cy],
                         [0, 0, 1]])
    return to_local.T @ np.diag([1/a**2, 1/b**2, -1]) @ to_local


def unproject(ellipse, radius, focal):
    """Return the two 3D circles (centre, normal) of `radius` that project to `ellipse`."""
    scale = np.diag([1, 1, 1/focal])
    cone = scale @ conic_matrix(ellipse) @ scale
    values, vectors = np.linalg.eigh(cone)
    if np.sum(values > 0) == 1:
        values, vectors = -values, vectors
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    l1, l2, l3 = values
    if not (l1 >= l2 > 0 > l3):
        return None
    side = math.sqrt((l1 - l2)/(l1 - l3))
    along = math.sqrt((l2 - l3)/(l1 - l3))
    inverse = np.linalg.inv(cone)
    circles = []
    for sign in (1, -1):
        normal = vectors @ np.array([sign*side, 0, along])
        normal /= np.linalg.norm(normal)
        # The centre of a planar section is where the cone gradient is normal to the plane.
        centre = inverse @ normal
        centre /= normal @ centre
        if centre[2] < 0:
            centre = -centre
        in_plane = np.cross(normal, [1, 0, 0] if abs(normal[0]) < .9 else [0, 1, 0])
        in_plane /= np.linalg.norm(in_plane)
        section_radius = math.sqrt(max(-(centre @ cone @ centre)/(in_plane @ cone @ in_plane), 0))
        if section_radius <= 0:
            return None
        centre *= radius/section_radius
        if normal @ centre > 0:  # gaze points toward the camera
            normal = -normal
        circles.append((centre, normal))
    return circles


def project_point(point, focal):
    return focal*point[:2]/point[2]


def project_circle(centre, normal, radius, focal, samples=36):
    """Exact image ellipse of a 3D circle, sampled and fitted (replaces upstream's conic route)."""
    import cv2
    helper = np.cross(normal, [1, 0, 0] if abs(normal[0]) < .9 else [0, 1, 0])
    u = helper/np.linalg.norm(helper)
    v = np.cross(normal, u)
    t = np.linspace(0, 2*math.pi, samples, endpoint=False)
    rim = centre + radius*(np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    if np.any(rim[:, 2] <= 0):
        return None
    image = (focal*rim[:, :2]/rim[:, 2:3]).astype(np.float32)
    (x, y), (w, h), angle = cv2.fitEllipse(image)
    return (x, y, w/2, h/2, math.radians(angle))


def nearest_intersect(origins, directions):
    """Least-squares point closest to all lines (2D or 3D)."""
    dim = origins.shape[1]
    lhs = np.zeros((dim, dim))
    rhs = np.zeros(dim)
    for o, d in zip(origins, directions):
        d = d/np.linalg.norm(d)
        p = np.eye(dim) - np.outer(d, d)
        lhs += p
        rhs += p @ o
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def ray_sphere(direction, centre, radius):
    """Nearer intersection of the camera ray `direction` with a sphere, or None."""
    d = direction/np.linalg.norm(direction)
    b = d @ centre
    disc = b*b - centre @ centre + radius*radius
    if disc < 0:
        return None
    return (b - math.sqrt(disc))*d


def similarity(observed, projected):
    """Ellipse2D::similarity from singleeyefitter/Ellipse.h (angle difference wrapped)."""
    def normalized(e):
        cx, cy, a, b, angle = e
        if b > a:
            a, b, angle = b, a, angle + math.pi/2
        return cx, cy, a, b, angle
    cx, cy, major, minor, angle = normalized(observed)
    ecx, ecy, emajor, eminor, eangle = normalized(projected)
    size = (eminor + emajor)/2
    rate_diff = eminor/emajor - minor/major
    dist = ((ecx - cx)/size)**2 + ((ecy - cy)/size)**2
    dangle = (angle - eangle + math.pi/2) % math.pi - math.pi/2
    angle_term = 10*math.exp(-5*eminor/emajor)*(dangle/math.radians(5))**2
    area_term = (1 - minor*major/(eminor*emajor))**2
    return math.exp(-(dist + area_term + 10*rate_diff**2 + angle_term)/2)


class EyeModelFitter:
    """singleeyefitter::EyeModelFitter, initialisation path only."""

    def __init__(self, focal, ransac_epsilon=10.0):
        self.focal = focal
        self.ransac_epsilon = ransac_epsilon
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.ellipses = []
        self.circles = []
        self.valid = []
        self.eye_centre = None
        self.eye_radius = None

    def add_observation(self, ellipse):
        self.ellipses.append(ellipse)

    def unproject_observations(self, pupil_radius=1.0, eye_z=20.0):
        pairs, origins, directions = [], [], []
        for ellipse in self.ellipses:
            pair = unproject(ellipse, pupil_radius, self.focal)
            if pair is None:
                pairs.append(None)
                continue
            c, v = pair[0]
            c_proj = project_point(c, self.focal)
            v_proj = project_point(c + v, self.focal) - c_proj
            pairs.append(pair)
            origins.append(c_proj)
            directions.append(v_proj/np.linalg.norm(v_proj))
        usable = [i for i, p in enumerate(pairs) if p is not None]
        if len(usable) < 2:
            self.eye_centre = None
            return
        origins, directions = np.array(origins), np.array(directions)

        def distances(point):
            rel = point - origins
            return np.abs(rel[:, 0]*directions[:, 1] - rel[:, 1]*directions[:, 0])

        # RANSAC parameters from upstream: n=2, w=0.3, p=0.9999.
        n, w = 2, .3
        iterations = int(math.ceil(math.log(1 - .9999)/math.log(1 - w**n)))
        best, best_error, best_inliers = None, math.inf, []
        for _ in range(iterations):
            sample = self.rng.choice(len(origins), n, replace=False)
            try:
                centre = nearest_intersect(origins[sample], directions[sample])
            except np.linalg.LinAlgError:
                continue
            inliers = np.flatnonzero(distances(centre) < self.ransac_epsilon)
            if len(inliers) <= w*len(origins):
                continue
            centre = nearest_intersect(origins[inliers], directions[inliers])
            error = np.minimum(distances(centre)**2, self.ransac_epsilon**2).sum()
            if error < best_error:
                best, best_error, best_inliers = centre, error, inliers
        self.valid = [False]*len(self.ellipses)
        if best is None:
            self.eye_centre = None
            self.circles = [p[0] if p else None for p in pairs]
            return
        for k in best_inliers:
            self.valid[usable[k]] = True
        self.eye_centre = np.array([*(best*eye_z/self.focal), eye_z])
        self.eye_radius = 1.0
        self.circles = []
        for k, pair in enumerate(pairs):
            if pair is None:
                self.circles.append(None)
                continue
            self.circles.append(self._disambiguate(pair, best))

    def _disambiguate(self, pair, eye_centre_proj):
        # Pick the circle whose projected gaze points away from the projected eye centre.
        c, v = pair[0]
        c_proj = project_point(c, self.focal)
        v_proj = project_point(c + v, self.focal) - c_proj
        return pair[0] if (c_proj - eye_centre_proj) @ v_proj >= 0 else pair[1]

    def initialise_model(self):
        if self.eye_centre is None:
            return False
        distances = []
        for circle, valid in zip(self.circles, self.valid):
            if circle is None or not valid:
                continue
            centre, normal = circle
            pupil = nearest_intersect(np.array([self.eye_centre, np.zeros(3)]),
                                      np.array([normal, centre/np.linalg.norm(centre)]))
            distances.append(np.linalg.norm(pupil - self.eye_centre))
        if not distances or np.mean(distances) <= 0:
            self.eye_centre = None
            return False
        self.eye_radius = float(np.mean(distances))
        scale = EYE_RADIUS_MM/self.eye_radius
        self.eye_radius = EYE_RADIUS_MM
        self.eye_centre = self.eye_centre*scale
        return True

    def pupil_on_eye(self, ellipse):
        """unproject_single_observation + initialise_single_observation for one ellipse."""
        if self.eye_centre is None:
            return None
        pair = unproject(ellipse, self.eye_radius, self.focal)
        if pair is None:
            return None
        circle_centre, _ = self._disambiguate(pair, project_point(self.eye_centre, self.focal))
        centre = ray_sphere(circle_centre, self.eye_centre, self.eye_radius)
        if centre is None:
            return None
        radius = self.eye_radius/circle_centre[2]*centre[2]
        normal = (centre - self.eye_centre)/self.eye_radius
        return centre, normal, radius


class EyeModelUpdater:
    """eye_tracker::EyeModelUpdater: spatially-binned sampling and reliability."""

    MAX_COUNT_DEFAULT = 30
    RELIABILITY_THRESHOLD = 0.8

    def __init__(self, focal, frame_size, bin_size=8, ransac_epsilon=5.0):
        self.focal = focal
        self.frame_size = frame_size
        self.bin_size = bin_size  # upstream: 16 px at 640x480
        self.fitter = EyeModelFitter(focal, ransac_epsilon)
        self.reset()

    def reset(self):
        self.fitter.reset()
        self.taken_bins = set()
        self.count = 0
        self.max_count = self.MAX_COUNT_DEFAULT
        self.built = False
        self.failed_builds = 0

    def add_more_samples(self, n=10):
        self.max_count += n
        self.built = False

    def add_observation(self, ellipse):
        if self.built or self.count >= self.max_count:
            return False
        x = ellipse[0] + self.frame_size[0]/2
        y = ellipse[1] + self.frame_size[1]/2
        cell = (int(round(x/self.bin_size)), int(round(y/self.bin_size)))
        if cell in self.taken_bins:
            return False
        self.taken_bins.add(cell)
        self.fitter.add_observation(ellipse)
        self.count += 1
        if self.count == self.max_count:
            self.fitter.unproject_observations()
            self.built = self.fitter.initialise_model()
            if not self.built:
                # Upstream would keep a null eye; ask for more varied samples instead.
                self.failed_builds += 1
                self.max_count += 10
        return True

    def update(self, ellipse):
        """Feed one detected ellipse. Returns a result dict for display/recording."""
        if not self.built:
            self.add_observation(ellipse)
            return dict(built=False, count=self.count, max_count=self.max_count)
        pupil = self.fitter.pupil_on_eye(ellipse)
        if pupil is None:
            return dict(built=True, reliability=0.0, reliable=False)
        centre, normal, radius = pupil
        projected = project_circle(centre, normal, radius, self.focal)
        reliability = similarity(ellipse, projected) if projected else 0.0
        return dict(built=True, reliability=reliability,
                    reliable=reliability > self.RELIABILITY_THRESHOLD,
                    pupil_centre=centre, gaze=normal, pupil_radius=radius,
                    eye_centre=self.fitter.eye_centre.copy(), eye_radius=self.fitter.eye_radius)
