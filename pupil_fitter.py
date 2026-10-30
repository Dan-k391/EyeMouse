"""2D pupil ellipse fitter ported from 3D-Eye-Tracker's main/pupilFitter.h.

Original C++ by Jason Orlosky, integrated by Yuta Itoh in
https://github.com/YutaItoh/3D-Eye-Tracker (MIT, see third_party/3D-Eye-Tracker/LICENSE).

The pipeline follows PupilFitter::pupilAreaFitRR step by step: darkest-area
search, two dark thresholds, Canny AND contour candidates, gradient-weighted
point refinement and two rounds of "ellipse method" outlier rejection. Loops are
vectorized with NumPy so it keeps up with 120 FPS capture.

Deviations from upstream, all deliberate:
* Pixel distances were tuned for 640x480; they are scaled to the input size.
* The dataset-specific search limits (i > 60, j < 530, x < 620) are dropped.
* The `angle > 5` check is dropped: it rejected near-axis-aligned pupils.
* A failed search returns None instead of silently using contour 0.
"""
import cv2
import numpy as np

UPSTREAM_WIDTH = 640


class PupilFitter:
    def __init__(self, frame_width=320, darkest_offset_low=10, darkest_offset_high=20,
                 canny_low=10, canny_high=30):
        self.scale = frame_width / UPSTREAM_WIDTH
        self.darkest_offset_low = darkest_offset_low    # upstream darkestPixelL1
        self.darkest_offset_high = darkest_offset_high  # upstream darkestPixelL2
        self.canny_low = canny_low
        self.canny_high = canny_high

    def px(self, value, minimum=1):
        return max(minimum, int(round(value * self.scale)))

    def fit(self, gray, roi=None):
        """Return {'center','axes','angle','points'} in full-frame pixels, or None."""
        ox, oy = 0, 0
        if roi is not None:
            ox, oy, w, h = roi
            gray = gray[oy:oy+h, ox:ox+w]
        try:
            found = self._fit(np.ascontiguousarray(gray))
        except cv2.error:
            return None
        if found is None:
            return None
        (cx, cy), axes, angle, points = found
        return dict(center=(cx+ox, cy+oy), axes=axes, angle=angle,
                    points=points+np.array([ox, oy], dtype=np.float32))

    def _fit(self, gray):
        rows, cols = gray.shape
        # Upstream default search square is 240 px at 640x480.
        size = min(self.px(240), rows, cols)
        if size < 30:
            return None
        dx, dy = self._darkest_area(gray)
        x0 = int(np.clip(dx - size//2, 0, cols - size))
        y0 = int(np.clip(dy - size//2, 0, rows - size))
        window = gray[y0:y0+size, x0:x0+size]
        darkest = self._darkest_pixel(window)

        _, thresh_low = cv2.threshold(window, darkest + self.darkest_offset_low, 255, cv2.THRESH_BINARY_INV)
        _, thresh_high = cv2.threshold(window, darkest + self.darkest_offset_high, 255, cv2.THRESH_BINARY_INV)
        contour_low = self._biggest(thresh_low)
        contour_high = self._biggest(thresh_high)
        if contour_low is None or contour_high is None:
            return None

        canny_low = cv2.Canny(window, self.canny_low, self.canny_low*3, apertureSize=3)
        canny_high = cv2.Canny(window, self.canny_high, self.canny_high*3, apertureSize=3)
        # Logical AND of dark-blob contours and Canny edges.
        with_outliers = np.vstack([self._candidates(contour_low, canny_low),
                                   self._candidates(contour_high, canny_high)])
        if len(with_outliers) <= 5:
            return None
        points = self._refine(with_outliers, window, self.px(8), 2)

        # "Ellipse method", twice: AND the original candidates with a drawn ellipse.
        for _ in range(2):
            if len(points) <= 5:
                return None
            canvas = np.zeros_like(window)
            box = cv2.fitEllipse(points.astype(np.float32))
            if 0 < box[0][0] < size and 0 < box[0][1] < size:
                cv2.ellipse(canvas, box, 255, 2, cv2.LINE_8)
            points = self._candidates(with_outliers, canvas).astype(np.float32)
        if len(points) <= 5:
            return None
        points = self._refine(points, window, self.px(10), 2)
        if len(points) <= 5:
            return None

        # Re-refine on the outline of the fitted ellipse itself.
        canvas = np.zeros_like(window)
        cv2.ellipse(canvas, cv2.fitEllipse(points.astype(np.float32)), 255, 1, cv2.LINE_8)
        outline, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if outline and len(outline[0]) > 5:
            points = self._refine(outline[0].reshape(-1, 2).astype(np.float32), canvas, self.px(6), 2)
        if len(points) <= 5:
            return None
        (cx, cy), axes, angle = cv2.fitEllipse(points.astype(np.float32))
        if not self._plausible(cx, cy, axes, size):
            return None
        return (cx+x0, cy+y0), axes, angle, points+np.array([x0, y0], dtype=np.float32)

    def _plausible(self, cx, cy, axes, size):
        small, large = sorted(axes)
        return (np.isfinite([cx, cy, small, large]).all() and small >= self.px(10, 3)
                and large <= size and large <= 3*small and 0 <= cx < size and 0 <= cy < size)

    def _darkest_area(self, gray):
        """getDarkestPixelArea: darkest sparse square (corners excluded) on a coarse grid."""
        rows, cols = gray.shape
        width = min(self.px(30), (min(rows, cols) - 1)//2 - 1)
        if width < 2:
            return cols//2, rows//2
        step = max(1, width//3)
        stride = self.px(10)
        centres_y = np.arange(width, rows - width, stride)
        centres_x = np.arange(width, cols - width, stride)
        if not len(centres_y) or not len(centres_x):
            return cols//2, rows//2
        offsets = list(range(-width, width+1, step))
        image = gray.astype(np.int32)
        total = np.zeros((len(centres_y), len(centres_x)), np.int32)
        for d in offsets:
            for c in offsets:
                if abs(d) == width and abs(c) == width:
                    continue
                total += image[np.ix_(centres_y + d, centres_x + c)]
        i, j = np.unravel_index(np.argmin(total), total.shape)
        return int(centres_x[j]), int(centres_y[i])

    def _darkest_pixel(self, window):
        """getDarkestPixelBetter: 1st percentile of a sparse sample grid."""
        step = self.px(5)
        samples = np.sort(window[2:-2:step, 2:-2:step], axis=None)
        return int(samples[len(samples)//100]) if len(samples) else int(window.min())

    def _biggest(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best, best_area = None, 0
        for contour in contours:
            if len(contour) <= self.px(40):
                continue
            area = cv2.contourArea(contour)
            if area > best_area:
                best, best_area = contour, area
        return None if best is None else best.reshape(-1, 2)

    def _candidates(self, contour, edges):
        """getCandidates: contour pixels (and +-z neighbours) that are set in `edges`."""
        if len(contour) <= 10:
            return np.empty((0, 2), np.int32)
        h, w = edges.shape
        thickness = self.px(3)
        reach = 2*thickness
        pts = np.round(contour).astype(np.int32)
        x, y = pts[:, 0], pts[:, 1]
        inside = (x - reach > 0) & (x + reach < w) & (y - reach > 0) & (y + reach < h)
        pts = pts[inside]
        shifts = [(0, 0)]
        for z in range(1, reach+1, 2):
            shifts += [(0, z), (0, -z), (z, 0), (-z, 0)]
        probe = (pts[:, None, :] + np.array(shifts)[None]).reshape(-1, 2)
        hit = probe[edges[probe[:, 1], probe[:, 0]] > 0]
        return np.unique(hit, axis=0) if len(hit) else np.empty((0, 2), np.int32)

    @staticmethod
    def _refine(points, gray, check_thickness, check_spacing):
        """refinePoints: move each point to the gradient-weighted mean along x and y."""
        points = np.asarray(points, np.float32)
        image = gray.astype(np.int16)
        h, w = image.shape
        reach = check_thickness*check_spacing
        offsets = np.arange(-reach, reach, check_spacing)
        xi = points[:, 0].astype(np.int32)
        yi = points[:, 1].astype(np.int32)
        refined = points.copy()
        for axis, limit in ((0, w), (1, h)):
            base = xi if axis == 0 else yi
            ok = (base - reach - 1 >= 0) & (base + reach + 1 < limit)
            if not ok.any():
                continue
            along = base[ok, None] + offsets[None]
            other = (yi if axis == 0 else xi)[ok, None].repeat(len(offsets), 1)
            def at(shift):
                return image[other, along+shift] if axis == 0 else image[along+shift, other]
            centre = at(0)
            weight = (np.abs(centre - at(-1)) + np.abs(centre - at(1))).astype(np.float32)
            total = weight.sum(1)
            position = points[ok, axis, None] + offsets[None]
            moved = np.where(total > 0, (position*weight).sum(1)/np.maximum(total, 1), points[ok, axis])
            refined[ok, axis] = moved
        return refined
