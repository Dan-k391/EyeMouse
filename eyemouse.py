"""Eye-controlled mouse for macOS: calibrate, then steer the cursor with your pupils.

    ./run.sh mouse              # calibrate, then control the cursor
    ./run.sh mouse --reuse      # skip calibration, use the last one
    ./run.sh mouse --no-click   # move the cursor only

Long blink: left click. Longer blink: right click. Touching the real mouse or trackpad
always wins: eye control pauses until it has been still for 2 s.
Ctrl+Option held: freeze the cursor, look at it, release to correct drift.
Ctrl+Option+Cmd: tap to pause or resume, hold 2 s to quit.
"""
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from binocular_demo import Eye, Recorder, Settings
from camera import DEFAULT_DEVICE, DEFAULT_LEFT, DEFAULT_RIGHT, HEIGHT, WIDTH, request_camera_access
import cursor
from gaze_mapper import BlinkClicker, EyeMapper, GazeMapper, OneEuroFilter, robust_centre

ROOT = Path(__file__).parent
CALIBRATION = ROOT/'calibrations'/'latest.json'
WINDOW = 'EyeMouse calibration'
ROLES = ('left', 'right')
MARGIN = .06
FEED_W, FEED_H = 2*WIDTH, 2*HEIGHT
CALIBRATE_AT = [(0, 0), (.5, 0), (1, 0), (0, .5), (.5, .5), (1, .5), (0, 1), (.5, 1), (1, 1),
                (.25, .25), (.75, .25), (.25, .75), (.75, .75)]
VALIDATE_AT = [(.25, .5), (.75, .5), (.5, .25), (.5, .75)]
SETTLE_S, COLLECT_S, TARGET_TIMEOUT_S = .7, .8, 3.0
MIN_TARGET_SAMPLES = 25
MAX_AGE_S = .03           # a pupil older than this counts as lost (a few frames at 120 FPS)
HAND_TOLERANCE = 4.0      # cursor points away from anything we posted means a hand moved it
HAND_PAUSE_S = 2.0
CLICK_HOLD_S = .25        # keep the cursor still after a click so hover menus survive
DRIFT_MIN_HOLD_S = .4
QUIT_HOLD_S = 2.0
REPORT_S = 5.0
ALIGN_RATE = .9           # both eyes must hold this pupil detection rate before calibrating
ALIGN_HOLD_S = 1.0
ALIGN_TIMEOUT_S = 60.0
DRIFT_KEYS = {'control', 'option'}
PAUSE_KEYS = {'control', 'option', 'command'}


class Pupils:
    """Newest pupil of each eye, fed by the tracker threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}
        self.capture = None
        self.seen = dict.fromkeys(ROLES, 0)
        self.found = dict.fromkeys(ROLES, 0)

    def __call__(self, role, captured, pupil):
        self.seen[role] += 1
        if pupil is None:
            return
        self.found[role] += 1
        sample = (captured, pupil['center'], max(pupil['axes']))
        with self.lock:
            self.latest[role] = sample
            if self.capture is not None:
                self.capture[role].append(sample)

    def start_capture(self):
        with self.lock:
            self.capture = {role: [] for role in ROLES}

    def stop_capture(self):
        with self.lock:
            captured, self.capture = self.capture, None
        return captured

    def counts(self):
        with self.lock:
            return {role: len(self.capture[role]) for role in ROLES} if self.capture else dict.fromkeys(ROLES, 0)

    def take_stats(self):
        """Frames seen and pupils found per eye since the last call."""
        stats = {role: (self.seen[role], self.found[role]) for role in ROLES}
        self.seen = dict.fromkeys(ROLES, 0)
        self.found = dict.fromkeys(ROLES, 0)
        return stats

    def fresh(self, now):
        with self.lock:
            return {role: (center, size) for role, (captured, center, size) in self.latest.items()
                    if now - captured < MAX_AGE_S}


class Aborted(Exception):
    pass


class Calibrator:
    """Full-screen target sequence. Target positions come from the real window rectangle."""

    def __init__(self, pupils, screen):
        self.pupils = pupils
        self.width, self.height = screen
        self.rect = (0, 0, *screen)

    def canvas(self):
        return np.full((self.height, self.width, 3), 70, np.uint8)

    def show(self, canvas):
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(1)
        if cursor.key_down(cursor.ESCAPE_KEY):
            raise Aborted

    def message(self, lines, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            canvas = self.canvas()
            remaining = math.ceil(end - time.perf_counter())
            for i, line in enumerate([*lines, f'{remaining}']):
                size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .9, 2)[0]
                cv2.putText(canvas, line, ((self.width - size[0])//2, self.height//2 - 60 + 44*i),
                            cv2.FONT_HERSHEY_SIMPLEX, .9, (240, 240, 240), 2, cv2.LINE_AA)
            self.show(canvas)

    def open(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.message(['Starting...'], 2)
        rect = cv2.getWindowImageRect(WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            self.rect = rect
        print(f'Calibration window: {self.rect} on a {self.width}x{self.height} screen', flush=True)

    def feed(self, canvas, eye, x, y):
        result, rate = eye.tracker.result, eye.tracker.detection_rate()
        good = rate >= ALIGN_RATE
        color = (80, 255, 80) if good else (60, 60, 255)
        if result is not None:
            panel = cv2.resize(result['frame'], (FEED_W, FEED_H), interpolation=cv2.INTER_LINEAR)
            if result['pupil']:
                (cx, cy), axes = result['pupil']['center'], result['pupil']['axes']
                scale = FEED_W/WIDTH
                cv2.ellipse(panel, ((cx*scale, cy*scale), (axes[0]*scale, axes[1]*scale), result['pupil']['angle']),
                            (80, 255, 80), 2, cv2.LINE_AA)
            canvas[y:y + FEED_H, x:x + FEED_W] = panel
        cv2.rectangle(canvas, (x - 3, y - 3), (x + FEED_W + 2, y + FEED_H + 2), color, 3)
        cv2.putText(canvas, f'{eye.role.upper()} eye: pupil found {rate:.0%}', (x, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)

    def align(self, eyes):
        """Wait until both pupils are locked; a missing eye would otherwise be dropped silently."""
        start = time.perf_counter()
        locked_at = None
        while True:
            now = time.perf_counter()
            rates = [eye.tracker.detection_rate() for eye in eyes]
            canvas = self.canvas()
            x0 = (self.width - 2*FEED_W - 40)//2
            y0 = (self.height - FEED_H)//2
            for slot, eye in enumerate(eyes):
                self.feed(canvas, eye, x0 + slot*(FEED_W + 40), y0)
            hint = 'Put the headset on and adjust it until both eyes are green. Esc cancels.'
            cv2.putText(canvas, hint, (x0, y0 + FEED_H + 44), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2, cv2.LINE_AA)
            self.show(canvas)
            locked_at = (locked_at or now) if min(rates) >= ALIGN_RATE else None
            if locked_at and now - locked_at >= ALIGN_HOLD_S:
                return True
            if now - start >= ALIGN_TIMEOUT_S:
                print(f'Both eyes never locked (pupil found {rates[0]:.0%} / {rates[1]:.0%})', flush=True)
                return False

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def canvas_point(self, fraction):
        return (int((MARGIN + (1 - 2*MARGIN)*fraction[0])*self.width),
                int((MARGIN + (1 - 2*MARGIN)*fraction[1])*self.height))

    def screen_point(self, canvas_point):
        x, y, w, h = self.rect
        return x + canvas_point[0]*w/self.width, y + canvas_point[1]*h/self.height

    def dot(self, point, radius):
        canvas = self.canvas()
        cv2.circle(canvas, point, int(radius), (60, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 3, (0, 0, 0), -1, cv2.LINE_AA)
        self.show(canvas)

    def target(self, fraction):
        """Show one target. Returns (screen_point, {role: samples}) after the eyes have settled."""
        point = self.canvas_point(fraction)
        start = time.perf_counter()
        while (elapsed := time.perf_counter() - start) < SETTLE_S:
            self.dot(point, 26 - 16*elapsed/SETTLE_S)
        self.pupils.start_capture()
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            self.dot(point, 10)
            enough = min(self.pupils.counts().values()) >= MIN_TARGET_SAMPLES
            if (elapsed >= COLLECT_S and enough) or elapsed >= TARGET_TIMEOUT_S:
                break
        return self.screen_point(point), self.pupils.stop_capture()

    def run_targets(self, fractions):
        return [self.target(fraction) for fraction in fractions]


def summarise(results):
    """One (screen target, pupil centre) per target per eye, plus each eye's median pupil size."""
    pairs, sizes = {role: [] for role in ROLES}, {role: [] for role in ROLES}
    for screen_point, samples in results:
        for role in ROLES:
            if len(samples[role]) >= MIN_TARGET_SAMPLES:
                pairs[role].append((screen_point, robust_centre([s[1] for s in samples[role]]).tolist()))
                sizes[role].extend(s[2] for s in samples[role])
    return pairs, {role: float(np.median(values)) if values else 0.0 for role, values in sizes.items()}


def fit(results):
    pairs, sizes = summarise(results)
    eyes = {}
    for role in ROLES:
        if pairs[role]:
            targets, centres = zip(*pairs[role])
            mapper = EyeMapper.fit(centres, targets, sizes[role])
            if mapper:
                eyes[role] = mapper
        print(f'{role}: {len(pairs[role])} of {len(results)} targets usable', flush=True)
    return GazeMapper(eyes) if eyes else None


def validate(mapper, results):
    """Mean error in screen points per eye and for the fused gaze, on targets the fit never saw."""
    pairs, sizes = summarise(results)
    errors = {}
    for role, eye in mapper.eyes.items():
        if pairs[role]:
            errors[role] = float(np.mean([np.linalg.norm(eye(centre) - target) for target, centre in pairs[role]]))
    if not errors:
        return None, {}
    mapper.set_weights(errors)
    fused = []
    for target, _ in pairs[next(iter(mapper.eyes))]:
        observed = {role: (centre, sizes[role]) for role in mapper.eyes for t, centre in pairs[role] if t == target}
        point = mapper(observed)
        if point is not None:
            fused.append(np.linalg.norm(point - target))
    return (float(np.mean(fused)) if fused else None), errors


def calibrate(pupils, screen, eyes):
    calibrator = Calibrator(pupils, screen)
    calibrator.open()
    try:
        if not calibrator.align(eyes):
            return None, None, {}
        calibrator.message(['Keep your head still and follow the dot with your eyes.', 'Esc cancels.'], 3)
        mapper = fit(calibrator.run_targets(CALIBRATE_AT))
        if mapper is None:
            return None, None, {}
        fused, errors = validate(mapper, calibrator.run_targets(VALIDATE_AT))
    finally:
        calibrator.close()
    return mapper, fused, errors


def save(mapper, fused, errors, screen):
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(dict(created=datetime.now().isoformat(timespec='seconds'), screen=screen,
                                           validation_error=fused, eye_errors=errors, eyes=mapper.to_dict()), indent=1))


def load():
    data = json.loads(CALIBRATION.read_text())
    return GazeMapper.from_dict(data['eyes']), data


class MouseLoop:
    def __init__(self, args, mapper, pupils, screen, eyes):
        self.args = args
        self.eyes = eyes
        self.mapper = mapper
        self.pupils = pupils
        self.screen = np.array(screen, float) - 1
        self.filter = OneEuroFilter(args.min_cutoff, args.beta)
        self.blink = BlinkClicker()
        self.offset = np.zeros(2)
        self.posted = deque(maxlen=60)
        self.point = np.array(cursor.position(), float)
        self.paused = False
        self.hand_until = 0.0
        self.hand_position = None
        self.hold_until = 0.0
        self.drift_since = None
        self.drift_samples = []
        self.chord_since = None
        self.frames = 0
        self.lost = 0
        self.clicks = 0

    def post(self, point):
        self.point = np.clip(point, 0, self.screen)
        cursor.move(*self.point)
        self.posted.append(self.point.copy())

    def hand_moved(self, now):
        """The real mouse moved if the cursor is somewhere we did not put it."""
        position = np.array(cursor.position(), float)
        if now < self.hand_until:
            if self.hand_position is not None and np.linalg.norm(position - self.hand_position) > 1:
                self.hand_until = now + HAND_PAUSE_S
            self.hand_position = position
            return True
        if self.hand_position is not None:
            print('Eye control resumed', flush=True)
            self.hand_position = None
            self.posted.clear()
            self.posted.append(position)
            self.resync(position)
        if self.posted and all(np.linalg.norm(position - p) > HAND_TOLERANCE for p in self.posted):
            print('Real mouse moved: eye control paused', flush=True)
            self.hand_until = now + HAND_PAUSE_S
            self.hand_position = position
            return True
        return False

    def resync(self, position):
        self.point = np.asarray(position, float)
        self.filter.reset()
        self.blink.reset()

    def hotkeys(self, now, gaze):
        """Returns False to quit. Chords are polled, so they work whichever app is in front."""
        held = cursor.modifiers()
        if PAUSE_KEYS <= held:
            self.drift_since = None
            if self.chord_since is None:
                self.chord_since = now
            elif now - self.chord_since >= QUIT_HOLD_S:
                return False
            return True
        if self.chord_since is not None:
            self.chord_since = None
            self.paused = not self.paused
            cursor.sound('Pop')
            print('Paused' if self.paused else 'Resumed', flush=True)
            self.resync(cursor.position())
        if held == DRIFT_KEYS and not self.paused:
            if self.drift_since is None:
                self.drift_since, self.drift_samples = now, []
                print('Look at the cursor, then release Ctrl+Option', flush=True)
            elif gaze is not None:
                self.drift_samples.append((now, gaze))
        elif self.drift_since is not None:
            self.finish_drift(now)
        return True

    def finish_drift(self, now):
        since, self.drift_since = self.drift_since, None
        if now - since < DRIFT_MIN_HOLD_S:
            return
        # The first moments of the hold are the eyes travelling to the frozen cursor.
        settled = [gaze for t, gaze in self.drift_samples if t >= since + DRIFT_MIN_HOLD_S/2]
        if len(settled) < 10:
            print('Drift not corrected: too few pupil samples', flush=True)
            return
        self.offset = self.point - robust_centre(settled)
        self.filter.reset(self.point)
        self.blink.reset()
        cursor.sound('Pop')
        print(f'Drift corrected: offset ({self.offset[0]:+.0f}, {self.offset[1]:+.0f}) points', flush=True)

    def tick(self, now):
        gaze = self.mapper(self.pupils.fresh(now))
        if not self.hotkeys(now, gaze):
            return False
        if self.hand_moved(now) or self.paused or self.chord_since is not None or self.drift_since is not None:
            return True
        self.frames += 1
        if gaze is None:
            self.lost += 1
            self.blink.closed(now)
            return True
        smoothed = self.filter(now, gaze + self.offset)
        click = self.blink.opened(now, smoothed)
        if self.blink.closed_at is not None or now < self.hold_until:
            return True  # the lids are still reopening, or a click just landed
        if click and not self.args.no_click:
            kind, anchor = click
            self.post(anchor)
            cursor.click(*self.point, right=kind == 'right')
            cursor.sound('Tink')
            self.clicks += 1
            self.hold_until = now + CLICK_HOLD_S
            self.filter.reset(self.point)
            print(f'{kind} click at ({self.point[0]:.0f}, {self.point[1]:.0f})', flush=True)
        elif np.linalg.norm(smoothed - self.point) >= .5:
            self.post(smoothed)
        return True

    def report(self):
        lost = self.lost/self.frames if self.frames else 0
        print(f'no gaze {lost:.0%} of ticks, {self.clicks} clicks', flush=True)
        for role, (seen, found) in self.pupils.take_stats().items():
            line = f'  {role}: {seen} frames, pupil found {found/max(seen, 1):.0%}'
            sample, mapper = self.pupils.latest.get(role), self.mapper.eyes.get(role)
            if sample and mapper:
                x, y = sample[1]
                line += (f', last ({x:.0f}, {y:.0f}) size {sample[2]:.0f}; calibrated x {mapper.low[0]:.0f}-'
                         f'{mapper.high[0]:.0f} y {mapper.low[1]:.0f}-{mapper.high[1]:.0f} '
                         f'size {mapper.pupil_size:.0f}; accepted {mapper.plausible(sample[1], sample[2])}')
            print(line, flush=True)
        if lost > .9:  # keep what the cameras saw, to tell a removed headset from a tracking fault
            for eye in self.eyes:
                frame = eye.stream.snapshot()[0]
                if frame is not None:
                    cv2.imwrite(str(CALIBRATION.parent/f'lost_{eye.role}.png'), frame)
        self.frames = self.lost = 0

    def run(self):
        print(__doc__.split('\n\n')[-1], flush=True)
        report = time.perf_counter() + REPORT_S
        while True:
            now = time.perf_counter()
            if not self.tick(now):
                print('Quit hotkey', flush=True)
                return
            if now >= report:
                self.report()
                report = now + REPORT_S
            time.sleep(.008)


def wait_for_cameras(eyes, seconds=12.0):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if all(eye.stream.state == 'streaming' for eye in eyes):
            return True
        time.sleep(.1)
    for eye in eyes:
        if eye.stream.state != 'streaming':
            print(f'{eye.role}: {eye.stream.state}: {eye.stream.message}', file=sys.stderr)
    return any(eye.stream.state == 'streaming' for eye in eyes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--left', type=int, default=DEFAULT_LEFT)
    p.add_argument('--right', type=int, default=DEFAULT_RIGHT)
    p.add_argument('--device-name', default=DEFAULT_DEVICE)
    p.add_argument('--reuse', action='store_true', help='Skip calibration and use the last one')
    p.add_argument('--no-click', action='store_true', help='Move the cursor but never click')
    p.add_argument('--calibrate-only', action='store_true', help='Calibrate, report accuracy and exit')
    p.add_argument('--contrast', action='store_true', help='Contrast enhancement (CLAHE) for the tracker')
    p.add_argument('--min-cutoff', type=float, default=.5, help='Smoothing while fixating: lower is steadier')
    p.add_argument('--beta', type=float, default=.005, help='Smoothing during saccades: higher is faster')
    args = p.parse_args(argv)
    args.exposure = args.brightness = args.gain = args.gamma = None
    args.focal = 280.0

    if not cursor.has_access() and not args.calibrate_only:
        cursor.request_access()
        print('Moving the cursor needs Accessibility access. In System Settings > Privacy & Security > '
              'Accessibility, turn on your terminal app, then run this again.', file=sys.stderr)
        return 1
    if not request_camera_access():
        print('Camera access is not granted. Allow your terminal app in System Settings > '
              'Privacy & Security > Camera, then run this again.', file=sys.stderr)
        return 1

    pupils = Pupils()
    settings = Settings(contrast=args.contrast)
    eyes = [Eye(role, index, args, settings, Recorder(ROOT/'recordings'))
            for role, index in zip(ROLES, (args.left, args.right))]
    for eye in eyes:
        eye.tracker.sink = pupils
        eye.tracker.track_model = False  # the mouse does not use it, and failed rebuilds stall the tracker
        eye.start()
    try:
        print('Opening cameras...', flush=True)
        if not wait_for_cameras(eyes):
            print('No camera is streaming.', file=sys.stderr)
            return 1
        screen = cursor.screen_size()
        if args.reuse:
            mapper, data = load()
            print(f'Using calibration from {data["created"]} (error {data["validation_error"]})', flush=True)
        else:
            try:
                mapper, fused, errors = calibrate(pupils, screen, eyes)
            except Aborted:
                print('Calibration cancelled', flush=True)
                return 1
            if mapper is None:
                print('Calibration failed: pupils not locked, too few samples, or they barely moved.', file=sys.stderr)
                return 1
            save(mapper, fused, errors, screen)
            shown = {role: round(error) for role, error in errors.items()}
            accuracy = 'unavailable (no validation samples)' if fused is None else f'{fused:.0f} points'
            print(f'Validation error: {accuracy} on a {screen[0]}x{screen[1]} screen (per eye {shown}); '
                  f'eyes in use: {", ".join(mapper.eyes)}', flush=True)
        if args.calibrate_only:
            return 0
        MouseLoop(args, mapper, pupils, screen, eyes).run()
        return 0
    finally:
        for eye in eyes:
            eye.stop()
        for eye in eyes:
            eye.join()


if __name__ == '__main__':
    sys.exit(main())
