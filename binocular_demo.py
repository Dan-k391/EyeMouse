"""Binocular eye-tracking demo for two GC0308 USB cameras on Windows.

2D pupil fitting and the 3D eye model are ported from
https://github.com/YutaItoh/3D-Eye-Tracker (MIT); see pupil_fitter.py and eye_model.py.

    run.bat                      # interactive demo
    run.bat --selftest 10        # headless check that both cameras stream
"""
import argparse
from collections import deque
import csv
from datetime import datetime
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from camera import FPS, HEIGHT, WIDTH, CameraStream, FpsMeter
from eye_model import EyeModelUpdater, ellipse_from_box, project_point
from pupil_fitter import PupilFitter

ROOT = Path(__file__).parent
WINDOW = 'Binocular Eye Tracker  (h: help)'
MIN_ROI = 24

GREEN, ORANGE, RED, CYAN, WHITE, GREY = (80, 255, 80), (0, 165, 255), (60, 60, 255), (255, 200, 60), (235, 235, 235), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Settings:
    def __init__(self, contrast=False, darkness_offset=10):
        self.contrast = contrast
        self.darkness_offset = darkness_offset


class Recorder:
    COLUMNS = ['host_time_s', 'unix_time', 'eye', 'camera_index', 'frame_seq', 'capture_time_s',
               'detected', 'pupil_x', 'pupil_y', 'pupil_axis1_px', 'pupil_axis2_px', 'pupil_angle_deg',
               'roi_x', 'roi_y', 'roi_w', 'roi_h', 'contrast', 'model_built', 'model_samples',
               'reliability', 'gaze_x', 'gaze_y', 'gaze_z', 'pupil3d_x_mm', 'pupil3d_y_mm', 'pupil3d_z_mm',
               'pupil_diameter_mm', 'eye_x_mm', 'eye_y_mm', 'eye_z_mm', 'capture_fps', 'tracking_fps']

    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.file = None
        self.writer = None
        self.path = None
        self.rows = 0
        self.started = time.perf_counter()

    @property
    def active(self):
        return self.file is not None

    def start(self):
        with self.lock:
            if self.file:
                return self.path
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory/f'eyetrack_{datetime.now():%Y%m%d_%H%M%S}.csv'
            self.file = open(self.path, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.COLUMNS)
            self.rows = 0
            self.started = time.perf_counter()
            return self.path

    def stop(self):
        with self.lock:
            if self.file:
                self.file.close()
            self.file = self.writer = None
            return self.path

    def write(self, row):
        with self.lock:
            if self.writer:
                self.writer.writerow([row.get(c, '') for c in self.COLUMNS])
                self.rows += 1


def fmt(value, digits=4):
    return '' if value is None else round(float(value), digits)


class EyeTracker(threading.Thread):
    """Consumes the newest frame of one camera; tracking FPS is measured here, separately."""

    def __init__(self, stream, settings, recorder, focal):
        super().__init__(daemon=True, name=f'tracker-{stream.role}')
        self.stream = stream
        self.settings = settings
        self.recorder = recorder
        self.fitter = PupilFitter(WIDTH)
        self.model = EyeModelUpdater(focal, (WIDTH, HEIGHT))
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.lock = threading.Lock()
        self.roi = None
        self.meter = FpsMeter()
        self.proc_ms = deque(maxlen=120)
        self.detections = deque(maxlen=240)
        self.skipped = 0
        self.processed = 0
        self.result = None
        self.stop_event = threading.Event()
        self.reset_model = threading.Event()
        self.more_samples = threading.Event()

    def set_roi(self, roi):
        with self.lock:
            self.roi = roi

    def tracking_fps(self):
        return self.meter.fps()

    def detection_rate(self):
        return sum(self.detections)/len(self.detections) if self.detections else 0.0

    def stop(self):
        self.stop_event.set()

    def run(self):
        last_seq = 0
        while not self.stop_event.is_set():
            frame, seq, captured = self.stream.wait_frame(last_seq, 0.25)
            if frame is None or seq == last_seq:
                if self.stream.state != 'streaming':
                    self.result = None
                continue
            if last_seq and seq > last_seq + 1:
                self.skipped += seq - last_seq - 1
            last_seq = seq
            if self.reset_model.is_set():
                self.reset_model.clear()
                self.model.reset()
            if self.more_samples.is_set():
                self.more_samples.clear()
                self.model.add_more_samples(10)
            start = time.perf_counter()
            with self.lock:
                roi = self.roi
            contrast = self.settings.contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if contrast:
                gray = self.clahe.apply(gray)
            self.fitter.darkest_offset_low = self.settings.darkness_offset
            self.fitter.darkest_offset_high = self.settings.darkness_offset + 10
            pupil = self.fitter.fit(gray, roi)
            model = None
            if pupil:
                ellipse = ellipse_from_box((pupil['center'], pupil['axes'], pupil['angle']), (WIDTH, HEIGHT))
                model = self.model.update(ellipse)
            now = time.perf_counter()
            self.proc_ms.append((time.perf_counter() - start)*1000)
            self.meter.tick(now)
            self.detections.append(pupil is not None)
            self.processed += 1
            self.result = dict(frame=frame, gray=gray if contrast else None, seq=seq, captured=captured,
                               pupil=pupil, model=model, roi=roi, model_built=self.model.built,
                               model_count=self.model.count, model_max=self.model.max_count)
            if self.recorder.active:
                self.recorder.write(self._row(seq, captured, pupil, model, roi, contrast))

    def _row(self, seq, captured, pupil, model, roi, contrast):
        row = dict(host_time_s=fmt(time.perf_counter() - self.recorder.started, 6), unix_time=f'{time.time():.6f}',
                   eye=self.stream.role, camera_index=self.stream.index, frame_seq=seq,
                   capture_time_s=fmt(captured - self.recorder.started, 6), detected=int(pupil is not None),
                   contrast=int(contrast), model_built=int(self.model.built),
                   model_samples=f'{self.model.count}/{self.model.max_count}',
                   capture_fps=fmt(self.stream.capture_fps(), 1), tracking_fps=fmt(self.tracking_fps(), 1))
        if roi:
            row.update(roi_x=roi[0], roi_y=roi[1], roi_w=roi[2], roi_h=roi[3])
        if pupil:
            row.update(pupil_x=fmt(pupil['center'][0], 2), pupil_y=fmt(pupil['center'][1], 2),
                       pupil_axis1_px=fmt(pupil['axes'][0], 2), pupil_axis2_px=fmt(pupil['axes'][1], 2),
                       pupil_angle_deg=fmt(pupil['angle'], 2))
        if model and model.get('reliability') is not None:
            row['reliability'] = fmt(model['reliability'], 3)
        if model and model.get('gaze') is not None:
            g, c, e = model['gaze'], model['pupil_centre'], model['eye_centre']
            row.update(gaze_x=fmt(g[0]), gaze_y=fmt(g[1]), gaze_z=fmt(g[2]),
                       pupil3d_x_mm=fmt(c[0], 3), pupil3d_y_mm=fmt(c[1], 3), pupil3d_z_mm=fmt(c[2], 3),
                       pupil_diameter_mm=fmt(2*model['pupil_radius'], 3),
                       eye_x_mm=fmt(e[0], 3), eye_y_mm=fmt(e[1], 3), eye_z_mm=fmt(e[2], 3))
        return row


class Eye:
    def __init__(self, role, index, args, settings, recorder):
        controls = dict(exposure=args.exposure, brightness=args.brightness, gain=args.gain, gamma=args.gamma)
        self.role = role
        self.stream = CameraStream(role, index, rotate180=(role == 'left'),
                                   device_name=args.device_name, controls=controls)
        self.tracker = EyeTracker(self.stream, settings, recorder, args.focal)

    def start(self):
        self.stream.start()
        self.tracker.start()

    def stop(self):
        self.tracker.stop()
        self.stream.stop()

    def join(self, timeout=10.0):  # opening holds the device for ~4 s
        self.stream.join(timeout)
        self.tracker.join(timeout)


# -- drawing --------------------------------------------------------------------
class Layout:
    def __init__(self, scale):
        self.scale = scale
        self.pw, self.ph = WIDTH*scale, HEIGHT*scale
        self.gap = 8
        self.header = 30
        self.status = 4*22 + 8
        self.footer = 62
        self.width = 2*self.pw + self.gap
        self.height = self.header + self.ph + self.status + self.footer

    def panel_x(self, slot):
        return slot*(self.pw + self.gap)


def text(img, s, org, color=WHITE, size=0.5, thick=1):
    # Same-weight drop shadow: OpenCV 5 widens glyph advance with thickness, so a
    # heavier outline pass would not line up with the text.
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, size, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, size, color, thick, cv2.LINE_AA)


def wrap(s, width):
    words, lines, line = s.split(), [], ''
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f'{line} {w}'.strip()
    return lines + [line] if line else lines


def to_screen(point, scale):
    return int(round(point[0]*scale)), int(round(point[1]*scale))


def draw_3d(panel, model, focal, scale):
    half = np.array([WIDTH/2, HEIGHT/2])
    eye_c, eye_r = model['eye_centre'], model['eye_radius']
    centre_px = project_point(eye_c, focal) + half
    radius_px = focal*eye_r/eye_c[2]
    color = GREEN if model['reliable'] else GREY
    cv2.circle(panel, to_screen(centre_px, scale), max(1, int(radius_px*scale)), CYAN, 1, cv2.LINE_AA)
    cv2.circle(panel, to_screen(centre_px, scale), 3, CYAN, 1, cv2.LINE_AA)
    start = project_point(model['pupil_centre'], focal) + half
    end = project_point(model['pupil_centre'] + 10*model['gaze'], focal) + half  # 10 mm, as upstream
    cv2.line(panel, to_screen(centre_px, scale), to_screen(start, scale), CYAN, 1, cv2.LINE_AA)
    cv2.line(panel, to_screen(start, scale), to_screen(end, scale), color, 2, cv2.LINE_AA)


def draw_panel(canvas, layout, slot, eye, focal, drag_rect, now):
    x0, y0, s = layout.panel_x(slot), layout.header, layout.scale
    stream, tracker = eye.stream, eye.tracker
    rotated = ' (rotated 180)' if stream.rotate180 else ''
    text(canvas, f'{eye.role.upper()}  camera index {stream.index}{rotated}', (x0 + 6, 21), WHITE, 0.6, 1)
    panel = canvas[y0:y0 + layout.ph, x0:x0 + layout.pw]
    result = tracker.result
    if stream.state == 'streaming' and result is not None:
        base = result['frame'] if result['gray'] is None else cv2.cvtColor(result['gray'], cv2.COLOR_GRAY2BGR)
        panel[:] = cv2.resize(base, (layout.pw, layout.ph), interpolation=cv2.INTER_LINEAR)
        roi = result['roi']
        if roi:
            cv2.rectangle(panel, to_screen(roi[:2], s), to_screen((roi[0]+roi[2], roi[1]+roi[3]), s), ORANGE, 1)
        pupil = result['pupil']
        if pupil:
            cx, cy = pupil['center']
            cv2.ellipse(panel, ((cx*s, cy*s), (pupil['axes'][0]*s, pupil['axes'][1]*s), pupil['angle']),
                        GREEN, 2, cv2.LINE_AA)
            cv2.drawMarker(panel, to_screen((cx, cy), s), GREEN, cv2.MARKER_CROSS, 14, 1)
        model = result['model']
        if model and model.get('gaze') is not None:
            draw_3d(panel, model, focal, s)
        elif not result['model_built']:
            text(panel, f'3D model: {result["model_count"]}/{result["model_max"]} samples - look around slowly',
                 (8, layout.ph - 12), CYAN, 0.5)
        stall = stream.stall_seconds()
        if stall > 0.5:
            panel[:] = (panel*0.4).astype(np.uint8)
            text(panel, f'NO FRAMES FOR {stall:.1f} s', (20, layout.ph//2 - 10), RED, 0.9, 2)
            text(panel, 'Check the USB cable/hub; press k to reconnect.', (20, layout.ph//2 + 20), WHITE, 0.55)
        for i, warning in enumerate(stream.warnings):
            text(panel, 'WARNING: ' + warning, (8, 22 + 20*i), ORANGE, 0.45)
    else:
        panel[:] = (32, 24, 24)
        if stream.state == 'error':
            title, color = 'CAMERA ERROR', RED
        else:
            title, color = 'CONNECTING...', ORANGE
        message = stream.message
        if stream.state == 'opening' and now - stream.opening_since > 10:
            message += f' (no frames after {now - stream.opening_since:.0f} s; try k to reconnect or replug)'
        text(panel, title, (20, 60), color, 1.0, 2)
        for i, line in enumerate(wrap(message, 58)):
            text(panel, line, (20, 100 + 24*i), WHITE, 0.55)
        if stream.state == 'error':
            text(panel, 'Retrying automatically. Other camera keeps running.', (20, layout.ph - 20), GREY, 0.5)
    if drag_rect and drag_rect[0] == slot:
        _, (ax, ay), (bx, by) = drag_rect
        cv2.rectangle(panel, to_screen((ax, ay), s), to_screen((bx, by), s), ORANGE, 1, cv2.LINE_AA)

    # status lines
    y = layout.header + layout.ph + 20
    cap_fps = stream.capture_fps()
    fmt_ = stream.format
    cap_color = GREEN if cap_fps >= 0.8*FPS else (ORANGE if cap_fps > 0 else RED)
    fmt_text = f'{fmt_.get("fourcc", "?")} {fmt_.get("width", "?")}x{fmt_.get("height", "?")}' if fmt_ else 'no format'
    text(canvas, f'Capture:  {cap_fps:6.1f} fps   {fmt_text}', (x0 + 6, y), cap_color)
    ms = np.mean(tracker.proc_ms) if tracker.proc_ms else 0
    trk_fps = tracker.tracking_fps()
    trk_color = GREEN if trk_fps >= 0.9*cap_fps and trk_fps > 0 else ORANGE
    text(canvas, f'Tracking: {trk_fps:6.1f} fps   {ms:4.1f} ms/frame   skipped {tracker.skipped}', (x0 + 6, y + 22), trk_color)
    pupil = result['pupil'] if result else None
    detail = (f'({pupil["center"][0]:.1f}, {pupil["center"][1]:.1f})  {pupil["axes"][0]:.0f}x{pupil["axes"][1]:.0f} px'
              if pupil else 'not found')
    roi_text = 'eye region set' if tracker.roi else 'full frame (drag to select eye)'
    text(canvas, f'Pupil: {tracker.detection_rate()*100:3.0f}% detected  {detail}  |  {roi_text}', (x0 + 6, y + 44))
    model = result['model'] if result else None
    if model and model.get('gaze') is not None:
        g = model['gaze']
        line = (f'3D: reliability {model["reliability"]:.2f}  gaze ({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f})  '
                f'pupil {2*model["pupil_radius"]:.1f} mm')
        text(canvas, line, (x0 + 6, y + 66), GREEN if model['reliable'] else ORANGE)
    elif result and result['model_built']:
        text(canvas, '3D: model built; waiting for a pupil', (x0 + 6, y + 66), GREY)
    else:
        count = result['model_count'] if result else 0
        maximum = result['model_max'] if result else EyeModelUpdater.MAX_COUNT_DEFAULT
        text(canvas, f'3D: collecting pupil samples {count}/{maximum}', (x0 + 6, y + 66), CYAN)


HELP = ['Drag on a feed: select eye region   Right-click: clear region',
        'c: contrast enhancement   [ / ]: darkness threshold   s: start/stop CSV recording',
        'r: reset 3D eye models   p: +10 model samples   x: swap left/right   k: reconnect   q/Esc: quit']


def render(layout, eyes, settings, recorder, focal, drag_rect, flash, show_help):
    now = time.perf_counter()
    canvas = np.full((layout.height, layout.width, 3), 18, np.uint8)
    for slot, eye in enumerate(eyes):
        draw_panel(canvas, layout, slot, eye, focal, drag_rect, now)
    y = layout.height - layout.footer + 18
    if recorder.active:
        rec = f'REC  {recorder.path.name}  {recorder.rows} rows'
        cv2.circle(canvas, (14, y - 5), 6, RED, -1)
        text(canvas, rec, (26, y), RED)
    else:
        text(canvas, 'Not recording (s to start)', (8, y), GREY)
    text(canvas, f'Contrast enhancement: {"ON" if settings.contrast else "off"}    '
                 f'Darkness threshold offset: {settings.darkness_offset}', (420, y),
         CYAN if settings.contrast else WHITE)
    if flash and now < flash[1]:
        text(canvas, flash[0], (8, y + 22), ORANGE)
    else:
        text(canvas, HELP[0] if not show_help else '', (8, y + 22), GREY, 0.45)
    if show_help:
        box = canvas[layout.header + 40: layout.header + 40 + 30*len(HELP) + 20, 20:layout.width - 20]
        box[:] = (box*0.25).astype(np.uint8)
        for i, line in enumerate(HELP):
            text(canvas, line, (34, layout.header + 70 + 30*i), WHITE, 0.6)
    return canvas


# -- app --------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.args = args
        self.settings = Settings(contrast=args.contrast)
        self.recorder = Recorder(args.record_dir)
        self.layout = Layout(args.scale)
        self.indices = [args.left, args.right]
        self.eyes = []
        self.drag = None
        self.flash = None
        self.show_help = False

    def start_eyes(self):
        self.eyes = [Eye('left', self.indices[0], self.args, self.settings, self.recorder),
                     Eye('right', self.indices[1], self.args, self.settings, self.recorder)]
        for eye in self.eyes:
            eye.start()

    def stop_eyes(self):
        for eye in self.eyes:
            eye.stop()
        for eye in self.eyes:
            eye.join()

    def say(self, message, seconds=3.0):
        self.flash = (message, time.perf_counter() + seconds)
        print(message, flush=True)

    def frame_point(self, x, y):
        L = self.layout
        slot = 0 if x < L.pw + L.gap//2 else 1
        fx = (x - L.panel_x(slot))/L.scale
        fy = (y - L.header)/L.scale
        inside = 0 <= fx < WIDTH and 0 <= fy < HEIGHT
        return slot, (float(np.clip(fx, 0, WIDTH - 1)), float(np.clip(fy, 0, HEIGHT - 1))), inside

    def on_mouse(self, event, x, y, flags, param):
        slot, point, inside = self.frame_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.drag = (slot, point, point)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            if slot == self.drag[0]:
                self.drag = (self.drag[0], self.drag[1], point)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            s, (ax, ay), (bx, by) = self.drag
            self.drag = None
            x0, y0 = int(min(ax, bx)), int(min(ay, by))
            w, h = int(abs(bx - ax)), int(abs(by - ay))
            if w < MIN_ROI or h < MIN_ROI:
                self.say(f'Eye region too small (min {MIN_ROI}x{MIN_ROI} camera pixels); drag a larger box.')
                return
            self.eyes[s].tracker.set_roi((x0, y0, w, h))
            self.say(f'{self.eyes[s].role} eye region: x={x0} y={y0} {w}x{h}')
        elif event == cv2.EVENT_RBUTTONDOWN and inside:
            self.eyes[slot].tracker.set_roi(None)
            self.say(f'{self.eyes[slot].role} eye region cleared (searching full frame)')

    def handle_key(self, key):
        if key in (ord('q'), 27):
            return False
        if key == ord('c'):
            self.settings.contrast = not self.settings.contrast
            self.say(f'Contrast enhancement {"ON" if self.settings.contrast else "off"}')
        elif key == ord('['):
            self.settings.darkness_offset = max(2, self.settings.darkness_offset - 2)
        elif key == ord(']'):
            self.settings.darkness_offset = min(80, self.settings.darkness_offset + 2)
        elif key == ord('s'):
            if self.recorder.active:
                path = self.recorder.stop()
                self.say(f'Recording saved: {path} ({self.recorder.rows} rows)', 6)
            else:
                self.say(f'Recording to {self.recorder.start()}')
        elif key == ord('r'):
            for eye in self.eyes:
                eye.tracker.reset_model.set()
            self.say('3D eye models reset; look around slowly to rebuild them')
        elif key == ord('p'):
            for eye in self.eyes:
                eye.tracker.more_samples.set()
            self.say('Collecting 10 more pupil samples per eye')
        elif key == ord('x'):
            self.indices.reverse()
            self.say(f'Swapping: left = index {self.indices[0]}, right = index {self.indices[1]} (reopening)')
            self.stop_eyes()
            self.start_eyes()
        elif key == ord('k'):
            for eye in self.eyes:
                eye.stream.reconnect()
            self.say('Reconnecting cameras...')
        elif key == ord('h'):
            self.show_help = not self.show_help
        return True

    def render(self):
        drag = self.drag
        return render(self.layout, self.eyes, self.settings, self.recorder, self.args.focal,
                      drag, self.flash, self.show_help)

    def run(self):
        self.start_eyes()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKey(15) & 0xFF
                if key != 0xFF and not self.handle_key(key):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.recorder.active:
                print(f'Recording saved: {self.recorder.stop()} ({self.recorder.rows} rows)')
            self.stop_eyes()
            cv2.destroyAllWindows()

    def selftest(self, seconds, snapshot=None):
        """Headless: open both cameras, track, and report measured rates. Exit 0 on success."""
        self.start_eyes()
        if self.args.record:
            print(f'Recording to {self.recorder.start()}')
        samples = {eye.role: dict(capture=[], tracking=[], ms=[]) for eye in self.eyes}
        start = time.perf_counter()
        try:
            while time.perf_counter() - start < seconds:
                time.sleep(1.0)
                elapsed = time.perf_counter() - start
                line = [f't={elapsed:4.1f}s']
                for eye in self.eyes:
                    cap, trk = eye.stream.capture_fps(), eye.tracker.tracking_fps()
                    line.append(f'{eye.role}[{eye.stream.state}] cap {cap:5.1f} trk {trk:5.1f}')
                    if eye.stream.state == 'streaming' and elapsed > 2.5:
                        samples[eye.role]['capture'].append(cap)
                        samples[eye.role]['tracking'].append(trk)
                        if eye.tracker.proc_ms:
                            samples[eye.role]['ms'].append(float(np.mean(eye.tracker.proc_ms)))
                print('  '.join(line), flush=True)
            if snapshot:
                cv2.imwrite(str(snapshot), self.render())
                print(f'Snapshot: {snapshot}')
        finally:
            if self.recorder.active:
                print(f'Recording saved: {self.recorder.stop()} ({self.recorder.rows} rows)')
            report = []
            ok = True
            for eye in self.eyes:
                s = samples[eye.role]
                cap = float(np.mean(s['capture'])) if s['capture'] else 0.0
                trk = float(np.mean(s['tracking'])) if s['tracking'] else 0.0
                ms = float(np.mean(s['ms'])) if s['ms'] else 0.0
                fourcc = eye.stream.format.get('fourcc', '-')
                size = f'{eye.stream.format.get("width", "-")}x{eye.stream.format.get("height", "-")}'
                passed = (eye.stream.state == 'streaming' and fourcc == 'MJPG' and cap >= self.args.min_fps)
                ok &= passed
                report.append(f'{eye.role:5s} index {eye.stream.index}: {"PASS" if passed else "FAIL"}  '
                              f'state={eye.stream.state} format={fourcc} {size}  capture={cap:.1f} fps  '
                              f'tracking={trk:.1f} fps ({ms:.1f} ms/frame)  frames tracked={eye.tracker.processed}  '
                              f'pupil detected={eye.tracker.detection_rate()*100:.0f}%')
                if eye.stream.state != 'streaming':
                    report.append(f'      reason: {eye.stream.message}')
                for w in eye.stream.warnings:
                    report.append(f'      warning: {w}')
            self.stop_eyes()
        print('\nSelf-test (both cameras streaming simultaneously):')
        print('\n'.join(report))
        print('RESULT:', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--left', type=int, default=1, help='DirectShow index of the left-eye camera (rotated 180)')
    p.add_argument('--right', type=int, default=2, help='DirectShow index of the right-eye camera')
    p.add_argument('--device-name', default='USB Camera',
                   help='Required DirectShow device name; "" disables the check')
    p.add_argument('--focal', type=float, default=280.0,
                   help='Focal length in pixels at 320x240 for the 3D model (estimate; default 280)')
    p.add_argument('--contrast', action='store_true', help='Start with contrast enhancement (CLAHE) on')
    p.add_argument('--exposure', type=float, help='Manual exposure (DirectShow log2 seconds, e.g. -8)')
    p.add_argument('--brightness', type=float)
    p.add_argument('--gain', type=float)
    p.add_argument('--gamma', type=float)
    p.add_argument('--scale', type=int, default=2, choices=[1, 2, 3], help='Display zoom')
    p.add_argument('--record-dir', default=str(ROOT/'recordings'))
    p.add_argument('--record', action='store_true', help='Start recording CSV immediately')
    p.add_argument('--selftest', type=float, metavar='SECONDS', help='Headless camera/tracking check')
    p.add_argument('--snapshot', help='With --selftest: save a PNG of the UI at the end')
    p.add_argument('--min-fps', type=float, default=100.0, help='Self-test capture FPS threshold')
    args = p.parse_args(argv)
    if args.left == args.right:
        p.error('--left and --right must be different camera indices')
    app = App(args)
    if args.selftest:
        return app.selftest(args.selftest, args.snapshot)
    if args.record:
        print(f'Recording to {app.recorder.start()}')
    app.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
