"""DirectShow capture thread for one GC0308 eye camera.

Adapted from D:\\eyetrack\\capture.py (device-name guard, image controls). Each
camera reads on its own thread so neither blocks the other at 120 FPS; both stay
in one process because separate processes can enumerate DirectShow devices in a
different order.
"""
from collections import deque
import threading
import time

import cv2

WIDTH, HEIGHT, FPS = 320, 240, 120
EXPECTED_FOURCC = 'MJPG'
OPEN_RETRY_S = 2.0
WARMUP_S = 1.0


def fourcc_name(value):
    value = int(value)
    text = ''.join(chr((value >> 8*i) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.isascii() else f'0x{value:08X}'


def device_names():
    import comtypes
    from pygrabber.dshow_graph import FilterGraph
    comtypes.CoInitialize()
    try:
        return FilterGraph().get_input_devices()
    finally:
        comtypes.CoUninitialize()


def apply_controls(cap, settings):
    """Optional hardware controls (from D:\\eyetrack\\capture.py); None leaves the driver default."""
    result = {}
    for key, prop in [('exposure', cv2.CAP_PROP_EXPOSURE), ('brightness', cv2.CAP_PROP_BRIGHTNESS),
                      ('gain', cv2.CAP_PROP_GAIN), ('gamma', cv2.CAP_PROP_GAMMA)]:
        value = settings.get(key)
        if value is None:
            continue
        if key == 'exposure':
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: manual
        accepted = bool(cap.set(prop, float(value)))
        result[key] = dict(requested=value, accepted=accepted, actual=cap.get(prop))
    return result


class FpsMeter:
    def __init__(self, window=1.0):
        self.window = window
        self.times = deque()

    def tick(self, now):
        self.times.append(now)
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()

    def fps(self, now=None):
        now = time.perf_counter() if now is None else now
        times = tuple(self.times)  # the owning thread appends concurrently
        # Frames arrive in a burst right after opening; wait for a meaningful span.
        if len(times) < 2 or now - times[-1] > self.window or times[-1] - times[0] < 0.25:
            return 0.0
        return (len(times) - 1)/(times[-1] - times[0])


class CameraStream(threading.Thread):
    """Owns one cv2.VideoCapture. Publishes the newest frame; never queues stale ones."""

    def __init__(self, role, index, rotate180=False, device_name='USB Camera', controls=None):
        super().__init__(daemon=True, name=f'camera-{role}')
        self.role = role
        self.index = index
        self.rotate180 = rotate180
        self.device_name = device_name
        self.controls = controls or {}
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.timestamp = 0.0
        self.meter = FpsMeter()
        self.state = 'opening'   # opening | streaming | error | stopped
        self.message = 'Opening camera...'
        self.opening_since = time.perf_counter()
        self.warnings = []
        self.format = {}
        self.control_report = {}
        self.stop_event = threading.Event()
        self.reconnect_event = threading.Event()

    # -- public, thread-safe --------------------------------------------------
    def snapshot(self):
        with self.cond:
            return self.frame, self.seq, self.timestamp

    def wait_frame(self, after_seq, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != after_seq or self.stop_event.is_set(), timeout)
            return self.frame, self.seq, self.timestamp

    def capture_fps(self):
        return self.meter.fps()

    def stall_seconds(self):
        if self.state != 'streaming' or not self.timestamp:
            return 0.0
        return time.perf_counter() - self.timestamp

    def stop(self):
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()

    def reconnect(self):
        self.reconnect_event.set()

    # -- thread ---------------------------------------------------------------
    def _fail(self, message):
        self.state = 'error'
        self.message = message
        self.format = {}
        with self.cond:
            self.frame = None
            self.cond.notify_all()

    def _open(self):
        try:
            names = device_names()
        except Exception as error:  # COM/pygrabber failure: report rather than guess
            self._fail(f'Cannot list DirectShow cameras: {error}')
            return None
        if self.index >= len(names):
            self._fail(f'No camera at index {self.index} (found {len(names)}: {", ".join(names) or "none"}). '
                       'Is it plugged in?')
            return None
        if self.device_name and names[self.index] != self.device_name:
            self._fail(f'Index {self.index} is "{names[self.index]}", not "{self.device_name}". '
                       f'Devices: {", ".join(f"{i}={n}" for i, n in enumerate(names))}')
            return None
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self._fail(f'Could not open index {self.index}. Close other apps using it '
                       '(Camera app, OBS, D:\\eyetrack server) and replug if needed.')
            return None
        # Order matters on the GC0308: size and rate first, MJPG last. Setting MJPG
        # first silently falls back to washed-out YUY2 at 30 FPS.
        self.message = f'Configuring {WIDTH}x{HEIGHT} @ {FPS} FPS {EXPECTED_FOURCC} (takes ~4 s)...'
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*EXPECTED_FOURCC))
        self.control_report = apply_controls(cap, self.controls)
        self.format = dict(fourcc=fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)),
                           width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           fps_property=cap.get(cv2.CAP_PROP_FPS))
        return cap

    def _check_format(self, warmup_fps):
        warnings = []
        f = self.format
        if f.get('fourcc') != EXPECTED_FOURCC:
            warnings.append(f'Format is {f.get("fourcc")}, not MJPG (expect washed-out 30 FPS)')
        if (f.get('width'), f.get('height')) != (WIDTH, HEIGHT):
            warnings.append(f'Size is {f.get("width")}x{f.get("height")}, not {WIDTH}x{HEIGHT}')
        # CAP_PROP_FPS reads back 120 even in the YUY2 fallback, so judge measured FPS.
        if warmup_fps < 0.8*FPS:
            warnings.append(f'Only {warmup_fps:.0f} FPS measured (long exposure in low light, '
                            'USB bandwidth, or format fallback)')
        self.warnings = warnings

    def run(self):
        while not self.stop_event.is_set():
            self.state, self.message, self.warnings = 'opening', f'Opening index {self.index}...', []
            self.opening_since = time.perf_counter()
            cap = self._open()
            if cap is None:
                self.stop_event.wait(OPEN_RETRY_S)
                continue
            self.reconnect_event.clear()
            self.meter = FpsMeter()
            opened_at = time.perf_counter()
            first_frame = None
            warm_count = 0
            failures = 0
            try:
                while not self.stop_event.is_set() and not self.reconnect_event.is_set():
                    ok, frame = cap.read()
                    now = time.perf_counter()
                    if not ok or frame is None:
                        failures += 1
                        last = self.timestamp if first_frame else opened_at
                        if failures >= 10 and now - last > 1.0:
                            self._fail(f'Index {self.index} stopped delivering frames '
                                       '(unplugged or driver error); reopening...')
                            break
                        time.sleep(0.005)
                        continue
                    failures = 0
                    if self.rotate180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.meter.tick(now)
                    if first_frame is None:
                        first_frame = now
                    warm_count += 1
                    if self.state != 'streaming' and now - first_frame >= WARMUP_S:
                        self._check_format((warm_count - 1)/(now - first_frame))
                        self.state, self.message = 'streaming', 'Streaming'
                    with self.cond:
                        self.frame = frame
                        self.seq += 1
                        self.timestamp = now
                        self.cond.notify_all()
                    if self.state == 'opening':
                        self.message = 'Measuring frame rate...'
                if self.state == 'opening' and first_frame is None and not self.stop_event.is_set():
                    self._fail(f'Index {self.index} opened but sent no frames')
            finally:
                cap.release()
            if self.state == 'error':
                self.stop_event.wait(OPEN_RETRY_S)
        self.state, self.message = 'stopped', 'Stopped'
