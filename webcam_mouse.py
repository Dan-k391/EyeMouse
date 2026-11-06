"""Webcam-only mouse for macOS: no headset, just the built-in camera.

    ./run.sh webcam                # iris and head steer the cursor
    ./run.sh webcam --mode head    # your nose steers it: steadier, but you move your head
    ./run.sh webcam --reuse        # skip calibration, use the last one
    ./run.sh webcam --no-click     # move the cursor only

The controls match the eye mouse: long blink to click, the real mouse always wins,
Ctrl+Option drift correction, Ctrl+Option+Cmd to pause or quit.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np

from camera import BACKEND, DEFAULT_DEVICE, device_names, request_camera_access
import cursor
from eyemouse import CALIBRATE_AT, VALIDATE_AT, Aborted, Calibrator, MouseLoop, Pupils
from face_gaze import MODES, FaceMapper, face_features
from gaze_mapper import robust_centre

ROOT = Path(__file__).parent
MODEL = ROOT/'models'/'face_landmarker.task'
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/'
             'float16/latest/face_landmarker.task')
CALIBRATION = ROOT/'calibrations'/'webcam.json'
ROLE = 'face'
WIDTH, HEIGHT = 1280, 720
MAX_AGE_S = .15            # the webcam delivers about 30 FPS
FACE_LOST_S = .3
BLINK_SCORE = .5
MIN_TARGET_SAMPLES = 15
CHUNK = 5                  # samples per fitted row: keeps noise down, and head turns as several rows
HEAD_SETTLE_S = 1.2
HEAD_TURNS = ('left', 'right', 'up', 'down')
PROMPTS = dict(gaze='Keep your head still and follow the dot with your eyes.',
               head='Point your nose at the dot. Your eyes can stay on it too.')
FACE_WAIT_S = 30.0


class FaceSamples(Pupils):
    def __init__(self):
        super().__init__(roles=(ROLE,), max_age=MAX_AGE_S)
        self.face_at = 0.0

    def visible(self, now):
        """Only closed eyes on a visible face can be a blink; a lost face is not a click."""
        return now - self.face_at < FACE_LOST_S


class FaceTracker(threading.Thread):
    """Reads the webcam and publishes gaze features. Closed eyes publish nothing, like a hidden pupil."""

    def __init__(self, index, samples):
        super().__init__(daemon=True, name='face-tracker')
        self.index = index
        self.samples = samples
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.fps = 0.0

    def stop(self):
        self.stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.index, BACKEND)
        if not cap.isOpened():
            self.error = f'Could not open camera index {self.index}'
            self.ready.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                               running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True)
        start = time.perf_counter()
        last_ms = -1
        frames = 0
        try:
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                self.ready.set()
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    now = time.perf_counter()
                    if not ok:
                        time.sleep(.01)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # VIDEO mode requires strictly increasing timestamps.
                    last_ms = max(last_ms + 1, int((now - start)*1000))
                    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
                    self.publish(now, result, frame.shape)
                    frames += 1
                    self.fps = frames/max(time.perf_counter() - start, 1e-3)
        except Exception as error:  # surface it on the main thread instead of a silent dead cursor
            self.error = f'Face tracking stopped: {error}'
            self.ready.set()
        finally:
            cap.release()

    def publish(self, now, result, shape):
        if not result.face_landmarks:
            self.samples(ROLE, now, None)
            return
        self.samples.face_at = time.perf_counter()
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        closed = min(scores.get('eyeBlinkLeft', 0), scores.get('eyeBlinkRight', 0)) > BLINK_SCORE
        height, width = shape[:2]
        points = np.array([(p.x*width, p.y*height) for p in result.face_landmarks[0]])
        features = None if closed else face_features(points, np.asarray(result.facial_transformation_matrixes[0]))
        self.samples(ROLE, now, None if features is None else dict(center=features, axes=(0.0, 0.0)))


def fit_rows(results):
    """Short-chunk medians of the features, each paired with its target."""
    rows, targets = [], []
    for screen_point, captured in results:
        values = [sample[1] for sample in captured[ROLE]]
        for i in range(0, len(values) - CHUNK + 1, CHUNK):
            rows.append(robust_centre(values[i:i + CHUNK]))
            targets.append(screen_point)
    return rows, targets


def validation_error(mapper, results):
    errors = []
    for screen_point, captured in results:
        values = [sample[1] for sample in captured[ROLE]]
        if len(values) >= MIN_TARGET_SAMPLES:
            errors.append(np.linalg.norm(mapper.point(robust_centre(values)) - screen_point))
    return float(np.mean(errors)) if errors else None


def wait_for_face(calibrator, samples):
    end = time.perf_counter() + FACE_WAIT_S
    while time.perf_counter() < end:
        if samples.visible(time.perf_counter()):
            return True
        calibrator.message(['Looking for your face. Sit facing the camera.'], 1)
    return False


def calibrate(mode, samples, screen):
    calibrator = Calibrator(samples, screen)
    calibrator.min_samples = MIN_TARGET_SAMPLES
    calibrator.open()
    try:
        if not wait_for_face(calibrator, samples):
            print('No face seen. Check the lighting and that the camera can see you.', file=sys.stderr)
            return None, None
        calibrator.message([PROMPTS[mode], 'Esc cancels.'], 3)
        results = calibrator.run_targets(CALIBRATE_AT)
        if mode == 'gaze':  # teach the fit that a turning head alone must not move the cursor
            calibrator.message(['Now keep your eyes on the dot while you turn your head a little.'], 3)
            for turn in HEAD_TURNS:
                calibrator.prompt = f'Eyes on the dot. Turn your head a little {turn}.'
                results.append(calibrator.target((.5, .5), settle=HEAD_SETTLE_S))
            calibrator.prompt = ''
            calibrator.message(['Last few dots: look at them normally.'], 2)
        mapper = FaceMapper.fit(mode, *fit_rows(results))
        if mapper is None:
            print('Calibration failed: too few face samples.', file=sys.stderr)
            return None, None
        return mapper, validation_error(mapper, calibrator.run_targets(VALIDATE_AT))
    finally:
        calibrator.close()


class WebcamLoop(MouseLoop):
    def __init__(self, args, mapper, samples, screen, tracker):
        super().__init__(args, mapper, samples, screen, [])
        self.tracker = tracker

    def report(self):
        lost = self.lost/self.frames if self.frames else 0
        seen, found = self.pupils.take_stats()[ROLE]
        print(f'no gaze {lost:.0%} of ticks, {self.clicks} clicks; {seen} frames at {self.tracker.fps:.0f} FPS, '
              f'eyes open and face found {found/max(seen, 1):.0%}', flush=True)
        self.frames = self.lost = 0
        if self.tracker.error:
            raise RuntimeError(self.tracker.error)


def find_camera():
    """First camera that is not a headset eye camera."""
    for index, name in enumerate(device_names()):
        if name != DEFAULT_DEVICE:
            print(f'Using camera {index}: {name}', flush=True)
            return index
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', type=int, help='Camera index (default: first non-headset camera)')
    p.add_argument('--mode', choices=sorted(MODES), default='gaze')
    p.add_argument('--reuse', action='store_true', help='Skip calibration and use the last one')
    p.add_argument('--no-click', action='store_true', help='Move the cursor but never click')
    p.add_argument('--calibrate-only', action='store_true', help='Calibrate, report accuracy and exit')
    p.add_argument('--min-cutoff', type=float, default=.4, help='Smoothing while still: lower is steadier')
    p.add_argument('--beta', type=float, default=.004, help='Smoothing while moving: higher is faster')
    args = p.parse_args(argv)

    if not MODEL.exists():
        print(f'Face model missing. Download it to {MODEL}:\n  curl -o {MODEL} {MODEL_URL}', file=sys.stderr)
        return 1
    if not cursor.has_access() and not args.calibrate_only:
        cursor.request_access()
        print('Moving the cursor needs Accessibility access. In System Settings > Privacy & Security > '
              'Accessibility, turn on your terminal app, then run this again.', file=sys.stderr)
        return 1
    if not request_camera_access():
        print('Camera access is not granted. Allow your terminal app in System Settings > '
              'Privacy & Security > Camera, then run this again.', file=sys.stderr)
        return 1
    index = find_camera() if args.camera is None else args.camera
    if index is None:
        print('No webcam found.', file=sys.stderr)
        return 1

    samples = FaceSamples()
    tracker = FaceTracker(index, samples)
    tracker.start()
    try:
        tracker.ready.wait(20)
        if tracker.error or not tracker.ready.is_set():
            print(tracker.error or 'The camera or face model did not start.', file=sys.stderr)
            return 1
        screen = cursor.screen_size()
        if args.reuse:
            data = json.loads(CALIBRATION.read_text())
            mapper = FaceMapper.from_dict(data['mapper'])
            print(f'Using {mapper.mode} calibration from {data["created"]} '
                  f'(error {data["validation_error"]})', flush=True)
        else:
            try:
                mapper, error = calibrate(args.mode, samples, screen)
            except Aborted:
                print('Calibration cancelled', flush=True)
                return 1
            if mapper is None:
                return 1
            CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION.write_text(json.dumps(dict(created=datetime.now().isoformat(timespec='seconds'), screen=screen,
                                                   validation_error=error, mapper=mapper.to_dict()), indent=1))
            accuracy = 'unavailable (no validation samples)' if error is None else f'{error:.0f} points'
            print(f'Validation error: {accuracy} on a {screen[0]}x{screen[1]} screen ({args.mode} mode)', flush=True)
        if args.calibrate_only:
            return 0
        WebcamLoop(args, mapper, samples, screen, tracker).run()
        return 0
    finally:
        tracker.stop()
        tracker.join(5)


if __name__ == '__main__':
    sys.exit(main())
