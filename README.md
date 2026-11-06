# EyeMouse: binocular eye-tracking demo

A Windows and macOS demo for two GC0308 USB eye cameras. It shows both feeds with fitted
pupil ellipses and a per-eye 3D eye model, and can record to CSV. The 2D pupil
fitter and 3D eye model are Python ports of
[YutaItoh/3D-Eye-Tracker](https://github.com/YutaItoh/3D-Eye-Tracker) (MIT). See
[third_party/3D-Eye-Tracker](third_party/3D-Eye-Tracker/NOTICE.md) for the
license and file mapping.

## Launch

```bat
D:\EyeMouse\run.bat
```

The first run creates `.venv` with Python 3.12 if it is missing. Close anything
else using the cameras first (Windows Camera, OBS, the old `D:\eyetrack` server).
Opening takes about 5 seconds: the driver needs ~3 s to apply the format, then
the frame rate is measured for 1 s.

Useful options (they also work with `run.bat`):

| Option | Purpose |
|---|---|
| `--left 1 --right 2` | DirectShow indices (defaults). The left feed is rotated 180°. |
| `--selftest 10` | Headless check: opens both cameras, tracks for 10 s, prints PASS/FAIL. Exit code 0 means pass. |
| `--contrast` | Start with contrast enhancement on. |
| `--record` | Start CSV recording immediately. |
| `--exposure -8` etc. | Optional manual exposure/brightness/gain/gamma. Defaults leave the driver settings alone. |
| `--focal 280` | Focal length in pixels at 320×240 used by the 3D model (an estimate, see below). |
| `--scale 1` | Smaller window. |

On macOS, launch from Terminal.app instead:

```bash
./run.sh
```

The first run creates `.venv` with `python3` and asks for camera permission
(run it from Terminal.app: apps without a camera prompt, such as some embedded
shells, are refused). Three things differ from Windows:

- Camera indices default to `--left 0 --right 1`, because OpenCV numbers cameras
  in `uniqueID` order: the USB cameras sort before the FaceTime camera.
- The two cameras report the same model, so nothing identifies which eye is
  which. If the feeds are on the wrong side, press `x` or swap the indices.
- AVFoundation decodes MJPG itself and offers 320×240 @ 120 FPS as `420v`, so
  the MJPG check and the MJPG property order below do not apply.

## Eye mouse (macOS)

```bash
./run.sh mouse              # calibrate, then control the cursor
./run.sh mouse --reuse      # skip calibration, use the last one
./run.sh mouse --no-click   # move the cursor only
./run.sh mouse --calibrate-only   # measure accuracy and exit
```

Run it from Terminal.app, which needs Camera and Accessibility permission. It
first shows both eye feeds and waits until both pupils are locked. Then follow
13 dots with your eyes while keeping your head still, then 4 more that measure
accuracy on points the fit never saw. The result is printed in screen points and
saved to `calibrations/latest.json`.

| Input | Action |
|---|---|
| Look | Moves the cursor |
| Blink both eyes for 0.35–1.2 s | Left click, at the spot you looked at before the blink |
| Blink both eyes for 1.2–2.5 s | Right click |
| Move the real mouse or trackpad | Pauses eye control until it has been still for 2 s |
| Hold Ctrl+Option, look at the frozen cursor, release | Corrects drift |
| Tap Ctrl+Option+Cmd | Pause or resume |
| Hold Ctrl+Option+Cmd for 2 s | Quit |
| Esc | Cancel during calibration |

Each eye gets a second-order polynomial from pupil position to screen position,
and the two are averaged, weighted by validation error. Either eye alone still
works, and an eye 3× worse than the other is dropped. The 3D eye model is not
used: the cameras stay fixed relative to the eyes, so it adds nothing here, and
it is turned off because failed rebuilds stall the tracker thread.

The headset cannot see your head turn, so turning your head moves the cursor.
Keep your head roughly still and use the drift correction after moving it.
Recalibrate whenever the headset has been taken off, because `--reuse` assumes
the cameras have not moved. Only the main display is supported.

## Webcam mouse (macOS)

No headset needed: this uses the built-in camera and MediaPipe's face landmarks.

```bash
./run.sh webcam                  # iris and head steer the cursor
./run.sh webcam --mode head      # your nose steers it
./run.sh webcam --reuse          # skip calibration, use the last one
./run.sh webcam --no-click       # move the cursor only
./run.sh webcam --calibrate-only # measure accuracy and exit
```

It needs `models/face_landmarker.task`, which is not in git. Download it from
Google's [MediaPipe models](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
into `models/`. Face the camera directly with your face evenly lit.

The controls are the same as the eye mouse. In `gaze` mode, calibration adds a
step where you keep your eyes on a dot and turn your head, which teaches the
mapping to ignore head movement. A webcam sees far less iris detail than a
headset at 30 FPS rather than 120, so expect coarser aim. `head` mode gives up
eye steering for a steadier cursor, so try it if `gaze` mode is too jittery.

Calibration is stored separately in `calibrations/webcam.json`. Run `--reuse`
only if you have not moved the camera or your seat.

## Controls

| Input | Action |
|---|---|
| Drag on a feed | Select that eye's region (the pupil search runs only inside it) |
| Right-click a feed | Clear the region (search the full frame) |
| `c` | Toggle contrast enhancement (CLAHE). The display shows the enhanced image the tracker sees. |
| `[` / `]` | Darkness threshold offset (upstream `darkestPixelL1`; L2 = L1 + 10) |
| `s` | Start/stop CSV recording to `recordings\eyetrack_YYYYMMDD_HHMMSS.csv` |
| `r` | Reset both 3D eye models (upstream key) |
| `p` | Collect 10 more samples for the 3D models (upstream key) |
| `x` | Swap which camera is left/right (reopens both) |
| `k` | Reconnect both cameras |
| `h` | Help overlay |
| `q` / Esc | Quit (an active recording is saved) |

## Capture settings

Each camera opens with DirectShow at 320×240, 120 FPS, MJPG. Width, height and
FPS are set **before** MJPG. On these cameras, setting MJPG first falls back to
washed-out YUY2 at about 30 FPS.

Measured on this PC (both cameras at once, 2026-10-21):

| Property order | Format reported | Measured FPS | Mean brightness (same scene) |
|---|---|---|---|
| W, H, FPS, then MJPG | MJPG | 123.2 / 123.1 | 19.5 / 122.4 |
| MJPG, then W, H, FPS | YUY2 | 30.8 / 30.8 | 84.3 / 206.7 |

`CAP_PROP_FPS` reads back 120 in **both** cases, so the app never trusts it. It
measures the real rate from frame timestamps and warns on the feed if the
format is not MJPG, the size is wrong, or fewer than 96 FPS arrive.

## What is shown

The status lines under each feed separate two rates:

- **Capture FPS**: frames delivered by the camera thread (the camera's real rate).
- **Tracking FPS**: frames the tracker processed, with ms per frame and the
  number of newest-frame skips. The tracker always takes the newest frame, so
  a slow tracker never delays capture.

Overlays: green pupil ellipse, orange eye region, and once the 3D model is
built, a cyan projected eyeball with the gaze line (green when upstream's
reliability is above 0.8).

## 3D eye model

The model works like upstream's. Look around slowly while ~30 pupil samples
spread over the image are collected. Each ellipse is unprojected into two
candidate 3D circles. The eyeball centre is found by RANSAC from the projected
gaze lines, and the eye is scaled to a 12 mm radius. After that, each pupil ray
is intersected with the eyeball to give a gaze vector, a 3D pupil centre and a
pupil diameter. Upstream never runs its Ceres refinement, so this port does not
either.

Coordinates are in each camera's frame: x right, y down, z forward, in mm. For
the left eye this is the 180°-rotated image. Metric values (pupil mm, eye
position) depend on `--focal`. The GC0308 lens intrinsics were not calibrated
here, so treat them as approximate. Gaze direction is less sensitive to this.

## Failure handling

Each camera runs on its own thread, so one failing camera never stops the other.
Its panel shows a clear reason and the app retries every 2 s:

- **Wrong device at the index**: the device must be named `USB Camera`, so OBS
  or the integrated webcam is never used by mistake. The message lists every
  device with its index.
- **No camera at the index** (unplugged): the message lists the devices found.
- **Cannot open**: usually another app is using the camera.
- **Stopped delivering frames**: the camera is reopened automatically.
- **No frames for N s** while streaming: shown over the feed with a hint to
  check USB or press `k`.
- **Format fallback / low FPS**: shown as warnings on the feed.

These cases were tested with `--right 3` (OBS) and `--left 7` (missing). A
physical unplug was not tested. If a driver read blocks completely, the stall
message stays until the device returns. Replug it or restart the app.

## CSV columns

One row per tracked frame per eye: `host_time_s`, `unix_time`, `eye`,
`camera_index`, `frame_seq`, `capture_time_s`, `detected`, `pupil_x`,
`pupil_y`, `pupil_axis1_px`, `pupil_axis2_px`, `pupil_angle_deg` (OpenCV
RotatedRect, full-frame pixels after rotation), `roi_*`, `contrast`,
`model_built`, `model_samples`, `reliability`, `gaze_x/y/z`, `pupil3d_*_mm`,
`pupil_diameter_mm`, `eye_*_mm`, `capture_fps`, `tracking_fps`. Columns are
blank when there is no value (no pupil, model not built).

## Verification

```bat
run.bat --selftest 10
.venv\Scripts\python.exe -m unittest discover -s tests
```

Last self-test on this PC: left (index 1) and right (index 2) both PASS, MJPG
320×240, capture 123.9 / 124.0 FPS, tracking 121.9 / 122.7 FPS. Unit tests check
the ports on synthetic data: a pupil image with a glint, a 3D round trip, eye
centre recovery, and fitter speed (~3 ms per full frame).

**Not yet verified with a worn headset.** During testing the cameras pointed
at the room, so the "detected" percentages were false hits on scenery. Tune
the eye regions and darkness offset on real eye images.

## Files and reuse

- `binocular_demo.py`: UI, per-eye tracker threads, CSV recorder, self-test
- `camera.py`: DirectShow capture thread. The device-name guard and control
  handling come from `D:\eyetrack\capture.py`.
- `pupil_fitter.py`: port of upstream `pupilFitter.h`
- `eye_model.py`: port of upstream `EyeModelUpdater` + `singleeyefitter`
- `third_party/3D-Eye-Tracker/`: upstream MIT license and notice

The earlier project in `D:\eyetrack` (a Flask dashboard with screen-gaze
calibration) is unchanged. Its findings carried over: indices 1 and 2 are the
eye cameras, index 3 is OBS, and DirectShow devices should stay in one process.
