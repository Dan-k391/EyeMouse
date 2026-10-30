# EyeMouse: binocular eye-tracking demo

A Windows demo for two GC0308 USB eye cameras. It shows both feeds with fitted
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
