# 3D-Eye-Tracker (upstream)

- Source: https://github.com/YutaItoh/3D-Eye-Tracker
- Revision used: `a8e2180577308596b6ac8078558df1e430789191` (2024-11-28)
- License: MIT, reproduced unmodified in [LICENSE](LICENSE)

Upstream credits its 2D pupil detector to Jason Orlosky and its 3D eye model
code (`singleeyefitter`) to Leszek Swirski; Yuta Itoh integrated them.

## What this demo ports

| This project | Upstream file(s) |
|---|---|
| `pupil_fitter.py` | `main/pupilFitter.h` (`PupilFitter::pupilAreaFitRR` and helpers) |
| `eye_model.py` `EyeModelUpdater` | `main/eye_model_updater.cpp/.h` (spatial bins, 30 samples, reliability > 0.8, rendering) |
| `eye_model.py` `EyeModelFitter` | `singleeyefitter/SingleEyeFitter.cpp` (`unproject_observations` with RANSAC, `initialise_model`, `unproject_single_observation`, `initialise_single_observation`) |
| `eye_model.py` `unproject`, `similarity` | `singleeyefitter/singleeyefitter/projection.h`, `Ellipse.h` |

The port is a Python reimplementation, not a copy of the C++ sources. No upstream
binaries or bundled third-party libraries (Ceres, Eigen, spii, Boost-licensed
`DistancePointEllipse.h`) are included or used.
