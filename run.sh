#!/bin/sh
# Binocular eye-tracking demo launcher for macOS. Extra arguments are passed through,
# e.g.  ./run.sh --selftest 10    or    ./run.sh --left 1 --right 0
# ./run.sh mouse runs the headset eye mouse, ./run.sh webcam the webcam-only one.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt || {
        echo "Setup failed. Install Python 3, then run this again."
        exit 1
    }
fi
case "$1" in
    mouse) script=eyemouse.py; shift ;;
    webcam) script=webcam_mouse.py; shift ;;
    *) script=binocular_demo.py ;;
esac
exec .venv/bin/python "$script" "$@"
