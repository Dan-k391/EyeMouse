@echo off
rem Binocular eye-tracking demo launcher. Extra arguments are passed through,
rem e.g.  run.bat --selftest 10    or    run.bat --left 2 --right 1
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv || goto :setup_failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :setup_failed
)
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" binocular_demo.py %*
exit /b %errorlevel%

:setup_failed
echo Setup failed. Install Python 3.12 with the py launcher, then run this again.
exit /b 1
