@echo off
setlocal

cd /d %~dp0

REM One-click Windows runner:
REM - Creates a local venv in .venv (first run)
REM - Installs dependencies
REM - Starts the server

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

echo Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Starting server...
set LABELER_OPEN_BROWSER=1
".venv\Scripts\python.exe" run_web_labeler.py


