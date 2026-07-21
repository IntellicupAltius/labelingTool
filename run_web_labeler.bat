@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d %~dp0

REM One-click Windows runner:
REM - Creates a local venv in .venv (first run)
REM - Installs dependencies
REM - Starts the server
REM Writes logs to run_web_labeler.log so failures are visible.

set "LOG=%~dp0run_web_labeler.log"
echo ==== %DATE% %TIME% ==== > "%LOG%"
echo Working dir: %CD%>> "%LOG%"
echo USERNAME=%USERNAME%>> "%LOG%"
echo USERPROFILE=%USERPROFILE%>> "%LOG%"
echo LOCALAPPDATA=%LOCALAPPDATA%>> "%LOG%"
echo TEMP=%TEMP%>> "%LOG%"

REM Use local venv only — all operations use .venv\Scripts\python.exe directly.
set "VENV_DIR=%~dp0.venv"
echo VENV_DIR=%VENV_DIR%>> "%LOG%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo ERROR: Missing local venv: %VENV_DIR%\Scripts\python.exe>> "%LOG%"
  echo.
  echo ERROR: .venv not found in this folder.
  echo Copy the .venv folder from the old working install here:
  echo   %~dp0.venv
  echo.
  pause
  exit /b 1
)

echo Installing/updating dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
if %ERRORLEVEL% neq 0 (
  echo ERROR: pip upgrade failed. See %LOG%
  pause
  exit /b 1
)
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if %ERRORLEVEL% neq 0 (
  echo ERROR: dependency install failed. See %LOG%
  pause
  exit /b 1
)

echo Starting server...
echo.
set LABELER_OPEN_BROWSER=1
"%VENV_DIR%\Scripts\python.exe" run_web_labeler.py
if %ERRORLEVEL% neq 0 (
  echo.
  echo Server exited with error code %ERRORLEVEL%
  echo Check %LOG% for details.
  pause
  exit /b 1
)

echo.
echo Server stopped.
pause


