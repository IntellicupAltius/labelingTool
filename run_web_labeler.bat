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

REM Pick a Python launcher
set "PYLAUNCH="
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYLAUNCH=py -3"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PYLAUNCH=python"
  ) else (
    where python3 >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PYLAUNCH=python3"
    ) else (
      echo ERROR: Python not found. Install Python 3.10+ and check "Add to PATH".>> "%LOG%"
      echo ERROR: Python not found. Install Python 3.10+ and check "Add to PATH".
      goto :error
    )
  )
)

REM Use local venv only. This matches the "old folder works" update flow:
REM extract new ZIP -> copy old .venv into the new folder -> run.
set "VENV_DIR=%~dp0.venv"
echo VENV_DIR=%VENV_DIR%>> "%LOG%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo ERROR: Missing local venv: %VENV_DIR%\Scripts\python.exe>> "%LOG%"
  echo.
  echo ERROR: Missing local venv in this folder.
  echo.
  echo Fix (recommended):
  echo - Copy the .venv folder from the OLD working install into this new folder:
  echo     OLD: ^<old_tool_folder^>\.venv
  echo     NEW: %~dp0.venv
  echo.
  echo Then run this .bat again.
  goto :error
)

echo Installing/updating dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
if %ERRORLEVEL% neq 0 goto :error
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if %ERRORLEVEL% neq 0 goto :error

echo Starting server...
set LABELER_OPEN_BROWSER=1
"%VENV_DIR%\Scripts\python.exe" run_web_labeler.py >> "%LOG%" 2>&1
if %ERRORLEVEL% neq 0 goto :error

goto :eof

:error
echo.
echo FAILED. See log: %LOG%
echo ---- Last 40 lines ----
powershell -NoProfile -Command "if (Test-Path '%LOG%') { Get-Content -Tail 40 '%LOG%' }" 2>nul
echo -----------------------
pause


