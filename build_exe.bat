@echo off
setlocal

set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  exit /b 1
)

echo [INFO] Building exe via PyInstaller spec...
"%PY%" -m PyInstaller --noconfirm yolov26_l.spec
if errorlevel 1 (
  echo [ERROR] EXE build failed.
  exit /b 1
)

echo [OK] EXE output: dist\yolov26_l\yolov26_l.exe
exit /b 0
