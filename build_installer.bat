@echo off
setlocal

set ROOT=%~dp0
set ISS=%ROOT%installer\yolov26_l.iss
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist "%ROOT%dist\yolov26_l\yolov26_l.exe" (
  echo [ERROR] EXE not found: dist\yolov26_l\yolov26_l.exe
  echo Build exe first:
  echo   .venv\Scripts\python.exe -m PyInstaller --noconfirm yolov26_l.spec
  exit /b 1
)

if not exist %ISCC% (
  echo [ERROR] Inno Setup not found at:
  echo   %ISCC%
  echo Install Inno Setup 6, or edit this script to your ISCC.exe path.
  exit /b 1
)

echo [INFO] Building installer...
%ISCC% "%ISS%"
if errorlevel 1 (
  echo [ERROR] Installer build failed.
  exit /b 1
)

echo [OK] Installer created under dist_installer\
exit /b 0
