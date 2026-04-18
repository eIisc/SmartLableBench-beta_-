@echo off
setlocal

set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  exit /b 1
)

echo [1/2] Build release: SLB-beta-v0.01P
"%PY%" -m PyInstaller --noconfirm --clean --distpath dist_release --workpath build_release "SLB-beta-v0.01P.spec"
if errorlevel 1 (
  echo [ERROR] Release build failed.
  exit /b 1
)

echo [2/2] Build test: SLB-beta-v0.01
"%PY%" -m PyInstaller --noconfirm --clean --distpath dist_test --workpath build_test "SLB-beta-v0.01.spec"
if errorlevel 1 (
  echo [ERROR] Test build failed.
  exit /b 1
)

echo [OK] Release output: dist_release\SLB-beta-v0.01P\SLB-beta-v0.01P.exe
echo [OK] Test output:    dist_test\SLB-beta-v0.01\SLB-beta-v0.01.exe
exit /b 0
