@echo off
chcp 65001 >nul
title TTC_Siren (C# engine)
cd /d "%~dp0"

rem One click: the Python server on http://127.0.0.1:5000 (every endpoint as before, no key), with the C# engine solving /ai_move.
rem The first run builds the engine (scripts\build_engine.bat, needs the .NET 10 SDK); the server starts and stops the engine by itself.
rem   run_local.bat           CPython (python.exe on PATH)
rem   run_local.bat pypy      PyPy from PYPY\pypy3.exe (scripts\install_pypy.bat)

if not exist "dist\engine\TTC.Siren.Engine.exe" (
  echo C# engine not built yet: building it once ...
  set "TTC_BUILD_CHAIN=1"
  call "%~dp0scripts\build_engine.bat"
  set "TTC_BUILD_CHAIN="
)
if not exist "dist\engine\TTC.Siren.Engine.exe" (
  echo The C# engine could not be built. See the messages above.
  pause
  exit /b 1
)

set "PY=python.exe"
if /i "%~1"=="pypy" set "PY=%~dp0PYPY\pypy3.exe"
if /i "%~1"=="pypy" if not exist "%PY%" (
  echo PYPY\pypy3.exe not found: run scripts\install_pypy.bat first.
  pause
  exit /b 1
)

"%PY%" app.py
pause
