@echo off
chcp 65001 >nul
title Build TTC_Siren (C# engine + CPython + PyPy)
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
set "TTC_BUILD_CHAIN=1"

echo ============================================================
echo  [1/3] C# engine -^> dist\engine\TTC.Siren.Engine.exe
echo ============================================================
call "%ROOT%\CSharp\build_engine.bat"

echo.
echo ============================================================
echo  [2/3] CPython build -^> dist\TTC_Siren.exe
echo ============================================================
call "%~dp0build_exe.bat"

echo.
echo ============================================================
echo  [3/3] PyPy build -^> dist\TTC_Siren_PYPY\TTC_Siren_PYPY.exe
echo ============================================================
call "%~dp0build_exe_pypy.bat"

echo.
echo ============================================================
echo  All builds finished. Outputs (in dist\ of the repository):
echo    C# engine: dist\engine\TTC.Siren.Engine.exe     (both servers start it by themselves)
echo    CPython:   dist\TTC_Siren.exe                    (single file)
echo    PyPy:      dist\TTC_Siren_PYPY\TTC_Siren_PYPY.exe  (ship this whole subfolder)
echo    Zip the dist folder to distribute everything; for one small package use build_release.bat.
echo ============================================================
pause
