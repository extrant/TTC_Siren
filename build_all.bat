@echo off
chcp 65001 >nul
title Build TTC_Siren (CPython + PyPy)
cd /d "%~dp0"
set "TTC_BUILD_CHAIN=1"

echo ============================================================
echo  [1/2] CPython build -^> dist\TTC_Siren.exe
echo ============================================================
call "%~dp0build_exe.bat"

echo.
echo ============================================================
echo  [2/2] PyPy build -^> dist\TTC_Siren_PYPY\TTC_Siren_PYPY.exe
echo ============================================================
call "%~dp0build_exe_pypy.bat"

echo.
echo ============================================================
echo  All builds finished. Outputs:
echo    CPython: dist\TTC_Siren.exe                    (single file)
echo    PyPy:    dist\TTC_Siren_PYPY\TTC_Siren_PYPY.exe  (ship this whole subfolder)
echo    Everything lives under dist\ now - zip that one folder to distribute both.
echo ============================================================
pause
