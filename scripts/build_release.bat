@echo off
chcp 65001 >nul
title Build the TTC_Siren release (TTC_Siren.exe + C# engine)
cd /d "%~dp0.."

rem The smallest package to hand out: dist\release\TTC_Siren.exe (CPython, one file) and dist\release\engine\TTC.Siren.Engine.exe
rem (self-contained), zipped to dist\TTC_Siren_release.zip. The person who gets it needs neither Python nor .NET.
rem Building needs Python with the packages of requirements.txt, PyInstaller (installed if missing) and the .NET 10 SDK.

set "TTC_BUILD_CHAIN=1"
set "OUT=dist\release"

echo ============================================================
echo  [1/3] C# engine
echo ============================================================
call "%~dp0build_engine.bat"
if errorlevel 1 goto :failed
if not exist "dist\engine\TTC.Siren.Engine.exe" goto :failed

echo.
echo ============================================================
echo  [2/3] TTC_Siren.exe (CPython, PyInstaller)
echo ============================================================
call "%~dp0build_exe.bat"
if not exist "dist\TTC_Siren.exe" goto :failed

echo.
echo ============================================================
echo  [3/3] dist\release and dist\TTC_Siren_release.zip
echo ============================================================
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%\engine"
copy /Y "dist\TTC_Siren.exe" "%OUT%\" >nul
copy /Y "dist\engine\TTC.Siren.Engine.exe" "%OUT%\engine\" >nul
copy /Y "dist\engine\engine.appsettings.json" "%OUT%\engine\" >nul
copy /Y "scripts\release_readme.txt" "%OUT%\使用说明.txt" >nul
if exist "dist\TTC_Siren_release.zip" del /q "dist\TTC_Siren_release.zip"
powershell -NoProfile -Command "Compress-Archive -Path '%OUT%\*' -DestinationPath 'dist\TTC_Siren_release.zip' -Force"
if not exist "dist\TTC_Siren_release.zip" goto :failed

echo.
echo RELEASE READY
echo   Folder: %OUT%   (TTC_Siren.exe + engine\)
echo   Zip:    dist\TTC_Siren_release.zip
echo   Unzip anywhere and double-click TTC_Siren.exe; keep the engine folder next to it.
pause
exit /b 0

:failed
echo.
echo RELEASE BUILD FAILED. See the messages above.
pause
exit /b 1
