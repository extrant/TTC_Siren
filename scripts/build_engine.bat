@echo off
chcp 65001 >nul
title Build the C# engine (TTC.Siren.Engine)
cd /d "%~dp0.."

rem The C# engine solves the minimax search of /ai_move for the Python server (decision A-30). The server starts it by itself
rem (ai\engine_launcher.py) from dist\engine, next to dist\TTC_Siren.exe, so the published folder is all a user needs.
rem Self-contained single file: the machine that runs it needs no .NET installed. Building needs the .NET 10 SDK.

where dotnet >nul 2>&1
if %errorlevel% neq 0 (
  echo .NET SDK 10 not found: https://dotnet.microsoft.com/download
  goto :build_failed
)

if exist "dist\engine\TTC.Siren.Engine.exe" (
  del /q "dist\engine\TTC.Siren.Engine.exe" 2>nul
  if exist "dist\engine\TTC.Siren.Engine.exe" (
    echo dist\engine\TTC.Siren.Engine.exe is in use: stop the server ^(and the engine^) first.
    goto :build_failed
  )
)
if exist "dist\engine" rmdir /s /q "dist\engine"

dotnet publish csharp\src\TTC.Siren.Engine -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true -p:DebugType=none -o dist\engine --nologo
if %errorlevel% neq 0 goto :build_failed
if not exist "dist\engine\TTC.Siren.Engine.exe" goto :build_failed
rem leftovers of the ASP.NET Core web SDK that a loopback engine does not use
del /q "dist\engine\web.config" "dist\engine\*.staticwebassets.endpoints.json" 2>nul

for %%i in ("dist\engine\TTC.Siren.Engine.exe") do set FILESIZE=%%~zi
set /a SIZE_MB=FILESIZE/1024/1024
echo.
echo BUILD SUCCESS [C# engine]
echo Output: dist\engine\TTC.Siren.Engine.exe  (%SIZE_MB% MB)
echo The server (python app.py, scripts\run_pypy.bat, dist\TTC_Siren.exe) starts it by itself when it is not running.
goto :build_end

:build_failed
echo BUILD FAILED [C# engine]
if not defined TTC_BUILD_CHAIN pause
exit /b 1

:build_end
if not defined TTC_BUILD_CHAIN pause
