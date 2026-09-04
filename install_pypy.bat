@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "PYPY_VERSION=pypy3.11-v7.3.23-win64"
set "PYPY_URL=https://downloads.python.org/pypy/%PYPY_VERSION%.zip"
set "ROOT=%~dp0"
set "PYPY_DIR=%ROOT%PYPY"
set "TMP_ZIP=%ROOT%_pypy_download.zip"
set "TMP_EXTRACT=%ROOT%_pypy_extract"

if exist "%PYPY_DIR%\pypy3.exe" (
    echo [1/3] 已检测到 %PYPY_DIR%\pypy3.exe，跳过下载。
    goto :install_deps
)

echo [1/3] 正在下载 PyPy: %PYPY_URL%
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYPY_URL%' -OutFile '%TMP_ZIP%'"
if errorlevel 1 (
    echo 下载失败，请检查网络后重试。
    pause
    exit /b 1
)

echo [2/3] 正在解压到 %PYPY_DIR%
if exist "%TMP_EXTRACT%" rmdir /S /Q "%TMP_EXTRACT%"
powershell -NoProfile -Command "Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '%TMP_EXTRACT%' -Force"
move /Y "%TMP_EXTRACT%\%PYPY_VERSION%" "%PYPY_DIR%" >nul
rmdir /S /Q "%TMP_EXTRACT%"
del /Q "%TMP_ZIP%"

:install_deps
echo [3/3] 初始化 pip 并安装依赖 (numpy)...
"%PYPY_DIR%\pypy3.exe" -m ensurepip --default-pip
"%PYPY_DIR%\pypy3.exe" -m pip install --no-input --upgrade pip
"%PYPY_DIR%\pypy3.exe" -m pip install --no-input numpy flask rich
if errorlevel 1 (
    echo.
    echo 依赖安装失败（可能是网络波动），请重新运行 install_pypy.bat 重试。
    pause
    exit /b 1
)

echo.
echo 完成：PyPy 已安装在 %PYPY_DIR%
echo 使用 run_pypy.bat 启动服务。
pause
