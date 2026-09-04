@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
set "PYPY_EXE=%ROOT%PYPY\pypy3.exe"

if not exist "%PYPY_EXE%" (
    echo 未找到 %PYPY_EXE%
    echo 请先运行 install_pypy.bat 安装 PyPy。
    pause
    exit /b 1
)

cd /d "%ROOT%"
"%PYPY_EXE%" app.py %*

pause
