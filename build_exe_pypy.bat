@echo off
chcp 65001 >nul
title Build TTC_Siren_PyPy
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PYPY_SRC=%ROOT%PYPY"
set "OUT_DIR=%ROOT%dist\TTC_Siren_PYPY"

if not exist "%PYPY_SRC%\pypy3.exe" (
    echo 未找到 %PYPY_SRC%\pypy3.exe，先运行 install_pypy.bat 安装 PyPy...
    call "%ROOT%install_pypy.bat"
)
if not exist "%PYPY_SRC%\pypy3.exe" (
    echo 仍未找到 PyPy，安装失败，中止打包。
    pause
    exit /b 1
)

echo [1/5] 清理旧的输出目录（只清自己这份，不动 dist\TTC_Siren.exe）...
rmdir /s /q "%OUT_DIR%" 2>nul
if not exist "%ROOT%dist" mkdir "%ROOT%dist"
mkdir "%OUT_DIR%" 2>nul

echo [2/5] 复制 PyPy 运行时（含已安装的 numpy/flask/rich 等依赖）...
robocopy "%PYPY_SRC%" "%OUT_DIR%\PYPY" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NC /NS >nul

echo [3/5] 复制项目源码（app.py / ai_server.py / core / ai / config / data）...
copy /Y "%ROOT%app.py" "%OUT_DIR%\" >nul
copy /Y "%ROOT%ai_server.py" "%OUT_DIR%\" >nul
copy /Y "%ROOT%csv_compat.py" "%OUT_DIR%\" >nul
copy /Y "%ROOT%console_ui.py" "%OUT_DIR%\" >nul
if exist "%ROOT%icon.ico" copy /Y "%ROOT%icon.ico" "%OUT_DIR%\" >nul
robocopy "%ROOT%ai" "%OUT_DIR%\ai" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NC /NS >nul
robocopy "%ROOT%core" "%OUT_DIR%\core" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NC /NS >nul
robocopy "%ROOT%config" "%OUT_DIR%\config" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NC /NS >nul
mkdir "%OUT_DIR%\data" 2>nul
copy /Y "%ROOT%data\幻卡数据库.csv" "%OUT_DIR%\data\" >nul

echo [4/5] 编译原生启动器 TTC_Siren_PYPY.exe（PyInstaller 不支持 PyPy，用 csc.exe 编译一个小的启动 exe）...
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" (
    echo 未找到 csc.exe（.NET Framework 4.x），无法编译启动器。
    echo 你仍可以进入 "%OUT_DIR%" 手动运行 PYPY\pypy3.exe app.py
    pause
    exit /b 1
)

"%CSC%" /nologo /target:exe /platform:x64 /out:"%OUT_DIR%\TTC_Siren_PYPY.exe" "%ROOT%pypy_launcher.cs"
if errorlevel 1 (
    echo 启动器编译失败。
    pause
    exit /b 1
)

echo [5/5] 完成
echo.
if not exist "%OUT_DIR%\TTC_Siren_PYPY.exe" goto :build_failed

echo BUILD SUCCESS [PyPy]
echo Output: %OUT_DIR%\TTC_Siren_PYPY.exe
echo Note: this is a portable folder plus a native launcher exe, not a single-file build
echo       ^(PyInstaller cannot target PyPy^). Ship the whole "%OUT_DIR%" folder and
echo       double-click TTC_Siren_PYPY.exe inside it to run.
goto :build_end

:build_failed
echo BUILD FAILED

:build_end
if not defined TTC_BUILD_CHAIN pause
