@echo off
chcp 65001 >nul
title Build TTC_Siren.exe
rem The repository root; PyInstaller gets absolute paths (with --specpath it would read relative ones from the spec folder)
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"

where pyinstaller >nul 2>&1
if %errorlevel% neq 0 pip install pyinstaller

rem 只清理本脚本自己产出的东西，dist\TTC_Siren_PYPY 与 dist\engine 是另外的脚本的，不能一起删掉
rmdir /s /q build 2>nul
if exist "dist\TTC_Siren.exe" del /q "dist\TTC_Siren.exe"
if exist "dist\TTC_Siren" rmdir /s /q "dist\TTC_Siren"
if not exist "dist" mkdir "dist"

pyinstaller --name TTC_Siren --onefile --console --clean --noconfirm --distpath "%ROOT%\dist" --workpath "%ROOT%\build" --specpath "%ROOT%\build" --paths "%ROOT%\Python" --icon "%ROOT%\Python\scripts\icon.ico" --add-data "%ROOT%\Python\data\幻卡数据库.csv;data" --collect-submodules core --collect-submodules ai --collect-submodules config --collect-all numpy --hidden-import csv_compat --exclude-module pandas --exclude-module matplotlib --exclude-module scipy --exclude-module PIL --exclude-module cv2 --exclude-module tensorflow --exclude-module torch --exclude-module PyQt5 --exclude-module PySide2 --exclude-module PySide6 --exclude-module tkinter --exclude-module notebook --exclude-module ipython --exclude-module jupyter --exclude-module zmq --exclude-module pytest --exclude-module unittest --exclude-module sqlite3 --exclude-module pydoc --exclude-module pdb --exclude-module profile --exclude-module cProfile "%ROOT%\Python\app.py"
rmdir /s /q build 2>nul

echo.
if not exist "dist\TTC_Siren.exe" goto :build_failed

for %%i in ("dist\TTC_Siren.exe") do set FILESIZE=%%~zi
set /a SIZE_MB=FILESIZE/1024/1024
echo BUILD SUCCESS [CPython]
echo Output: dist\TTC_Siren.exe
echo Size: %SIZE_MB% MB
echo Note: single-file build; it starts the C# engine from an "engine" folder next to it (dist\engine).
goto :build_end

:build_failed
echo BUILD FAILED

:build_end
if not defined TTC_BUILD_CHAIN pause
