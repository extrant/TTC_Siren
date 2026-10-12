# -*- coding: utf-8 -*-
"""
TTC_Siren — 一键启动 / one-click start

    python app.py [--host 127.0.0.1] [--port 5000]

启动九宫幻卡求解服务（Python/app.py，默认 http://127.0.0.1:5000，接口与以前完全一致）。/ai_move 的 minimax 搜索由 C# 引擎完成：
第一次启动时自动构建到 dist/engine（需要 .NET 10 SDK），之后服务会自己启动和关闭它。设置 TTC_CS_ENGINE=off 则使用原来的 Python 搜索。

Starts the Triple Triad solver service (Python/app.py, http://127.0.0.1:5000 by default, the same API as always). The minimax search of
/ai_move runs in the C# engine: the first start builds it into dist/engine (needs the .NET 10 SDK); afterwards the service starts and
stops it by itself. TTC_CS_ENGINE=off uses the original Python search instead.
"""
import os
import runpy
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(ROOT, 'Python')
ENGINE_DIR = os.path.join(ROOT, 'dist', 'engine')
ENGINE_EXE = os.path.join(ENGINE_DIR, 'TTC.Siren.Engine.exe' if os.name == 'nt' else 'TTC.Siren.Engine')
ENGINE_PROJECT = os.path.join(ROOT, 'CSharp', 'src', 'TTC.Siren.Engine')


def _switched_off(name):
    return os.environ.get(name, 'on').strip().lower() in ('off', '0', 'false', 'no')


def check_packages():
    """The packages of Python/requirements.txt; a clear message instead of a traceback when one is missing."""
    missing = []
    for module in ('flask', 'numpy', 'pandas', 'rich'):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print('缺少 Python 包 / missing packages: %s' % ', '.join(missing))
        print('  pip install -r Python/requirements.txt')
        sys.exit(1)


def build_engine():
    """Builds the C# engine into dist/engine once (the same publish as CSharp/build_engine.bat). Without the .NET SDK it only explains."""
    if _switched_off('TTC_CS_ENGINE') or os.environ.get('TTC_CS_ENGINE_EXE') or os.path.isfile(ENGINE_EXE):
        return
    dotnet = shutil.which('dotnet')
    if dotnet is None:
        print(' * 未找到 .NET 10 SDK，无法构建 C# 求解引擎 / .NET 10 SDK not found, the C# engine cannot be built.')
        print('   安装 https://dotnet.microsoft.com/download ，或设置 TTC_CS_ENGINE=off 使用 Python 搜索。')
        return
    print(' * 第一次启动：构建 C# 求解引擎到 dist/engine（约一分钟）/ first start: building the C# engine into dist/engine ...', flush=True)
    args = [dotnet, 'publish', ENGINE_PROJECT, '-c', 'Release', '-o', ENGINE_DIR, '--nologo']
    if os.name == 'nt':
        args += ['-r', 'win-x64', '--self-contained', 'true', '-p:PublishSingleFile=true', '-p:IncludeNativeLibrariesForSelfExtract=true',
                 '-p:EnableCompressionInSingleFile=true', '-p:DebugType=none']
    if subprocess.run(args).returncode != 0 or not os.path.isfile(ENGINE_EXE):
        print(' * C# 引擎构建失败（见上面的输出）/ building the C# engine failed (see above).')
        return
    for leftover in ('web.config', 'TTC.Siren.Engine.staticwebassets.endpoints.json'):
        try:
            os.remove(os.path.join(ENGINE_DIR, leftover))
        except OSError:
            pass


def main():
    check_packages()
    build_engine()
    # run the server exactly as `python app.py` inside Python/ always ran: its folder is the working directory and the first import path
    os.chdir(PYTHON_DIR)
    sys.path[0] = PYTHON_DIR
    sys.argv[0] = os.path.join(PYTHON_DIR, 'app.py')
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
