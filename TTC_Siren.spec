# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('data\\幻卡数据库.csv', 'data')]
binaries = []
hiddenimports = ['csv_compat']
hiddenimports += collect_submodules('core')
hiddenimports += collect_submodules('ai')
hiddenimports += collect_submodules('config')
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'matplotlib', 'scipy', 'PIL', 'cv2', 'tensorflow', 'torch', 'PyQt5', 'PySide2', 'PySide6', 'tkinter', 'notebook', 'ipython', 'jupyter', 'zmq', 'pytest', 'unittest', 'sqlite3', 'pydoc', 'pdb', 'profile', 'cProfile'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TTC_Siren',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['E:\\幻卡C#\\全自动幻卡锦标赛_测试版\\TTC_Siren\\icon.ico'],
)
