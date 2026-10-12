# -*- coding: utf-8 -*-
"""The server code reads ``data/幻卡数据库.csv`` relative to the working directory (as ``python app.py`` always ran from its own folder).
Run every test from the Python folder, however pytest was started (``python -m pytest`` in the repository root or in ``Python/``)."""
import os
import sys

PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PYTHON_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
