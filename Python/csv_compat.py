# -*- coding: utf-8 -*-
"""
csv_compat.py
-------------
用 stdlib csv 模块实现 ai_server.py / ai/ai.py 实际用到的那一小撮 pandas 接口
（DataFrame.iterrows() 与 pd.notna()），不引入完整 pandas 依赖。

背景：PyPy 在 Windows 上没有 pandas 的预编译轮子，源码构建又需要本机 C 编译器，
而这两处用法只是逐行读取卡牌数据库 CSV，用不上 pandas 的其余能力，
所以两个文件里改成 `import csv_compat as pd` 即可，其余调用代码不用动。
"""
from __future__ import annotations

import csv
from typing import Iterator, List, Tuple


class Row(dict):
    """一行数据，支持 row['列名'] 访问，行为等价于 pandas Series 的用法。"""
    pass


class Table:
    """极简表格封装，只实现项目里用到的 iterrows()。"""

    def __init__(self, rows: List[Row]):
        self._rows = rows

    def iterrows(self) -> Iterator[Tuple[int, Row]]:
        return enumerate(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def read_csv(path: str) -> Table:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return Table([Row(r) for r in csv.DictReader(f)])


def notna(value) -> bool:
    return value is not None and value != ""
