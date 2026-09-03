# -*- coding: utf-8 -*-
"""
console_ui.py
-------------
统一终端可视化系统，替代散落各处的 print() 刷屏日志：

  - 固定区域重绘的"思考中"状态行（Rich Live，禁止逐行累加打印）
  - 左右双栏面板：左侧输入/环境（棋盘、规则、参数），右侧输出/决策（预测棋盘、决策、置信度）
  - 终端能力探测：Windows Terminal / ConPTY 真彩色时启用流光特效，
    传统 CMD 等不支持高阶 ANSI 的终端自动降级为基础色彩，避免乱码

用法：
    from console_ui import ui, init_console_window

    init_console_window(120, 30)   # 启动时固定控制台窗口尺寸
    ui.start()                     # 启动后台渲染线程

    ui.begin_request("ai_move")
    ui.update_input(board=board, rules=rules, params={"求解器": "Minimax"})
    ui.update_thinking("Minimax 深度7 节点12,345 ...")
    ui.update_output(board=predicted_board, decision="U8 R3 D2 L5 → (1,1)", confidence="72%")
    ui.end_request("完成")
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ═══════════════════════════════════════════════════════════════════
# 终端能力探测 & 窗口初始化
# ═══════════════════════════════════════════════════════════════════


def detect_terminal() -> dict:
    """探测当前终端环境，决定使用高级特效还是降级渲染。"""
    console = Console()
    color_system = console.color_system  # None / "standard" / "256" / "truecolor" / "windows"
    is_tty = console.is_terminal
    return {
        "is_windows": os.name == "nt",
        "is_windows_terminal": bool(os.environ.get("WT_SESSION")),
        "is_vscode": os.environ.get("TERM_PROGRAM") == "vscode",
        "color_system": color_system,
        "supports_truecolor": color_system == "truecolor" and is_tty,
        "is_tty": is_tty,
    }


def init_console_window(cols: int = 120, lines: int = 30) -> None:
    """尝试将控制台窗口/缓冲区初始化为固定尺寸（仅 Windows，失败静默忽略）。"""
    if os.name != "nt" or not sys.stdout.isatty():
        return
    try:
        os.system(f"mode con: cols={cols} lines={lines}")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# 共享状态
# ═══════════════════════════════════════════════════════════════════

_ICON_FRAMES = ["✦", "✶", "✷", "✸", "✹", "✸", "✷", "✶"]
_ICON_COLOR = "bold rgb(217,153,87)"
_BASE_RGB = (110, 110, 110)
_HILITE_RGB = (255, 255, 255)


@dataclass
class _State:
    active: bool = False
    label: str = "空闲"
    note: str = ""
    endpoint: str = ""
    started_at: float = 0.0

    input_board: object = None
    input_rules: list = field(default_factory=list)
    input_params: dict = field(default_factory=dict)

    output_board: object = None
    output_decision: str = ""
    output_confidence: str = ""
    output_extra: dict = field(default_factory=dict)


class ConsoleUI:
    """常驻后台渲染线程：固定区域重绘状态行 + 输入/输出双栏面板，取代逐行刷屏。"""

    def __init__(self, fps: float = 8.0):
        self.fps = fps
        self.console = Console()
        self.caps = detect_terminal()
        self._state = _State()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick = 0
        self._start_time = time.time()

    # ── 生命周期 ──

    def start(self) -> "ConsoleUI":
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    # ── 状态更新 API（线程安全，供请求处理线程调用）──

    def begin_request(self, endpoint: str, label: str = "求解中") -> None:
        with self._lock:
            self._state.active = True
            self._state.label = label
            self._state.endpoint = endpoint
            self._state.note = ""
            self._state.started_at = time.time()
            self._state.input_params = {}
            self._state.output_board = None
            self._state.output_decision = ""
            self._state.output_confidence = ""
            self._state.output_extra = {}

    def end_request(self, label: str = "完成") -> None:
        with self._lock:
            self._state.active = False
            self._state.label = label

    def update_thinking(self, note: str) -> None:
        with self._lock:
            self._state.note = note

    def update_input(self, board=None, rules=None, params: Optional[dict] = None) -> None:
        with self._lock:
            if board is not None:
                self._state.input_board = board
            if rules is not None:
                self._state.input_rules = list(rules)
            if params:
                self._state.input_params.update(params)

    def update_output(self, board=None, decision: Optional[str] = None,
                       confidence: Optional[str] = None, extra: Optional[dict] = None) -> None:
        with self._lock:
            if board is not None:
                self._state.output_board = board
            if decision is not None:
                self._state.output_decision = decision
            if confidence is not None:
                self._state.output_confidence = confidence
            if extra:
                self._state.output_extra.update(extra)

    # ── 渲染 ──

    def _shimmer_text(self, text: str, phase: float) -> Text:
        if not self.caps["supports_truecolor"]:
            return Text(text, style="bold cyan")
        n = max(len(text), 1)
        width = max(2.0, n / 3)
        center = -width + phase * (n + 2 * width)
        t = Text()
        for i, ch in enumerate(text):
            dist = abs(i - center)
            k = max(0.0, 1 - dist / (width / 1.4))
            r = int(_BASE_RGB[0] + (_HILITE_RGB[0] - _BASE_RGB[0]) * k)
            g = int(_BASE_RGB[1] + (_HILITE_RGB[1] - _BASE_RGB[1]) * k)
            b = int(_BASE_RGB[2] + (_HILITE_RGB[2] - _BASE_RGB[2]) * k)
            t.append(ch, style=f"rgb({r},{g},{b})")
        return t

    def _thinking_line(self) -> Text:
        with self._lock:
            active, label, note = self._state.active, self._state.label, self._state.note
        if active:
            if self.caps["supports_truecolor"]:
                icon_idx = int(self._tick * 6.0 / self.fps) % len(_ICON_FRAMES)
                icon = _ICON_FRAMES[icon_idx]
                elapsed = time.time() - self._start_time
                cycle = (elapsed % 1.6) / 1.6
                phase = cycle * 2 if cycle < 0.5 else 2 - cycle * 2
                line = Text(icon + "  ", style=_ICON_COLOR)
                line.append(self._shimmer_text(label, phase))
            else:
                line = Text("* " + label, style="bold yellow")
        else:
            line = Text("o " + label, style="dim")
        if note:
            line.append("    " + note, style="dim")
        return line

    def _card_cell(self, card) -> Text:
        if card is None:
            return Text("\n·  空  ·\n", style="dim", justify="center")
        owner_style = "bold red" if card.owner == "red" else "bold blue" if card.owner == "blue" else "white"
        owner_letter = card.owner[0].upper() if card.owner else "?"
        cid = card.card_id if card.card_id is not None else "?"
        body = (
            f"[{owner_letter}:{cid}]\n"
            f"U{card.get_display_value('up')}\n"
            f"L{card.get_display_value('left')}  R{card.get_display_value('right')}\n"
            f"D{card.get_display_value('down')}"
        )
        return Text(body, style=owner_style, justify="center")

    def _board_table(self, board) -> Table:
        table = Table(show_header=False, box=box.SQUARE, padding=(0, 1), show_lines=True, expand=True)
        for _ in range(3):
            table.add_column(justify="center", ratio=1)
        for r in range(3):
            table.add_row(*[self._card_cell(board.get_card(r, c)) for c in range(3)])
        return table

    def _input_panel(self) -> Panel:
        with self._lock:
            board = self._state.input_board
            rules = list(self._state.input_rules)
            params = dict(self._state.input_params)
        body = Table.grid(padding=(0, 1), expand=True)
        body.add_column(ratio=1)
        body.add_row(self._board_table(board) if board is not None else Align.center(Text("等待棋盘数据…", style="dim")))
        body.add_row(Text(f"规则: {', '.join(rules) if rules else '无'}", style="cyan"))
        for key, value in params.items():
            body.add_row(Text(f"{key}: {value}", style="grey70"))
        return Panel(body, title="输入 / 环境", border_style="cyan", expand=True)

    def _output_panel(self) -> Panel:
        with self._lock:
            board = self._state.output_board
            decision = self._state.output_decision
            confidence = self._state.output_confidence
            extra = dict(self._state.output_extra)
        body = Table.grid(padding=(0, 1), expand=True)
        body.add_column(ratio=1)
        body.add_row(self._board_table(board) if board is not None else Align.center(Text("等待决策结果…", style="dim")))
        if decision:
            body.add_row(Text(f"决策: {decision}", style="bold green"))
        if confidence:
            body.add_row(Text(f"胜率/置信度: {confidence}", style="green"))
        for key, value in extra.items():
            body.add_row(Text(f"{key}: {value}", style="grey70"))
        return Panel(body, title="输出 / 决策", border_style="green", expand=True)

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(Panel(self._thinking_line(), border_style="grey50"), size=3, name="thinking"),
            Layout(name="panels"),
        )
        layout["panels"].split_row(
            Layout(self._input_panel(), name="input"),
            Layout(self._output_panel(), name="output"),
        )
        return layout

    def _loop(self) -> None:
        with Live(console=self.console, refresh_per_second=self.fps, screen=False) as live:
            while not self._stop_event.is_set():
                live.update(self._render())
                self._tick += 1
                time.sleep(1 / self.fps)


class SearchReporter:
    """节流输出搜索进度到状态行，替代原先逐行 print 的 ConsoleSearchReporter。"""

    def __init__(self, ui_instance: "ConsoleUI", interval: float = 0.5):
        self.ui = ui_instance
        self.interval = interval
        self.last_print = 0.0
        self.endgame_nodes = 0
        self.endgame_start: Optional[float] = None

    def on_minimax_progress(self, progress_info: dict) -> None:
        now = time.time()
        if now - self.last_print < self.interval:
            return
        self.last_print = now
        stats = progress_info.get("stats", {})
        elapsed = max(progress_info.get("time_elapsed", 0.0), 1e-6)
        nodes = progress_info.get("nodes_searched", 0)
        nps = nodes / elapsed
        self.ui.update_thinking(
            f"Minimax 深度={progress_info.get('depth')} 节点={nodes:,} nps={nps:,.0f} "
            f"tt命中={stats.get('tt_hit_rate', 0) * 100:.1f}% "
            f"剪枝={stats.get('cutoff_rate', 0) * 100:.1f}% 用时={elapsed:.2f}s"
        )

    def start_endgame(self) -> None:
        self.endgame_nodes = 0
        self.endgame_start = time.time()
        self.last_print = 0.0

    def on_endgame_node(self, move_index: int, move_count: int,
                         scenario_index: int, scenario_count: int) -> None:
        self.endgame_nodes += 1
        now = time.time()
        if now - self.last_print < self.interval:
            return
        self.last_print = now
        elapsed = max(now - (self.endgame_start or now), 1e-6)
        self.ui.update_thinking(
            f"残局推演 招法={move_index}/{move_count} 场景={scenario_index}/{scenario_count} "
            f"节点={self.endgame_nodes:,} nps={self.endgame_nodes / elapsed:,.0f} 用时={elapsed:.2f}s"
        )


ui = ConsoleUI()
