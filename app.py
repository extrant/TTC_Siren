# -*- coding: utf-8 -*-
"""
Unified Triple Triad AI Server
===============================
整合三个子系统：
  - /ai_move              求解器（继承自 ai_server）
  - /search_progress      搜索进度（继承自 ai_server）
  - /api/recommend        NPC 卡组推荐
  - /api/draft/analyze    选拔深算分析（前端传入 3 套卡组，返回最优）

运行：python app.py [--host 127.0.0.1] [--port 5000]

不修改 ai_server.py / npc_deck_recommender_demo.py / draft_analyzer_app.py 任何文件。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from flask import Flask, request, jsonify

# ── 引用现有求解器 app（继承 /ai_move、/search_progress 等全部路由）──
from ai_server import (
    app,
    get_all_cards,
    get_card_star_map,
    get_card_type_map,
)

# ── 核心领域对象 ──
from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player

# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

EXTRA_RULES = {"同数", "加算", "逆转", "王牌杀手", "同类强化", "同类弱化", "秩序", "混乱"}

# --- NPC 推荐 ---
MAX_PRELIMINARY_DECKS = 48
MAX_NPC_SOURCE_POOL = 18
MAX_NPC_DECKS = 10

# --- 选拔分析 ---
DEFAULT_WEIGHTS = {
    "base": 0.18,
    "rule": 0.24,
    "synergy": 0.16,
    "stability": 0.10,
    "verification": 0.32,
    "risk": 0.12,
}
MAX_POOL_ENUMERATION = 24024
POOL_VERIFY_LIMIT = 18
DEFAULT_TIME_BUDGET = 7.0

# ═══════════════════════════════════════════════════════════════════
# 共享工具函数
# ═══════════════════════════════════════════════════════════════════


def _real_card_id(card: Card) -> int:
    """返回基础卡牌 ID（常规或生成 ID 取模 1000）。"""
    cid = int(card.card_id or 0)
    if getattr(card, "_draft_synthetic", False):
        return cid
    if 1000 <= cid < 100000:
        return cid % 1000
    return cid


def _card_star(card: Card, star_map: Dict[int, int]) -> int:
    """读取卡牌星级，带保守回退。"""
    draft_star = getattr(card, "_draft_star", None)
    if draft_star is not None:
        return int(draft_star)
    return int(star_map.get(_real_card_id(card), 1))


def _card_to_dict(card: Card, star_map: Dict[int, int]) -> Dict:
    """统一序列化卡牌为 JSON 友好格式。"""
    result = {
        "id": _real_card_id(card),
        "up": card.base_up,
        "right": card.base_right,
        "down": card.base_down,
        "left": card.base_left,
        "star": _card_star(card, star_map),
        "type": card.card_type or "",
    }
    label = getattr(card, "_draft_label", None)
    if label:
        result["label"] = label
    return result


def _side_values(card: Card) -> List[int]:
    """返回卡牌四面数值 [上, 右, 下, 左]。"""
    return [card.base_up, card.base_right, card.base_down, card.base_left]


def _parse_id_list(value) -> List[int]:
    """解析逗号/空格分隔的 ID 或可迭代对象。"""
    if isinstance(value, str):
        parts = value.replace("，", ",").replace(" ", ",").split(",")
        return [int(item) for item in parts if item.strip()]
    return [int(item) for item in value]


def _parse_rules(value) -> List[str]:
    """解析规则文本，仅保留支持的额外规则。"""
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace("，", ",").split(",")]
    else:
        raw = [str(item).strip() for item in value]
    return [rule for rule in raw if rule in EXTRA_RULES]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """将数值裁剪到展示友好范围内。"""
    return max(low, min(high, value))


def _positional_score(row: int, col: int) -> float:
    """棋盘位置价值。"""
    if (row, col) in ((0, 0), (0, 2), (2, 0), (2, 2)):
        return 1.4
    if (row, col) == (1, 1):
        return 1.1
    return 0.9


# ═══════════════════════════════════════════════════════════════════
# 统一结果数据类
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DeckResult:
    """卡组分析结果，统一用于 NPC 推荐和选拔分析。"""
    deck: List[Dict]
    final_score: float
    components: Dict[str, float]
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    selection: List[str] = field(default_factory=list)
    groups: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 子系统 A：NPC 卡组推荐
# ═══════════════════════════════════════════════════════════════════


def _clone_card(card: Card, owner: Optional[str] = None) -> Card:
    """复制卡牌并可选设置 owner。"""
    copied = card.copy()
    copied.owner = owner
    copied.can_use = True
    return copied


def _load_card_index() -> Dict[int, Card]:
    """从项目数据库加载有效卡牌 ID 索引。"""
    return {_real_card_id(c): c for c in get_all_cards() if _real_card_id(c) > 0}


def _npc_deck_limit_warnings(deck: Sequence[Card], star_map: Dict[int, int]) -> List[str]:
    """验证官方玩家卡组限制（仅用于玩家侧）。"""
    high = sum(1 for c in deck if _card_star(c, star_map) >= 4)
    five = sum(1 for c in deck if _card_star(c, star_map) == 5)
    warnings = []
    if high > 2:
        warnings.append(f"玩家卡组★4以上有{high}张，超过2张限制")
    if five > 1:
        warnings.append(f"玩家卡组★5有{five}张，超过1张限制")
    return warnings


def _npc_player_deck_legal(deck: Sequence[Card], star_map: Dict[int, int]) -> bool:
    return len(deck) == 5 and not _npc_deck_limit_warnings(deck, star_map)


def _npc_base_card_score(card: Card, star_map: Dict[int, int]) -> float:
    """评分原始卡牌强度（不考虑规则适配）。"""
    values = _side_values(card)
    high_edges = sum(1 for v in values if v >= 8)
    low_edges = sum(1 for v in values if v <= 3)
    adjacent_high = any(values[i] >= 8 and values[(i + 1) % 4] >= 8 for i in range(4))
    return sum(values) * 2.2 + high_edges * 7.0 - low_edges * 3.0 + _card_star(card, star_map) * 3.0 + (6.0 if adjacent_high else 0.0)


def _npc_rule_card_score(card: Card, rules: Sequence[str]) -> Tuple[float, List[str]]:
    """评分单卡对特殊规则的适配度。"""
    values = _side_values(card)
    score = 0.0
    reasons = []
    if "同数" in rules:
        repeats = 4 - len(set(values))
        score += repeats * 8.0
        if repeats:
            reasons.append("重复边值可服务同数")
    if "加算" in rules:
        sums = [a + b for a, b in itertools.combinations(values, 2)]
        hits = sum(1 for t in sums if t in (8, 9, 10, 11, 12, 13))
        score += min(hits * 2.5, 14.0)
        if hits >= 3:
            reasons.append("加算常见和较多")
    if "逆转" in rules:
        avg = sum(values) / 4
        score += 18.0 if avg <= 5 else -14.0 if avg >= 8 else 0.0
        if avg <= 5:
            reasons.append("低均值适合逆转")
    if "王牌杀手" in rules:
        ace_edges = sum(1 for v in values if v in (1, 10))
        score += ace_edges * 7.0
        if ace_edges:
            reasons.append("含1或A可打王牌杀手")
    if ("同类强化" in rules or "同类弱化" in rules) and card.card_type:
        score += 5.0
    return score, reasons


def _npc_matchup_card_score(card: Card, npc_cards: Sequence[Card], rules: Sequence[str]) -> float:
    """估算单卡对 NPC 卡池的直接对位优势。"""
    score = 0.0
    directions = [("up", "down"), ("right", "left"), ("down", "up"), ("left", "right")]
    for my_dir, npc_dir in directions:
        wins = ties = losses = 0
        for npc_card in npc_cards:
            result = card.compare_values(my_dir, npc_card, npc_dir, list(rules))
            wins += result == 1
            ties += result == 0
            losses += result == -1
        score += wins * 2.2 + ties * 0.4 - losses * 1.4
    return score / max(len(npc_cards), 1)


def _npc_deck_heuristic(deck: Sequence[Card], npc_cards: Sequence[Card], rules: Sequence[str],
                        star_map: Dict[int, int]) -> Tuple[float, List[str], List[str]]:
    """启发式评分候选玩家卡组。"""
    warnings = _npc_deck_limit_warnings(deck, star_map)
    rule_reasons = []
    base = sum(_npc_base_card_score(c, star_map) for c in deck) / 5
    rule_scores = []
    matchup_scores = []
    for c in deck:
        rs, reasons = _npc_rule_card_score(c, rules)
        rule_scores.append(rs)
        rule_reasons.extend(reasons)
        matchup_scores.append(_npc_matchup_card_score(c, npc_cards, rules))
    coverage = sum(1 for side in range(4) if max(_side_values(c)[side] for c in deck) >= 8) * 4.0
    score = base + sum(rule_scores) / 5 + sum(matchup_scores) / 5 + coverage
    if coverage >= 16:
        rule_reasons.append("四向都有高边覆盖")
    return score, sorted(set(rule_reasons))[:4], warnings


def _unique_cards_from_ids(ids: Sequence[int], card_index: Dict[int, Card]) -> Tuple[List[Card], List[int]]:
    """ID → 去重卡牌列表，返回未命中 ID。"""
    cards = []
    missing = []
    seen = set()
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        card = card_index.get(cid)
        if card:
            cards.append(card)
        else:
            missing.append(cid)
    return cards, missing


def _generate_player_candidates(owned: Sequence[Card], npc_cards: Sequence[Card], rules: Sequence[str],
                                star_map: Dict[int, int]) -> List[Tuple[float, List[str], List[Card]]]:
    """生成合法玩家卡组：按星等分桶，在各合法分布内取高分卡组合。

    不再对全池做硬截断，而是将 395 张卡按 ★5 / ★4 / ★1-3 分桶，
    每桶取评分最高的若干张，再按合法分布 (★5≤1, ★4≤2) 跨桶组合。
    这样低星卡不会因为评分劣势被一刀裁掉，全量卡牌都有机会参与。
    """
    # ── 1. 评分并分桶 ──
    def _score(c: Card) -> float:
        return (_npc_base_card_score(c, star_map)
                + _npc_rule_card_score(c, rules)[0]
                + _npc_matchup_card_score(c, npc_cards, rules))

    ranked = sorted(owned, key=_score, reverse=True)

    bucket_5: List[Card] = []   # ★5
    bucket_4: List[Card] = []   # ★4
    bucket_low: List[Card] = [] # ★1-3

    for c in ranked:
        star = _card_star(c, star_map)
        if star == 5:
            bucket_5.append(c)
        elif star == 4:
            bucket_4.append(c)
        else:
            bucket_low.append(c)

    # 每桶取评分最高的若干张（保证每档内仍择优，而非「平衡」）
    TOP_5  = 5
    TOP_4  = 10
    TOP_LOW = 14

    b5  = bucket_5[:TOP_5]
    b4  = bucket_4[:TOP_4]
    bl  = bucket_low[:TOP_LOW]

    # ── 2. 按合法星等分布跨桶组合 ──
    # (★5, ★4, ★1-3)  —— 均满足 ★5≤1、★4+★5≤2
    distributions = [
        (0, 0, 5),
        (0, 1, 4),
        (0, 2, 3),
        (1, 0, 4),
        (1, 1, 3),
    ]

    candidates: List[Tuple[float, List[str], List[Card]]] = []
    for n5, n4, nlow in distributions:
        if len(b5) < n5 or len(b4) < n4 or len(bl) < nlow:
            continue
        for c5 in (itertools.combinations(b5, n5) if n5 > 0 else [()]):
            for c4 in (itertools.combinations(b4, n4) if n4 > 0 else [()]):
                for cl in itertools.combinations(bl, nlow):
                    deck = list(c5) + list(c4) + list(cl)
                    score, reasons, _w = _npc_deck_heuristic(deck, npc_cards, rules, star_map)
                    candidates.append((score, reasons, deck))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:MAX_PRELIMINARY_DECKS]


def _npc_deck_strength(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> float:
    """排名 NPC 可能五张手牌（不应用玩家限制）。"""
    rule_total = sum(_npc_rule_card_score(c, rules)[0] for c in deck) / 5
    base_total = sum(_npc_base_card_score(c, star_map) for c in deck) / 5
    coverage = sum(1 for side in range(4) if max(_side_values(c)[side] for c in deck) >= 8) * 3.0
    return base_total + rule_total + coverage


def _generate_npc_decks(npc_cards: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> List[List[Card]]:
    """生成可能的 NPC 五张上场组合。"""
    if len(npc_cards) < 5:
        return []
    source = sorted(npc_cards, key=lambda c: _npc_base_card_score(c, star_map) + _npc_rule_card_score(c, rules)[0], reverse=True)
    source = source[:MAX_NPC_SOURCE_POOL]
    decks = [list(combo) for combo in itertools.combinations(source, 5)]
    decks.sort(key=lambda d: _npc_deck_strength(d, rules, star_map), reverse=True)
    return decks[:MAX_NPC_DECKS]


def _copy_deck(deck: Sequence[Card], owner: str) -> List[Card]:
    return [_clone_card(c, owner) for c in deck]


def _npc_state_score(state: GameState, my_idx: int, star_map: Dict[int, int]) -> float:
    """从玩家视角评估模拟棋盘。"""
    red_count, blue_count = state.count_cards()
    score = blue_count - red_count if my_idx == 1 else red_count - blue_count
    for row in range(3):
        for col in range(3):
            card = state.board.get_card(row, col)
            if not card:
                continue
            sign = 1 if (my_idx == 1 and card.owner == "blue") or (my_idx == 0 and card.owner == "red") else -1
            score += sign * _positional_score(row, col) * 0.30
            score += sign * max(_side_values(card)) * 0.035
            score += sign * _card_star(card, star_map) * 0.06
    return score


def _npc_choose_greedy(state: GameState, my_idx: int, star_map: Dict[int, int]) -> Optional[Tuple[Card, Tuple[int, int]]]:
    """单层贪心走法选择，用于确定性模拟。"""
    moves = state.get_available_moves()
    if not moves:
        return None
    maximizing = state.current_player_idx == my_idx
    best_move = None
    best_score = float("-inf") if maximizing else float("inf")
    for move in moves:
        card, (row, col) = move
        record = state.make_move(row, col, card)
        if record is None:
            continue
        try:
            sc = _npc_state_score(state, my_idx, star_map)
        finally:
            state.undo_move(record)
        if (maximizing and sc > best_score) or (not maximizing and sc < best_score):
            best_score = sc
            best_move = move
    return best_move


def _npc_simulate_match(player_deck: Sequence[Card], npc_deck: Sequence[Card], rules: Sequence[str],
                        star_map: Dict[int, int], player_first: bool) -> float:
    """运行一局轻量战术模拟并返回归一化分数。"""
    state = GameState(
        Board(),
        [Player("npc", _copy_deck(npc_deck, "red")), Player("player", _copy_deck(player_deck, "blue"))],
        current_player_idx=1 if player_first else 0,
        rules=list(rules),
    )
    if "同类强化" in rules or "同类弱化" in rules:
        state.recalculate_type_modifiers()
    guard = 0
    while not state.is_game_over() and guard < 9:
        guard += 1
        move = _npc_choose_greedy(state, 1, star_map)
        if move is None:
            break
        card, (row, col) = move
        state.make_move(row, col, card)
    sc = _npc_state_score(state, 1, star_map)
    return 100.0 / (1.0 + math.exp(-sc / 3.0))


def _npc_simulation_components(player_deck: Sequence[Card], npc_decks: Sequence[Sequence[Card]],
                               rules: Sequence[str], star_map: Dict[int, int]) -> Dict[str, float]:
    """对玩家卡组评估所有 NPC 可能手牌。"""
    scores = []
    for npc_deck in npc_decks:
        scores.append(_npc_simulate_match(player_deck, npc_deck, rules, star_map, player_first=True))
        scores.append(_npc_simulate_match(player_deck, npc_deck, rules, star_map, player_first=False))
    if not scores:
        return {"average": 0.0, "floor": 0.0, "min": 0.0, "games": 0}
    ordered = sorted(scores)
    floor_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.20)))
    return {
        "average": round(sum(scores) / len(scores), 2),
        "floor": round(ordered[floor_index], 2),
        "min": round(ordered[0], 2),
        "games": len(scores),
    }


def recommend_decks(owned_ids: Sequence[int], npc_ids: Sequence[int], rules=(),
                    top_n: int = 5, card_index: Optional[Dict[int, Card]] = None,
                    star_map: Optional[Dict[int, int]] = None) -> Dict:
    """NPC 卡组推荐核心。

    Args:
        owned_ids: 玩家持有卡牌 ID 列表。
        npc_ids: 已知 NPC 卡池 ID 列表。
        rules: 特殊规则。
        top_n: 返回推荐数量。
    Returns:
        JSON 可序列化的推荐报告。
    """
    rules = _parse_rules(rules)
    card_index = card_index or _load_card_index()
    star_map = star_map or get_card_star_map()
    owned_cards, missing_owned = _unique_cards_from_ids(owned_ids, card_index)
    npc_cards, missing_npc = _unique_cards_from_ids(npc_ids, card_index)
    process = [
        f"玩家持有牌：输入{len(owned_ids)}个，命中{len(owned_cards)}张",
        f"NPC牌库：输入{len(npc_ids)}个，命中{len(npc_cards)}张",
        "玩家侧应用限制：★4以上≤2，★5≤1",
        "NPC侧不应用玩家选卡限制，只从已知牌库选择5张上场牌",
    ]
    if rules:
        process.append(f"规则：{', '.join(rules)}")
    if missing_owned:
        process.append(f"未找到的玩家牌ID：{missing_owned}")
    if missing_npc:
        process.append(f"未找到的NPC牌ID：{missing_npc}")
    if len(owned_cards) < 5:
        return {"process": process + ["玩家可用牌不足5张，无法推荐"], "results": [], "best": None, "npc_decks": []}
    if len(npc_cards) < 5:
        return {"process": process + ["NPC已知牌不足5张，无法模拟"], "results": [], "best": None, "npc_decks": []}

    npc_decks = _generate_npc_decks(npc_cards, rules, star_map)
    process.append(f"NPC候选上场组合：{len(npc_decks)}组")
    candidates = _generate_player_candidates(owned_cards, npc_cards, rules, star_map)
    process.append(f"玩家合法候选卡组：{len(candidates)}组进入模拟")
    if not candidates:
        return {"process": process + ["没有找到符合玩家选卡规则的5张组合"], "results": [], "best": None, "npc_decks": []}

    results: List[DeckResult] = []
    for heuristic, heuristic_reasons, deck in candidates:
        sim = _npc_simulation_components(deck, npc_decks, rules, star_map)
        final = heuristic * 0.45 + sim["average"] * 0.40 + sim["floor"] * 0.15
        components = {
            "heuristic": round(heuristic, 2),
            "simulation_average": sim["average"],
            "simulation_floor": sim["floor"],
            "simulation_min": sim["min"],
            "games": sim["games"],
        }
        reasons = list(heuristic_reasons)
        reasons.append(f"对{len(npc_decks)}组NPC可能上场牌模拟{sim['games']}局")
        if sim["floor"] >= 50:
            reasons.append("低分位仍保持不落后")
        warnings = _npc_deck_limit_warnings(deck, star_map)
        results.append(DeckResult(
            deck=[_card_to_dict(c, star_map) for c in deck],
            final_score=round(final, 2),
            components=components,
            reasons=reasons[:6],
            warnings=warnings,
        ))
    results.sort(key=lambda r: r.final_score, reverse=True)
    best = results[0]
    process.append(f"最高分：{best.final_score}")
    process.extend([f"推荐理由：{reason}" for reason in best.reasons])
    return {
        "process": process,
        "npc_decks": [[_card_to_dict(c, star_map) for c in deck] for deck in npc_decks[:3]],
        "best": best.__dict__,
        "results": [r.__dict__ for r in results[:max(1, int(top_n))]],
    }


# ═══════════════════════════════════════════════════════════════════
# 子系统 B：选拔 / 选卡分析
# ═══════════════════════════════════════════════════════════════════


def _draft_valid_database_cards(star_map: Dict[int, int]) -> List[Card]:
    """只返回数据库中可用的有效卡牌。"""
    def _is_valid(card: Card) -> bool:
        values = [card.base_up, card.base_right, card.base_down, card.base_left]
        return bool(_real_card_id(card)) and all(1 <= v <= 10 for v in values) and _card_star(card, star_map) >= 1
    return [c for c in get_all_cards() if _is_valid(c)]


def _draft_parse_card_value(value) -> int:
    """解析 UI 传入的边值，支持 A 表示 10。"""
    if isinstance(value, str) and value.strip().upper() == "A":
        return 10
    return int(value)


def _draft_payload_value(payload: Dict, *keys):
    """读取 payload 中第一个存在且非空的值。"""
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _draft_card_sides_from_payload(payload: Dict) -> Tuple[int, int, int, int]:
    """按上/右/下/左读取卡牌数值。"""
    sides = _draft_payload_value(payload, "sides", "values")
    if sides is not None:
        return tuple(_draft_parse_card_value(value) for value in sides[:4])

    original = _draft_payload_value(payload, "original", "raw")
    if original is not None:
        if isinstance(original, str):
            original = [item.strip() for item in original.replace("，", ",").split(",")]
        raw = [_draft_parse_card_value(value) for value in original[:4]]
        return raw[3], raw[0], raw[2], raw[1]

    return (
        _draft_parse_card_value(_draft_payload_value(payload, "up", "u", "numU")),
        _draft_parse_card_value(_draft_payload_value(payload, "right", "r", "numR")),
        _draft_parse_card_value(_draft_payload_value(payload, "down", "d", "numD")),
        _draft_parse_card_value(_draft_payload_value(payload, "left", "l", "numL")),
    )


def _draft_parse_card_id(payload: Dict, fallback_id: int) -> Tuple[int, bool]:
    """读取数字卡牌 ID；非数字 UI 节点使用合成 ID。"""
    raw_id = _draft_payload_value(payload, "id", "cardId", "card_id")
    try:
        return int(raw_id), False
    except (TypeError, ValueError):
        return fallback_id, True


def _draft_mark_card(card: Card, payload: Dict, fallback_label: str, synthetic: bool = False) -> Card:
    """附加仅用于选拔分析展示的 UI 元信息。"""
    raw_id = _draft_payload_value(payload, "id", "cardId", "card_id")
    card._draft_label = str(payload.get("label") or payload.get("node") or raw_id or fallback_label)
    if synthetic:
        card._draft_synthetic = True
    if payload.get("star") not in (None, ""):
        card._draft_star = int(payload["star"])
    return card


def _draft_dict_to_card(payload: Dict, fallback_id: int = 900000, fallback_label: str = "") -> Card:
    """从 JSON 构建 Card 对象。"""
    up, right, down, left = _draft_card_sides_from_payload(payload)
    card_id, synthetic = _draft_parse_card_id(payload, fallback_id)
    card = Card(
        up, right, down, left,
        card_id=card_id,
        card_type=payload.get("type") if payload.get("type") and payload.get("type") != "无类型" else None,
    )
    if fallback_label or synthetic or payload.get("label") or payload.get("node"):
        _draft_mark_card(card, payload, fallback_label or str(card_id), synthetic)
    return card


def _draft_deck_legal(deck: Sequence[Card], star_map: Dict[int, int]) -> Tuple[bool, List[str]]:
    """验证官方卡组星级限制。"""
    high = sum(1 for c in deck if _card_star(c, star_map) >= 4)
    five = sum(1 for c in deck if _card_star(c, star_map) == 5)
    warnings = []
    if high > 2:
        warnings.append(f"★4以上卡牌有{high}张，超过2张限制")
    if five > 1:
        warnings.append(f"★5卡牌有{five}张，超过1张限制")
    return not warnings, warnings


def _draft_single_card_base(card: Card, star_map: Dict[int, int]) -> float:
    """评分原始卡牌强度（选拔版）。"""
    values = _side_values(card)
    avg = sum(values) / 4
    high_edges = sum(1 for v in values if v >= 8)
    low_edges = sum(1 for v in values if v <= 3)
    adjacent_high = (
        (card.base_up >= 8 and card.base_right >= 8)
        or (card.base_right >= 8 and card.base_down >= 8)
        or (card.base_down >= 8 and card.base_left >= 8)
        or (card.base_left >= 8 and card.base_up >= 8)
    )
    sc = avg * 6.0 + high_edges * 8.0 - low_edges * 4.0 + _card_star(card, star_map) * 2.5
    if adjacent_high:
        sc += 6.0
    if high_edges >= 3:
        sc += 8.0
    return _clamp(sc)


def _draft_rule_score_for_card(card: Card, rules: Sequence[str]) -> Tuple[float, List[str]]:
    """评分单卡对选拔规则的适配度。"""
    values = _side_values(card)
    avg = sum(values) / 4
    score = 45.0
    reasons = []
    if "同数" in rules:
        dup = 4 - len(set(values))
        score += dup * 10.0
        if dup:
            reasons.append("重复边值适合同数")
    if "加算" in rules:
        pair_sums = [a + b for a, b in itertools.combinations(values, 2)]
        common = sum(1 for v in pair_sums if v in (8, 9, 10, 11, 12, 13))
        score += min(common * 3.5, 18.0)
        if common >= 3:
            reasons.append("有多组常见加算和")
    if "逆转" in rules:
        if avg <= 5:
            score += 24.0
            reasons.append("低均值适合逆转")
        elif avg >= 8:
            score -= 20.0
    if "王牌杀手" in rules:
        special = sum(1 for v in values if v in (1, 10))
        score += special * 9.0
        if special:
            reasons.append("含1或A可利用王牌杀手")
    if "同类强化" in rules and card.card_type:
        score += 7.0
    if "同类弱化" in rules and card.card_type:
        score += 3.0
    return _clamp(score), reasons


def _draft_deck_base(deck: Sequence[Card], star_map: Dict[int, int]) -> float:
    return sum(_draft_single_card_base(c, star_map) for c in deck) / max(len(deck), 1)


def _draft_deck_rule(deck: Sequence[Card], rules: Sequence[str]) -> Tuple[float, List[str]]:
    """评分整个卡组的规则适配度。"""
    scores = []
    reasons = []
    for c in deck:
        sc, cr = _draft_rule_score_for_card(c, rules)
        scores.append(sc)
        reasons.extend(cr)
    coverage_bonus = sum(1 for side in range(4) if max(_side_values(c)[side] for c in deck) >= 8) * 3.0
    return _clamp((sum(scores) / max(len(scores), 1)) + coverage_bonus), sorted(set(reasons))[:4]


def _draft_deck_synergy(deck: Sequence[Card], rules: Sequence[str]) -> Tuple[float, List[str]]:
    """评分五张卡之间的协同效果。"""
    reasons = []
    score = 50.0
    all_values = [v for c in deck for v in _side_values(c)]
    high_by_side = {
        "上": max(c.base_up for c in deck),
        "右": max(c.base_right for c in deck),
        "下": max(c.base_down for c in deck),
        "左": max(c.base_left for c in deck),
    }
    covered = sum(1 for v in high_by_side.values() if v >= 8)
    score += covered * 6.0
    if covered == 4:
        reasons.append("四向都有高边覆盖")
    weak_edges = sum(1 for v in all_values if v <= 3)
    score -= max(0, weak_edges - 5) * 2.5
    if "同数" in rules:
        dups = sum(cnt - 1 for cnt in {v: all_values.count(v) for v in set(all_values)}.values() if cnt >= 3)
        score += min(dups * 3.0, 18.0)
        if dups >= 3:
            reasons.append("牌组有同数连携素材")
    if "加算" in rules:
        popular = 0
        for ca, cb in itertools.combinations(deck, 2):
            for va in _side_values(ca):
                for vb in _side_values(cb):
                    if va + vb in (8, 10, 12):
                        popular += 1
        score += min(popular * 0.45, 18.0)
        if popular >= 18:
            reasons.append("牌组加算组合密度高")
    type_counts: Dict[str, int] = {}
    for c in deck:
        if c.card_type:
            type_counts[c.card_type] = type_counts.get(c.card_type, 0) + 1
    if "同类强化" in rules and type_counts:
        best_type, best_count = max(type_counts.items(), key=lambda kv: kv[1])
        score += best_count * 6.0
        if best_count >= 2:
            reasons.append(f"{best_type}类型可形成同类强化")
    if "同类弱化" in rules and type_counts:
        repeats = sum(cnt - 1 for cnt in type_counts.values() if cnt > 1)
        score -= repeats * 8.0
        if repeats == 0:
            reasons.append("类型分散，适合同类弱化")
    return _clamp(score), reasons[:4]


def _draft_deck_stability(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> Tuple[float, List[str]]:
    """估算开局和抽序稳定性。"""
    per_card = [_draft_single_card_base(c, star_map) for c in deck]
    sorted_sc = sorted(per_card, reverse=True)
    top3_avg = sum(sorted_sc[:3]) / max(min(3, len(sorted_sc)), 1)
    bot2_avg = sum(sorted_sc[-2:]) / max(min(2, len(sorted_sc)), 1)
    score = top3_avg * 0.65 + bot2_avg * 0.35
    reasons = []
    stars = [_card_star(c, star_map) for c in deck]
    if max(stars) - min(stars) <= 2:
        score += 5.0
        reasons.append("星级曲线平稳")
    if "秩序" in rules or "混乱" in rules:
        score += bot2_avg * 0.08
        reasons.append("可用顺序受限时下限更重要")
    if "选拔" in rules:
        score += 4.0
    return _clamp(score), reasons


def _draft_deck_risk(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> Tuple[float, List[str]]:
    """计算风险惩罚，值越高越危险。"""
    warnings = []
    penalty = 0.0
    legal, legal_warnings = _draft_deck_legal(deck, star_map)
    if not legal:
        penalty += 45.0
        warnings.extend(legal_warnings)
    all_values = [v for c in deck for v in _side_values(c)]
    weak_edges = sum(1 for v in all_values if v <= 3)
    if weak_edges >= 8 and "逆转" not in rules:
        penalty += (weak_edges - 7) * 3.0
        warnings.append("弱边偏多，非逆转规则下容易被反吃")
    if "逆转" in rules:
        high_edges = sum(1 for v in all_values if v >= 8)
        if high_edges >= 8:
            penalty += (high_edges - 7) * 4.0
            warnings.append("高边偏多，逆转规则下风险较高")
    return _clamp(penalty), warnings


def _draft_copy_deck(deck: Sequence[Card], owner: str) -> List[Card]:
    """复制卡组并指派 owner。"""
    result = []
    for c in deck:
        item = c.copy()
        item.owner = owner
        for attr in ("_draft_label", "_draft_synthetic", "_draft_star"):
            if hasattr(c, attr):
                setattr(item, attr, getattr(c, attr))
        result.append(item)
    return result


def _draft_generate_legal_deck(cards: Sequence[Card], star_map: Dict[int, int]) -> List[Card]:
    """随机生成一套满足星级限制的合法卡组。"""
    shuffled = list(cards)
    random.shuffle(shuffled)
    deck = []
    for c in shuffled:
        candidate = deck + [c]
        if len(candidate) <= 5 and _draft_deck_legal(candidate, star_map)[0]:
            deck.append(c)
        if len(deck) == 5:
            return deck
    return list(shuffled[:5])


def _draft_normalize_selection_choice(choice, fallback_id: int, fallback_label: str) -> List[Card]:
    """将一个 UI 候选项归一化为一张或多张卡。"""
    if isinstance(choice, dict):
        if "cards" in choice:
            card_payloads = choice["cards"]
        elif any(isinstance(choice.get(key), dict) for key in ("left", "right")):
            card_payloads = [choice[key] for key in ("left", "right") if key in choice]
        elif "card" in choice:
            card_payloads = [choice["card"]]
        else:
            card_payloads = [choice]
    else:
        card_payloads = choice

    cards = []
    for card_index, card_payload in enumerate(card_payloads, start=1):
        label = f"{fallback_label}-卡{card_index}"
        cards.append(_draft_dict_to_card(card_payload, fallback_id + card_index - 1, label))
    return cards


def _draft_is_group_choice(choice) -> bool:
    """判断一个候选项是否已经是组容器，而不是单张卡。"""
    if isinstance(choice, list):
        return True
    if not isinstance(choice, dict):
        return False
    if "cards" in choice or "card" in choice:
        return True
    return any(isinstance(choice.get(key), dict) for key in ("left", "right"))


def _draft_stage_choices(stage, stage_number: int) -> List:
    """读取某个备选段的候选组，并兼容前端扁平卡牌列表。"""
    if isinstance(stage, dict):
        choices = stage.get("choices") or stage.get("candidates") or stage.get("groups") or []
    else:
        choices = stage

    if (
        stage_number in (1, 2)
        and isinstance(choices, list)
        and len(choices) == 6
        and all(not _draft_is_group_choice(choice) for choice in choices)
    ):
        return [choices[index:index + 2] for index in range(0, 6, 2)]
    return choices or []


def _draft_expand_selection_candidates(raw_candidates) -> List[Tuple[List[Card], List[str]]]:
    """把三段 UI 选拔候选展开为完整 5 张卡组。"""
    stages = raw_candidates
    if isinstance(raw_candidates, dict):
        stages = raw_candidates.get("stages") or raw_candidates.get("groups") or raw_candidates.get("choices") or []

    normalized_stages: List[List[Tuple[List[Card], str]]] = []
    for stage_index, stage in enumerate(stages, start=1):
        choices = _draft_stage_choices(stage, stage_index)

        normalized_choices = []
        for choice_index, choice in enumerate(choices, start=1):
            label = f"备选{stage_index}-候选{choice_index}"
            cards = _draft_normalize_selection_choice(choice, 900000 + stage_index * 100 + choice_index * 10, label)
            normalized_choices.append((cards, label))
        if normalized_choices:
            normalized_stages.append(normalized_choices)

    expanded = []
    for combo in itertools.product(*normalized_stages):
        deck = [card for cards, _ in combo for card in cards]
        labels = [label for _, label in combo]
        if len(deck) == 5:
            expanded.append((deck, labels))
    return expanded


def _draft_selection_groups(raw_candidates, selection: Sequence[str], star_map: Dict[int, int]) -> List[Dict]:
    """按前端三段候选结构返回已选中的组。

    Args:
        raw_candidates: 前端传入的三段候选。
        selection: 形如 ["备选1-候选2", ...] 的选择标签。
        star_map: 卡牌星级映射。
    Returns:
        每个备选段选中组的卡牌详情，保留卡牌 id 和上/右/下/左数值。
    """
    stages = raw_candidates
    if isinstance(raw_candidates, dict):
        stages = raw_candidates.get("stages") or raw_candidates.get("groups") or raw_candidates.get("choices") or []

    groups = []
    for label in selection:
        try:
            stage_text, group_text = label.split("-")
            stage_index = int(stage_text.replace("备选", "")) - 1
            group_index = int(group_text.replace("候选", "")) - 1
        except (ValueError, IndexError):
            continue

        if stage_index < 0 or stage_index >= len(stages):
            continue
        stage = stages[stage_index]
        choices = _draft_stage_choices(stage, stage_index + 1)
        if group_index < 0 or group_index >= len(choices):
            continue

        cards = _draft_normalize_selection_choice(
            choices[group_index],
            900000 + (stage_index + 1) * 100 + (group_index + 1) * 10,
            label,
        )
        groups.append({
            "stage": stage_index + 1,
            "group": group_index + 1,
            "label": label,
            "cards": [_card_to_dict(card, star_map) for card in cards],
        })
    return groups


def _draft_build_random_opponent(all_cards: Sequence[Card], star_map: Dict[int, int], avoid_ids: set) -> List[Card]:
    """构建随机合法对手卡组用于深算验证。"""
    available = [c for c in all_cards if _real_card_id(c) not in avoid_ids]
    deck = _draft_generate_legal_deck(available or list(all_cards), star_map)
    return _draft_copy_deck(deck, "red")


def _draft_opponent_threat(card: Card, my_deck: Sequence[Card], rules: Sequence[str],
                           star_map: Dict[int, int]) -> float:
    """估算对手单卡对我方卡组的威胁程度。"""
    sc, _ = _draft_rule_score_for_card(card, rules)
    sc += _draft_single_card_base(card, star_map) * 0.35
    directions = [("up", "down"), ("right", "left"), ("down", "up"), ("left", "right")]
    for opp_dir, my_dir in directions:
        for my_card in my_deck:
            result = card.compare_values(opp_dir, my_card, my_dir, list(rules))
            if result == 1:
                sc += 4.0
            elif result == 0:
                sc += 0.7
            ov = card.get_effective_value(opp_dir, list(rules))
            mv = my_card.get_effective_value(my_dir, list(rules))
            if "同数" in rules and ov == mv:
                sc += 2.4
            if "加算" in rules and ov + mv in (8, 10, 12):
                sc += 1.8
    return sc


def _draft_build_threat_opponent(all_cards: Sequence[Card], star_map: Dict[int, int],
                                 avoid_ids: set, my_deck: Sequence[Card],
                                 rules: Sequence[str]) -> List[Card]:
    """构建偏向威胁我方的对手卡组。"""
    available = [c for c in all_cards if _real_card_id(c) not in avoid_ids]
    ranked = sorted(
        available or list(all_cards),
        key=lambda c: _draft_opponent_threat(c, my_deck, rules, star_map),
        reverse=True,
    )
    top_pool = ranked[:min(36, len(ranked))]
    deck = []
    attempts = 0
    while len(deck) < 5 and attempts < 80:
        attempts += 1
        pick_pool = top_pool[:max(8, min(len(top_pool), 16 + attempts // 8))]
        candidate = random.choice(pick_pool)
        if any(_real_card_id(c) == _real_card_id(candidate) for c in deck):
            continue
        if _draft_deck_legal(deck + [candidate], star_map)[0]:
            deck.append(candidate)
    if len(deck) < 5:
        for candidate in ranked:
            if any(_real_card_id(c) == _real_card_id(candidate) for c in deck):
                continue
            if _draft_deck_legal(deck + [candidate], star_map)[0]:
                deck.append(candidate)
            if len(deck) == 5:
                break
    return _draft_copy_deck(deck[:5], "red")


def _draft_state_score(state: GameState, ai_idx: int, star_map: Dict[int, int]) -> float:
    """从选拔卡组视角评估模拟棋盘。"""
    red_count, blue_count = state.count_cards()
    score = blue_count - red_count if ai_idx == 1 else red_count - blue_count
    for row in range(3):
        for col in range(3):
            card = state.board.get_card(row, col)
            if not card:
                continue
            sign = 1 if (
                (ai_idx == 1 and card.owner == "blue") or (ai_idx == 0 and card.owner == "red")
            ) else -1
            score += sign * _positional_score(row, col) * 0.35
            score += sign * max(_side_values(card)) * 0.035
            score += sign * _card_star(card, star_map) * 0.08
    return score


def _draft_choose_greedy(state: GameState, ai_idx: int, star_map: Dict[int, int]) -> Tuple:
    """单层贪心走法（选拔版）。"""
    best_move = None
    best_score = float("-inf") if state.current_player_idx == ai_idx else float("inf")
    for move in state.get_available_moves():
        card, (row, col) = move
        record = state.make_move(row, col, card)
        if record is None:
            continue
        try:
            sc = _draft_state_score(state, ai_idx, star_map)
        finally:
            state.undo_move(record)
        sc += random.uniform(-0.08, 0.08)
        if state.current_player_idx == ai_idx:
            if sc > best_score:
                best_score = sc
                best_move = move
        else:
            if sc < best_score:
                best_score = sc
                best_move = move
    return best_move


def _draft_simulate_match(deck: Sequence[Card], opp_deck: Sequence[Card], rules: Sequence[str],
                          star_map: Dict[int, int], my_first: bool) -> float:
    """轻量战术模拟并返回归一化结果（选拔版）。"""
    my_hand = _draft_copy_deck(deck, "blue")
    opp_hand = _draft_copy_deck(opp_deck, "red")
    state = GameState(
        Board(),
        [Player("opp", opp_hand), Player("me", my_hand)],
        current_player_idx=1 if my_first else 0,
        rules=list(rules),
    )
    if "同类强化" in rules or "同类弱化" in rules:
        state.recalculate_type_modifiers()
    guard = 0
    while not state.is_game_over() and guard < 9:
        guard += 1
        move = _draft_choose_greedy(state, 1, star_map)
        if not move:
            break
        card, (row, col) = move
        state.make_move(row, col, card)
    sc = _draft_state_score(state, 1, star_map)
    return 100.0 / (1.0 + math.exp(-sc / 3.0))


def _draft_verification_score(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int],
                              all_cards: Sequence[Card], deadline: float, min_rounds: int = 2) -> Tuple[float, Dict]:
    """反复轻量对战直到预算耗尽。"""
    scores = []
    avoid_ids = {_real_card_id(c) for c in deck}
    rounds = 0
    threat_rounds = 0
    while (time.perf_counter() < deadline or rounds < min_rounds) and rounds < 2000:
        use_threat = rounds % 3 == 2
        if use_threat:
            opp = _draft_build_threat_opponent(all_cards, star_map, avoid_ids, deck, rules)
            threat_rounds += 1
        else:
            opp = _draft_build_random_opponent(all_cards, star_map, avoid_ids)
        scores.append(_draft_simulate_match(deck, opp, rules, star_map, my_first=True))
        scores.append(_draft_simulate_match(deck, opp, rules, star_map, my_first=False))
        rounds += 1
        if time.perf_counter() >= deadline and rounds >= min_rounds:
            break
    if not scores:
        return 50.0, {"rounds": 0, "average": 50.0, "floor": 50.0, "min": 50.0}
    avg = sum(scores) / len(scores)
    srt = sorted(scores)
    floor_idx = min(len(srt) - 1, max(0, int(len(srt) * 0.10)))
    verified = avg * 0.65 + srt[floor_idx] * 0.35
    return _clamp(verified), {
        "rounds": rounds,
        "threat_rounds": threat_rounds,
        "simulations": len(scores),
        "average": round(avg, 2),
        "floor": round(srt[floor_idx], 2),
        "min": round(srt[0], 2),
    }


def _draft_score_deck(deck: Sequence[Card], rules: Sequence[str], weights: Dict[str, float],
                      star_map: Dict[int, int], verification=None,
                      selection: List[str] = None, groups: List[Dict] = None) -> DeckResult:
    """评分一套选拔卡组并生成解释详情。"""
    base = _draft_deck_base(deck, star_map)
    rule, rule_reasons = _draft_deck_rule(deck, rules)
    synergy, synergy_reasons = _draft_deck_synergy(deck, rules)
    stability, stability_reasons = _draft_deck_stability(deck, rules, star_map)
    risk, warnings = _draft_deck_risk(deck, rules, star_map)
    ver_val, ver_meta = verification or (
        50.0,
        {"rounds": 0, "simulations": 0, "average": 50.0, "floor": 50.0, "min": 50.0},
    )
    w_sum = max(weights["base"] + weights["rule"] + weights["synergy"] + weights["stability"] + weights["verification"], 0.001)
    final = (
        base * weights["base"]
        + rule * weights["rule"]
        + synergy * weights["synergy"]
        + stability * weights["stability"]
        + ver_val * weights["verification"]
    ) / w_sum
    final -= risk * weights["risk"]
    reasons = (rule_reasons + synergy_reasons + stability_reasons) or ["整体表现均衡"]
    if ver_meta.get("simulations", 0):
        reasons.append(
            f"深算{ver_meta['simulations']}局，威胁采样{ver_meta.get('threat_rounds', 0)}轮，"
            f"均值{ver_meta['average']}，P10下限{ver_meta['floor']}"
        )
    return DeckResult(
        deck=[_card_to_dict(c, star_map) for c in deck],
        final_score=round(_clamp(final), 2),
        components={
            "base": round(base, 2),
            "rule": round(rule, 2),
            "synergy": round(synergy, 2),
            "stability": round(stability, 2),
            "verification": round(ver_val, 2),
            "risk": round(risk, 2),
        },
        reasons=reasons[:6],
        warnings=warnings,
        selection=selection or [],
        groups=groups or [],
    )


def _draft_verify_worker(payload: Tuple[List[Card], List[str], Dict[int, int], List[Card], float]) -> Tuple[float, Dict]:
    """进程池 worker。"""
    deck, rules, star_map, all_cards, seconds = payload
    deadline = time.perf_counter() + seconds
    return _draft_verification_score(deck, rules, star_map, all_cards, deadline)


def _draft_normalize_weights(payload: Dict) -> Dict[str, float]:
    """读取 UI 权重，缺失用默认值。"""
    weights = dict(DEFAULT_WEIGHTS)
    for key in weights:
        if key in payload:
            weights[key] = max(0.0, float(payload[key]))
    return weights


# ═══════════════════════════════════════════════════════════════════
# 子系统 C：胜率估算
# ═══════════════════════════════════════════════════════════════════

WINRATE_DEFAULT_GAMES = 20


def _winrate_simulate_match(player_deck: Sequence[Card], opp_deck: Sequence[Card],
                            rules: Sequence[str], star_map: Dict[int, int],
                            player_first: bool) -> Tuple[int, float]:
    """模拟一局并返回 (胜负: 1=胜/0=平/-1=负, 原始棋盘分)。

    复用 _npc_simulate_match 的贪心模拟核心，
    但直接返回胜负判定 + 原始分数，不做归一化。
    """
    state = GameState(
        Board(),
        [Player("opp", _copy_deck(opp_deck, "red")),
         Player("player", _copy_deck(player_deck, "blue"))],
        current_player_idx=1 if player_first else 0,
        rules=list(rules),
    )
    if "同类强化" in rules or "同类弱化" in rules:
        state.recalculate_type_modifiers()
    guard = 0
    while not state.is_game_over() and guard < 9:
        guard += 1
        move = _npc_choose_greedy(state, 1, star_map)
        if move is None:
            break
        card, (row, col) = move
        state.make_move(row, col, card)
    sc = _npc_state_score(state, 1, star_map)
    if sc > 0:
        outcome = 1
    elif sc < 0:
        outcome = -1
    else:
        outcome = 0
    return outcome, sc


def _winrate_stats(player_deck: Sequence[Card], opp_deck: Sequence[Card],
                   rules: Sequence[str], star_map: Dict[int, int],
                   games: int) -> Dict:
    """对单组对手模拟 N 局（交替先手），返回统计字典。"""
    wins = losses = draws = 0
    first_wins = first_losses = first_draws = 0
    second_wins = second_losses = second_draws = 0
    scores = []

    for i in range(games):
        player_first = (i % 2 == 0)
        outcome, sc = _winrate_simulate_match(
            player_deck, opp_deck, rules, star_map, player_first,
        )
        scores.append(100.0 / (1.0 + math.exp(-sc / 3.0)))
        if player_first:
            if outcome == 1:
                first_wins += 1
            elif outcome == -1:
                first_losses += 1
            else:
                first_draws += 1
        else:
            if outcome == 1:
                second_wins += 1
            elif outcome == -1:
                second_losses += 1
            else:
                second_draws += 1
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            draws += 1

    first_total = max(first_wins + first_losses + first_draws, 1)
    second_total = max(second_wins + second_losses + second_draws, 1)
    return {
        "winRate": round(wins / games * 100, 1),
        "lossRate": round(losses / games * 100, 1),
        "drawRate": round(draws / games * 100, 1),
        "games": games,
        "simulations": games,  # 兼容旧前端字段
        "avgScore": round(sum(scores) / len(scores), 1),
        "firstWinRate": round(first_wins / first_total * 100, 1),
        "firstLossRate": round(first_losses / first_total * 100, 1),
        "secondWinRate": round(second_wins / second_total * 100, 1),
        "secondLossRate": round(second_losses / second_total * 100, 1),
    }


_OpponentPayload = Tuple[
    Sequence[Card],  # player_deck
    Sequence[Card],  # opp_deck
    List[str],       # rules
    Dict[int, int],  # star_map
    int,             # games
    str,             # name
    List[Dict],      # opp_serialized (for response)
]


def _winrate_worker(payload: _OpponentPayload) -> Dict:
    """进程池 worker：模拟玩家牌组对单个对手。"""
    player_deck, opp_deck, rules, star_map, games, name, opp_serialized = payload
    stats = _winrate_stats(player_deck, opp_deck, rules, star_map, games)
    stats["name"] = name
    stats["deck"] = opp_serialized
    return stats


def estimate_winrate(player_deck_raw: List[Dict], opponents: List[Dict],
                     rules: Sequence[str] = (),
                     games_per_opponent: int = WINRATE_DEFAULT_GAMES,
                     star_map: Optional[Dict[int, int]] = None) -> Dict:
    """胜率估算核心。

    Args:
        player_deck_raw: 玩家 5 张上场牌 JSON。
        opponents: [{"name": str, "deck": [Card, ...], "poolIds": [...]}, ...]
        rules: 特殊规则。
        games_per_opponent: 每个对手模拟局数（默认 20）。
        star_map: 星级映射。

    Returns:
        {process, results}
    """
    star_map = star_map or get_card_star_map()
    rules = _parse_rules(rules)
    player_cards = [_draft_dict_to_card(item) for item in player_deck_raw]

    process = [
        f"玩家牌组：{len(player_cards)}张",
        f"规则：{', '.join(rules) if rules else '无'}",
    ]
    if len(player_cards) != 5:
        return {"process": process + ["玩家牌组必须为5张"], "results": []}

    # 过滤有效对手（deck 非空）
    valid_opponents = []
    for opp in opponents:
        opp_deck_raw = opp.get("deck", [])
        if not opp_deck_raw or len(opp_deck_raw) != 5:
            continue
        valid_opponents.append(opp)

    process.append(f"有效对手：{len(valid_opponents)}个")

    # 调试日志：记下收到的卡牌
    if player_cards:
        player_ids = [f"{c.card_id}(U{c.base_up}R{c.base_right}D{c.base_down}L{c.base_left})" for c in player_cards]
        process.append(f"玩家卡牌：[{', '.join(player_ids)}]")
    for opp in valid_opponents:
        opp_cards = [_draft_dict_to_card(item) for item in opp.get("deck", [])]
        if opp_cards:
            opp_ids = [f"{c.card_id}(U{c.base_up}R{c.base_right}D{c.base_down}L{c.base_left})" for c in opp_cards]
            process.append(f"{opp.get('name', '对手')}卡牌：[{', '.join(opp_ids)}]")

    if not valid_opponents:
        return {"process": process + ["没有有效的对手牌组"], "results": []}

    # ── 多进程模拟 ──
    worker_count = min(len(valid_opponents), os.cpu_count() or len(valid_opponents))
    process.append(f"并行进程：{worker_count}个，每个对手模拟{games_per_opponent}局")

    tasks: List[_OpponentPayload] = []
    for opp in valid_opponents:
        opp_cards = [_draft_dict_to_card(item) for item in opp["deck"]]
        tasks.append((
            player_cards,
            opp_cards,
            list(rules),
            star_map,
            games_per_opponent,
            opp.get("name", "未知"),
            opp["deck"],  # 序列化后的牌组，直接回传
        ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_winrate_worker, tasks))

    # 按胜率降序排列
    results.sort(key=lambda r: r["winRate"], reverse=True)

    best = results[0] if results else None
    if best:
        process.append(f"最高胜率：{best['name']} ({best['winRate']}%)")

    return {
        "process": process,
        "results": results,
    }


def analyze_candidates(mode: str, raw_candidates: List, rules: List[str], weights: Dict[str, float],
                       time_budget: float = DEFAULT_TIME_BUDGET) -> Dict:
    """选拔分析核心：分析固定卡组或枚举卡池组合。

    Args:
        mode: "decks"（三套固定卡组）或 "pool"（卡池枚举）。
        raw_candidates: 前端传入的候选数据。
        rules: 规则列表（至少包含 "选拔"）。
        weights: 评分权重。
        time_budget: 深算秒数预算。
    Returns:
        {process, best, results}
    """
    star_map = get_card_star_map()
    all_cards = _draft_valid_database_cards(star_map)
    time_budget = max(5.0, min(20.0, float(time_budget)))
    started = time.perf_counter()
    process = [
        f"规则：{', '.join(rules)}",
        f"权重：{weights}",
        f"深算预算：{time_budget:.1f}秒",
        "对手采样：约2/3随机合法选拔牌组 + 1/3高威胁合法选拔牌组",
    ]
    preliminary: List[Tuple[DeckResult, List[Card], List[str]]] = []
    selection_mode = mode in ("selection", "choices", "draft")

    if mode == "pool":
        pool = [_draft_dict_to_card(item, 910000 + index) for index, item in enumerate(raw_candidates)]
        process.append(f"候选卡池：{len(pool)}张")
        legal_count = 0
        scanned = 0
        for combo in itertools.combinations(pool, 5):
            scanned += 1
            if scanned > MAX_POOL_ENUMERATION:
                process.append(f"组合数超过上限，仅扫描前{MAX_POOL_ENUMERATION}组")
                break
            if not _draft_deck_legal(combo, star_map)[0]:
                continue
            legal_count += 1
            preliminary.append((_draft_score_deck(combo, rules, weights, star_map), list(combo), []))
        process.append(f"合法5张组合：{legal_count}组")
        preliminary.sort(key=lambda item: item[0].final_score, reverse=True)
        preliminary = preliminary[:POOL_VERIFY_LIMIT]
        process.append(f"进入深算验证：启发式前{len(preliminary)}组")
    elif selection_mode:
        expanded = _draft_expand_selection_candidates(raw_candidates)
        process.append(f"选拔段：{len(raw_candidates) if isinstance(raw_candidates, list) else 0}段")
        process.append(f"展开完整卡组：{len(expanded)}套")
        preliminary = [
            (
                _draft_score_deck(
                    deck,
                    rules,
                    weights,
                    star_map,
                    selection=labels,
                    groups=_draft_selection_groups(raw_candidates, labels, star_map),
                ),
                deck,
                labels,
            )
            for deck, labels in expanded
            if _draft_deck_legal(deck, star_map)[0]
        ]
        process.append(f"合法完整卡组：{len(preliminary)}套")
        preliminary.sort(key=lambda item: item[0].final_score, reverse=True)
    else:
        candidate_decks = []
        for deck_index, deck_payload in enumerate(raw_candidates):
            candidate_decks.append([
                _draft_dict_to_card(item, 920000 + deck_index * 10 + card_index)
                for card_index, item in enumerate(deck_payload)
            ])
        process.append(f"固定卡组：{len(candidate_decks)}套")
        preliminary = [(_draft_score_deck(deck, rules, weights, star_map), deck, []) for deck in candidate_decks]
        preliminary.sort(key=lambda item: item[0].final_score, reverse=True)

    if not preliminary:
        return {"process": process + ["没有找到合法卡组"], "results": [], "best": None}

    scored = []
    total = len(preliminary)
    use_parallel = mode == "decks" and total > 1
    if use_parallel:
        workers = min(total, os.cpu_count() or total)
        process.append(f"并行深算：{workers}个进程，每套约{time_budget:.1f}秒")
        tasks = [(deck, list(rules), star_map, all_cards, time_budget) for _, deck, _ in preliminary]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            verifications = list(executor.map(_draft_verify_worker, tasks))
        for idx, ((_, deck, labels), ver) in enumerate(zip(preliminary, verifications), start=1):
            scored.append(_draft_score_deck(
                deck,
                rules,
                weights,
                star_map,
                verification=ver,
                selection=labels,
                groups=_draft_selection_groups(raw_candidates, labels, star_map) if selection_mode else [],
            ))
            process.append(
                f"深算 {idx}/{total}: {ver[1]['simulations']}局，"
                f"威胁{ver[1].get('threat_rounds', 0)}轮，"
                f"均值{ver[1]['average']}，P10下限{ver[1]['floor']}"
            )
    else:
        process.append("顺序深算：共享总预算")
        for idx, (_, deck, labels) in enumerate(preliminary, start=1):
            now = time.perf_counter()
            remaining = max(total - idx + 1, 1)
            remaining_budget = max(0.2, started + time_budget - now)
            deck_deadline = now + remaining_budget / remaining
            ver = _draft_verification_score(deck, rules, star_map, all_cards, deck_deadline)
            scored.append(_draft_score_deck(
                deck,
                rules,
                weights,
                star_map,
                verification=ver,
                selection=labels,
                groups=_draft_selection_groups(raw_candidates, labels, star_map) if selection_mode else [],
            ))
            process.append(
                f"深算 {idx}/{total}: {ver[1]['simulations']}局，"
                f"威胁{ver[1].get('threat_rounds', 0)}轮，"
                f"均值{ver[1]['average']}，P10下限{ver[1]['floor']}"
            )

    scored.sort(key=lambda r: r.final_score, reverse=True)
    best = scored[0]
    elapsed = time.perf_counter() - started
    process.append(f"实际耗时：{elapsed:.2f}秒")
    process.append(f"最高分：{best.final_score}")
    if best.selection:
        process.append(f"推荐选择：{' + '.join(best.selection)}")
    process.extend([f"推荐理由：{reason}" for reason in best.reasons])
    if best.warnings:
        process.extend([f"风险提示：{w}" for w in best.warnings])
    return {
        "process": process,
        "best": best.__dict__,
        "recommendation": best.groups,
        "results": [r.__dict__ for r in scored],
    }


# ═══════════════════════════════════════════════════════════════════
# 路由（注册到继承自 ai_server 的 app 上）
# ═══════════════════════════════════════════════════════════════════


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """NPC 卡组推荐。

    请求 JSON:
        ownedIds:  int[]   — 玩家持有卡牌 ID
        npcIds:    int[]   — NPC 已知卡池 ID
        rules:     str[]   — 特殊规则 (可选)
        topN:      int     — 返回推荐数量 (默认 5)

    响应 JSON:
        process:   str[]   — 分析过程日志
        npc_decks: Card[][]— NPC 候选上场组合 (最多3组)
        best:      DeckResult — 最优推荐
        results:   DeckResult[] — 所有推荐
    """
    payload = request.get_json() or {}
    result = recommend_decks(
        _parse_id_list(payload.get("ownedIds", [])),
        _parse_id_list(payload.get("npcIds", [])),
        payload.get("rules", []),
        int(payload.get("topN", 5)),
    )
    return jsonify(result)


@app.route("/api/draft/analyze", methods=["POST"])
def api_draft_analyze():
    """选拔深算分析：对三段选拔候选在「选拔+可选规则」下评分，返回最优。

    请求 JSON:
        mode:       "selection"（默认，也兼容旧的 "decks" | "pool"）
        rules:      str[]              — 完整规则列表，如 ["选拔", "加算"]
        candidates: DraftStage[]       — 三段候选：备选1/2各3组双卡，备选3为3组单卡
        weights:    {base, rule, synergy, stability, verification, risk}  (可选)
        timeBudget: float              — 深算秒数 (默认 7.0)

    响应 JSON:
        process:        str[]       — 分析过程日志
        recommendation: Dict[]      — 按组选出的推荐组合
        best:           DeckResult  — 最优卡组，含 groups
        results:        DeckResult[]— 所有评分结果
    """
    payload = request.get_json() or {}
    rule_payload = payload.get("rules")
    if rule_payload is not None:
        if isinstance(rule_payload, str):
            rule_payload = [item.strip() for item in rule_payload.replace("，", ",").split(",")]
        extra = [rule for rule in rule_payload if rule != "选拔"]
    else:
        extra = payload.get("extraRules", [])
        if isinstance(extra, str):
            extra = [extra]
    rules = ["选拔"] + [rule for rule in extra if rule in EXTRA_RULES][:2]
    weights = _draft_normalize_weights(payload.get("weights", {}))
    candidates = payload.get("candidates", payload.get("groups", payload.get("draftChoices", payload.get("choices", []))))
    result = analyze_candidates(
        payload.get("mode", "selection"),
        candidates,
        rules,
        weights,
        float(payload.get("timeBudget", DEFAULT_TIME_BUDGET)),
    )
    return jsonify(result)


@app.route("/api/winrate", methods=["POST"])
def api_winrate():
    """胜率估算：玩家选定牌组后，对每个对手（NPC / PVP）模拟对局并返回胜率。

    请求 JSON:
        playerDeck:  Card[]     — 玩家 5 张上场牌
        opponents:   [{name, deck: Card[], poolIds: int[]}]  — 对手列表
        rules:       str[]      — 特殊规则 (可选)
        gamesPerOpponent: int   — 每对手模拟局数 (默认 20)

    响应 JSON:
        process: str[]          — 分析过程日志
        results: [{name, deck, winRate, lossRate, drawRate, games, avgScore, ...}]
    """
    payload = request.get_json() or {}
    result = estimate_winrate(
        payload.get("playerDeck", []),
        payload.get("opponents", []),
        payload.get("rules", []),
        int(payload.get("gamesPerOpponent", WINRATE_DEFAULT_GAMES)),
    )
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════


def self_test() -> Dict:
    """运行所有子系统的冒烟测试。"""
    results = {}

    # ── NPC 推荐自测 ──
    card_index = _load_card_index()
    owned_ids = list(card_index.keys())[:35]
    npc_ids = list(card_index.keys())[40:50]
    results["recommend"] = recommend_decks(owned_ids, npc_ids, ["加算"], top_n=3, card_index=card_index)
    results["recommend"]["process"] = results["recommend"]["process"][:3] + ["..."]  # 截断

    # ── 选拔分析自测 ──
    star_map = get_card_star_map()
    cards = _draft_valid_database_cards(star_map)
    decks = [[_card_to_dict(c, star_map) for c in _draft_generate_legal_deck(cards, star_map)] for _ in range(3)]
    results["draft"] = analyze_candidates("decks", decks, ["选拔", "加算"], DEFAULT_WEIGHTS, time_budget=5.0)
    results["draft"]["process"] = results["draft"]["process"][:3] + ["..."]

    return results

ascii_text = [
"████████╗████████╗ ██████╗    ███████╗██╗██████╗ ███████╗███╗   ██╗",
"╚══██╔══╝╚══██╔══╝██╔════╝    ██╔════╝██║██╔══██╗██╔════╝████╗  ██║",
"   ██║      ██║   ██║         ███████╗██║██████╔╝█████╗  ██╔██╗ ██║",
"   ██║      ██║   ██║         ╚════██║██║██╔══██╗██╔══╝  ██║╚██╗██║",
"   ██║      ██║   ╚██████╗    ███████║██║██║  ██║███████╗██║ ╚████║",
"   ╚═╝      ╚═╝    ╚═════╝    ╚══════╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝",
"",
"                                                Triple Triad Solver"
]
particles = ['.', ',', ':', '*', '+', '·', ' ']

def color_char(char, x, width):
    r = 255
    g = int(180 - 120 * (x / width))
    b = int(200 - 80 * (x / width))
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def render(frame):
    output = []
    for line in ascii_text:
        new_line = ""
        width = len(line)

        for i, ch in enumerate(line):
            decay_chance = max(0, (i - width * 0.6) / (width * 0.4))

            if ch != " " and random.random() < decay_chance + frame * 0.02:
                new_line += random.choice(particles)
            else:
                new_line += color_char(ch, i, width)

        output.append(new_line)

    return "\n".join(output)
    
    
# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    os.system('title TTC_Siren - Triple Triad AI Server')
    os.system('cls' if os.name == 'nt' else 'clear')
    print(render(0))
    parser = argparse.ArgumentParser(description="统一 Triple Triad AI 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="监听端口 (默认 5000)")
    parser.add_argument("--self-test", action="store_true", help="运行冒烟测试后退出")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        sys.exit(0)

    # 抑制 Flask 启动日志
    import logging
    log = logging.getLogger("werkzeug")
    #log.setLevel(logging.ERROR)

    print(f" * 统一 Triple Triad AI 服务运行在 http://{args.host}:{args.port}")
    print(f"   求解器:     POST /ai_move")
    print(f"   NPC推荐:   POST /api/recommend")
    print(f"   选拔分析:   POST /api/draft/analyze")
    print(f"   胜率估算:   POST /api/winrate")
    print(f"   Press CTRL+C to quit")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
