# -*- coding: utf-8 -*-
"""
Standalone NPC matchup deck recommender demo.

Input the player's owned card ids and an NPC's known card pool ids. The player
deck is always filtered by official deck limits, while the NPC pool is only
required to provide five playable cards because some NPCs can ignore those
limits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from flask import Flask, jsonify, request

from ai_server import get_all_cards, get_card_star_map
from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player


MAX_PLAYER_POOL = 24
MAX_PRELIMINARY_DECKS = 48
MAX_NPC_SOURCE_POOL = 18
MAX_NPC_DECKS = 10
EXTRA_RULES = {"同数", "加算", "逆转", "王牌杀手", "同类强化", "同类弱化", "秩序", "混乱"}


@dataclass
class Recommendation:
    deck: List[Dict]
    score: float
    components: Dict[str, float]
    reasons: List[str]
    warnings: List[str]


app = Flask(__name__)


def real_card_id(card: Card) -> int:
    """Return the base id for normal or generated card ids."""
    card_id = int(card.card_id or 0)
    return card_id % 1000 if card_id >= 1000 else card_id


def clone_card(card: Card, owner: Optional[str] = None) -> Card:
    """Copy a card and optionally assign owner."""
    copied = card.copy()
    copied.owner = owner
    copied.can_use = True
    return copied


def load_card_index() -> Dict[int, Card]:
    """Load valid cards from the project database by id."""
    return {real_card_id(card): card for card in get_all_cards() if real_card_id(card) > 0}


def parse_id_list(value: str | Iterable[int]) -> List[int]:
    """Parse comma/space separated ids or an iterable of ids."""
    if isinstance(value, str):
        parts = value.replace("，", ",").replace(" ", ",").split(",")
        return [int(item) for item in parts if item.strip()]
    return [int(item) for item in value]


def parse_rules(value: str | Iterable[str]) -> List[str]:
    """Parse rule text and keep supported extra rules."""
    if isinstance(value, str):
        raw_rules = [item.strip() for item in value.replace("，", ",").split(",")]
    else:
        raw_rules = [str(item).strip() for item in value]
    return [rule for rule in raw_rules if rule in EXTRA_RULES]


def card_star(card: Card, star_map: Dict[int, int]) -> int:
    """Return card rarity with a conservative fallback."""
    return int(star_map.get(real_card_id(card), 1))


def deck_limit_warnings(deck: Sequence[Card], star_map: Dict[int, int]) -> List[str]:
    """Validate official player deck limits."""
    high_count = sum(1 for card in deck if card_star(card, star_map) >= 4)
    five_count = sum(1 for card in deck if card_star(card, star_map) == 5)
    warnings = []
    if high_count > 2:
        warnings.append(f"玩家卡组★4以上有{high_count}张，超过2张限制")
    if five_count > 1:
        warnings.append(f"玩家卡组★5有{five_count}张，超过1张限制")
    return warnings


def player_deck_is_legal(deck: Sequence[Card], star_map: Dict[int, int]) -> bool:
    """Return whether a player deck satisfies official limits."""
    return len(deck) == 5 and not deck_limit_warnings(deck, star_map)


def card_to_dict(card: Card, star_map: Dict[int, int]) -> Dict:
    """Serialize card data for CLI/API output."""
    return {
        "id": real_card_id(card),
        "up": card.base_up,
        "right": card.base_right,
        "down": card.base_down,
        "left": card.base_left,
        "star": card_star(card, star_map),
        "type": card.card_type or "",
    }


def side_values(card: Card) -> List[int]:
    """Return card side values in up/right/down/left order."""
    return [card.base_up, card.base_right, card.base_down, card.base_left]


def base_card_score(card: Card, star_map: Dict[int, int]) -> float:
    """Score raw card strength before matchup adjustments."""
    values = side_values(card)
    high_edges = sum(1 for value in values if value >= 8)
    low_edges = sum(1 for value in values if value <= 3)
    adjacent_high = any(values[i] >= 8 and values[(i + 1) % 4] >= 8 for i in range(4))
    return sum(values) * 2.2 + high_edges * 7.0 - low_edges * 3.0 + card_star(card, star_map) * 3.0 + (6.0 if adjacent_high else 0.0)


def rule_card_score(card: Card, rules: Sequence[str]) -> Tuple[float, List[str]]:
    """Score one card's fit for current special rules."""
    values = side_values(card)
    score = 0.0
    reasons = []
    if "同数" in rules:
        repeats = 4 - len(set(values))
        score += repeats * 8.0
        if repeats:
            reasons.append("重复边值可服务同数")
    if "加算" in rules:
        sums = [a + b for a, b in itertools.combinations(values, 2)]
        hits = sum(1 for total in sums if total in (8, 9, 10, 11, 12, 13))
        score += min(hits * 2.5, 14.0)
        if hits >= 3:
            reasons.append("加算常见和较多")
    if "逆转" in rules:
        average = sum(values) / 4
        score += 18.0 if average <= 5 else -14.0 if average >= 8 else 0.0
        if average <= 5:
            reasons.append("低均值适合逆转")
    if "王牌杀手" in rules:
        ace_edges = sum(1 for value in values if value in (1, 10))
        score += ace_edges * 7.0
        if ace_edges:
            reasons.append("含1或A可打王牌杀手")
    if ("同类强化" in rules or "同类弱化" in rules) and card.card_type:
        score += 5.0
    return score, reasons


def matchup_card_score(card: Card, npc_cards: Sequence[Card], rules: Sequence[str]) -> float:
    """Estimate one owned card's direct edge against the NPC card pool."""
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


def deck_heuristic_score(deck: Sequence[Card], npc_cards: Sequence[Card], rules: Sequence[str],
                         star_map: Dict[int, int]) -> Tuple[float, List[str], List[str]]:
    """Score a candidate player deck before tactical simulation."""
    warnings = deck_limit_warnings(deck, star_map)
    rule_reasons = []
    base = sum(base_card_score(card, star_map) for card in deck) / 5
    rule_scores = []
    matchup_scores = []
    for card in deck:
        rule_score, reasons = rule_card_score(card, rules)
        rule_scores.append(rule_score)
        rule_reasons.extend(reasons)
        matchup_scores.append(matchup_card_score(card, npc_cards, rules))
    coverage = sum(1 for side in range(4) if max(side_values(card)[side] for card in deck) >= 8) * 4.0
    score = base + sum(rule_scores) / 5 + sum(matchup_scores) / 5 + coverage
    if coverage >= 16:
        rule_reasons.append("四向都有高边覆盖")
    return score, sorted(set(rule_reasons))[:4], warnings


def unique_cards_from_ids(ids: Sequence[int], card_index: Dict[int, Card]) -> Tuple[List[Card], List[int]]:
    """Resolve ids to unique cards and return missing ids."""
    cards = []
    missing = []
    seen = set()
    for card_id in ids:
        if card_id in seen:
            continue
        seen.add(card_id)
        card = card_index.get(card_id)
        if card:
            cards.append(card)
        else:
            missing.append(card_id)
    return cards, missing


def rank_owned_pool(owned_cards: Sequence[Card], npc_cards: Sequence[Card], rules: Sequence[str],
                    star_map: Dict[int, int]) -> List[Card]:
    """Keep the most relevant owned cards when the collection is large."""
    ranked = sorted(
        owned_cards,
        key=lambda card: base_card_score(card, star_map) + rule_card_score(card, rules)[0] + matchup_card_score(card, npc_cards, rules),
        reverse=True,
    )
    return ranked[:MAX_PLAYER_POOL]


def generate_player_candidates(owned_cards: Sequence[Card], npc_cards: Sequence[Card], rules: Sequence[str],
                               star_map: Dict[int, int]) -> List[Tuple[float, List[str], List[Card]]]:
    """Generate legal player decks and keep the strongest preliminary options."""
    pool = rank_owned_pool(owned_cards, npc_cards, rules, star_map)
    candidates = []
    for combo in itertools.combinations(pool, 5):
        deck = list(combo)
        if not player_deck_is_legal(deck, star_map):
            continue
        score, reasons, _ = deck_heuristic_score(deck, npc_cards, rules, star_map)
        candidates.append((score, reasons, deck))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:MAX_PRELIMINARY_DECKS]


def npc_deck_strength(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> float:
    """Rank NPC possible five-card hands without applying player deck limits."""
    rule_total = sum(rule_card_score(card, rules)[0] for card in deck) / 5
    base_total = sum(base_card_score(card, star_map) for card in deck) / 5
    coverage = sum(1 for side in range(4) if max(side_values(card)[side] for card in deck) >= 8) * 3.0
    return base_total + rule_total + coverage


def generate_npc_decks(npc_cards: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> List[List[Card]]:
    """Generate likely NPC five-card hands, intentionally ignoring player limits."""
    if len(npc_cards) < 5:
        return []
    source = sorted(npc_cards, key=lambda card: base_card_score(card, star_map) + rule_card_score(card, rules)[0], reverse=True)
    source = source[:MAX_NPC_SOURCE_POOL]
    decks = [list(combo) for combo in itertools.combinations(source, 5)]
    decks.sort(key=lambda deck: npc_deck_strength(deck, rules, star_map), reverse=True)
    return decks[:MAX_NPC_DECKS]


def copy_deck(deck: Sequence[Card], owner: str) -> List[Card]:
    """Copy a deck and set all cards to one owner."""
    return [clone_card(card, owner) for card in deck]


def positional_score(row: int, col: int) -> float:
    """Return a small positional value for board scoring."""
    if (row, col) in ((0, 0), (0, 2), (2, 0), (2, 2)):
        return 1.4
    return 1.1 if (row, col) == (1, 1) else 0.9


def state_score(state: GameState, my_idx: int, star_map: Dict[int, int]) -> float:
    """Evaluate a simulated board from the player's perspective."""
    red_count, blue_count = state.count_cards()
    score = blue_count - red_count if my_idx == 1 else red_count - blue_count
    for row in range(3):
        for col in range(3):
            card = state.board.get_card(row, col)
            if not card:
                continue
            sign = 1 if (my_idx == 1 and card.owner == "blue") or (my_idx == 0 and card.owner == "red") else -1
            score += sign * positional_score(row, col) * 0.30
            score += sign * max(side_values(card)) * 0.035
            score += sign * card_star(card, star_map) * 0.06
    return score


def choose_greedy_move(state: GameState, my_idx: int, star_map: Dict[int, int]) -> Optional[Tuple[Card, Tuple[int, int]]]:
    """Choose a one-ply move for deterministic demo simulation."""
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
            score = state_score(state, my_idx, star_map)
        finally:
            state.undo_move(record)
        if (maximizing and score > best_score) or (not maximizing and score < best_score):
            best_score = score
            best_move = move
    return best_move


def simulate_match(player_deck: Sequence[Card], npc_deck: Sequence[Card], rules: Sequence[str],
                   star_map: Dict[int, int], player_first: bool) -> float:
    """Run one lightweight tactical game and return a normalized score."""
    state = GameState(
        Board(),
        [Player("npc", copy_deck(npc_deck, "red")), Player("player", copy_deck(player_deck, "blue"))],
        current_player_idx=1 if player_first else 0,
        rules=list(rules),
    )
    if "同类强化" in rules or "同类弱化" in rules:
        state.recalculate_type_modifiers()
    guard = 0
    while not state.is_game_over() and guard < 9:
        guard += 1
        move = choose_greedy_move(state, 1, star_map)
        if move is None:
            break
        card, (row, col) = move
        state.make_move(row, col, card)
    score = state_score(state, 1, star_map)
    return 100.0 / (1.0 + math.exp(-score / 3.0))


def simulation_components(player_deck: Sequence[Card], npc_decks: Sequence[Sequence[Card]],
                          rules: Sequence[str], star_map: Dict[int, int]) -> Dict[str, float]:
    """Evaluate a player deck against likely NPC hands."""
    scores = []
    for npc_deck in npc_decks:
        scores.append(simulate_match(player_deck, npc_deck, rules, star_map, player_first=True))
        scores.append(simulate_match(player_deck, npc_deck, rules, star_map, player_first=False))
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
    """Recommend legal player decks against a known NPC card pool.

    Args:
        owned_ids: Player-owned card ids.
        npc_ids: Known NPC card pool ids. If more than five are supplied, likely
            five-card NPC hands are sampled from this pool.
        rules: Special rules used by the matchup.
        top_n: Number of recommendations to return.
        card_index: Optional test hook for id-to-card lookup.
        star_map: Optional test hook for card rarity lookup.

    Returns:
        A JSON-serializable recommendation report.
    """
    rules = parse_rules(rules)
    card_index = card_index or load_card_index()
    star_map = star_map or get_card_star_map()
    owned_cards, missing_owned = unique_cards_from_ids(owned_ids, card_index)
    npc_cards, missing_npc = unique_cards_from_ids(npc_ids, card_index)
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
        return {"process": process + ["玩家可用牌不足5张，无法推荐"], "results": [], "npc_decks": []}
    if len(npc_cards) < 5:
        return {"process": process + ["NPC已知牌不足5张，无法模拟"], "results": [], "npc_decks": []}

    npc_decks = generate_npc_decks(npc_cards, rules, star_map)
    process.append(f"NPC候选上场组合：{len(npc_decks)}组")
    candidates = generate_player_candidates(owned_cards, npc_cards, rules, star_map)
    process.append(f"玩家合法候选卡组：{len(candidates)}组进入模拟")
    if not candidates:
        return {"process": process + ["没有找到符合玩家选卡规则的5张组合"], "results": [], "npc_decks": []}

    results: List[Recommendation] = []
    for heuristic, heuristic_reasons, deck in candidates:
        sim = simulation_components(deck, npc_decks, rules, star_map)
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
        warnings = deck_limit_warnings(deck, star_map)
        results.append(Recommendation(
            deck=[card_to_dict(card, star_map) for card in deck],
            score=round(final, 2),
            components=components,
            reasons=reasons[:6],
            warnings=warnings,
        ))
    results.sort(key=lambda item: item.score, reverse=True)
    best = results[0]
    process.append(f"最高分：{best.score}")
    process.extend([f"推荐理由：{reason}" for reason in best.reasons])
    return {
        "process": process,
        "npc_decks": [[card_to_dict(card, star_map) for card in deck] for deck in npc_decks[:3]],
        "best": best.__dict__,
        "results": [item.__dict__ for item in results[:max(1, int(top_n))]],
    }


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Recommend decks from JSON payload."""
    payload = request.get_json() or {}
    result = recommend_decks(
        parse_id_list(payload.get("ownedIds", [])),
        parse_id_list(payload.get("npcIds", [])),
        payload.get("rules", []),
        int(payload.get("topN", 5)),
    )
    return jsonify(result)


def self_test() -> Dict:
    """Run a small deterministic smoke test with the bundled database."""
    card_index = load_card_index()
    owned_ids = list(card_index.keys())[:35]
    npc_ids = list(card_index.keys())[40:50]
    return recommend_decks(owned_ids, npc_ids, ["加算"], top_n=3, card_index=card_index)


def main() -> None:
    """Run the CLI demo or start the JSON API service."""
    parser = argparse.ArgumentParser(description="NPC matchup deck recommender demo")
    parser.add_argument("--owned", default="", help="comma separated player-owned card ids")
    parser.add_argument("--npc", default="", help="comma separated known NPC card pool ids")
    parser.add_argument("--rules", default="", help="comma separated special rules")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5065)
    args = parser.parse_args()
    if args.serve:
        print(f"NPC deck recommender running on http://{args.host}:{args.port}/api/recommend")
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
        return
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if not args.owned or not args.npc:
        parser.error("--owned and --npc are required unless --self-test or --serve is used")
    result = recommend_decks(parse_id_list(args.owned), parse_id_list(args.npc), args.rules, args.top)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
