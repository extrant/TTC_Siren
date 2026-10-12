# -*- coding: utf-8 -*-
"""
Standalone Triple Triad draft analyzer.

This file intentionally does not modify the existing solver. It reuses the
project card database helpers and provides a small web UI for pre-match draft
recommendations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from flask import Flask, jsonify, request

from ai_server import get_all_cards, get_card_star_map, get_card_type_map
from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player


EXTRA_RULES = ["同数", "加算", "逆转", "王牌杀手", "同类强化", "同类弱化", "秩序", "混乱"]
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
DRAFT_REFINE_TOP_K = 8
DRAFT_STAGE_ONE_RATIO = 3.0 / 7.0
DRAFT_STAGE_TWO_RATIO = 4.0 / 7.0
MAX_DRAFT_VERIFY_WORKERS = 6
MIN_DRAFT_VERIFY_SECONDS = 0.2


@dataclass
class DeckScore:
    deck: List[Dict]
    final_score: float
    components: Dict[str, float]
    reasons: List[str]
    warnings: List[str]
    selection: List[str] = field(default_factory=list)


app = Flask(__name__)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a numeric score into a display-friendly range."""
    return max(low, min(high, value))


def real_card_id(card: Card) -> int:
    """Return the base card id for regular or predicted cards."""
    card_id = int(card.card_id or 0)
    if getattr(card, "_draft_synthetic", False):
        return card_id
    if 1000 <= card_id < 100000:
        return card_id % 1000
    return card_id


def card_star(card: Card, star_map: Dict[int, int]) -> int:
    """Read a card star value with a conservative fallback."""
    draft_star = getattr(card, "_draft_star", None)
    if draft_star is not None:
        return int(draft_star)
    return int(star_map.get(real_card_id(card), 1))


def is_valid_database_card(card: Card, star_map: Dict[int, int]) -> bool:
    """Filter out placeholder or malformed database rows."""
    values = [card.base_up, card.base_right, card.base_down, card.base_left]
    return bool(real_card_id(card)) and all(1 <= value <= 10 for value in values) and card_star(card, star_map) >= 1


def valid_database_cards(star_map: Dict[int, int]) -> List[Card]:
    """Return only playable cards from the existing card database."""
    return [card for card in get_all_cards() if is_valid_database_card(card, star_map)]


def card_to_dict(card: Card, star_map: Dict[int, int]) -> Dict:
    """Serialize a Card for the draft UI."""
    result = {
        "id": real_card_id(card),
        "up": card.base_up,
        "right": card.base_right,
        "down": card.base_down,
        "left": card.base_left,
        "star": card_star(card, star_map),
        "type": card.card_type or "无类型",
    }
    label = getattr(card, "_draft_label", None)
    if label:
        result["label"] = label
    return result


def parse_card_value(value) -> int:
    """Parse a UI card side value, accepting A as 10."""
    if isinstance(value, str) and value.strip().upper() == "A":
        return 10
    return int(value)


def payload_value(payload: Dict, *keys):
    """Read the first present non-empty value from a payload."""
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def card_sides_from_payload(payload: Dict) -> Tuple[int, int, int, int]:
    """Read card sides as up/right/down/left from UI payload."""
    sides = payload_value(payload, "sides", "values")
    if sides is not None:
        return tuple(parse_card_value(value) for value in sides[:4])

    original = payload_value(payload, "original", "raw")
    if original is not None:
        if isinstance(original, str):
            original = [item.strip() for item in original.replace("，", ",").split(",")]
        raw = [parse_card_value(value) for value in original[:4]]
        return raw[3], raw[0], raw[2], raw[1]

    return (
        parse_card_value(payload_value(payload, "up", "u", "numU")),
        parse_card_value(payload_value(payload, "right", "r", "numR")),
        parse_card_value(payload_value(payload, "down", "d", "numD")),
        parse_card_value(payload_value(payload, "left", "l", "numL")),
    )


def parse_card_id(payload: Dict, fallback_id: int) -> Tuple[int, bool]:
    """Parse a numeric card id, falling back to a synthetic draft id."""
    raw_id = payload_value(payload, "id", "cardId", "card_id")
    try:
        return int(raw_id), False
    except (TypeError, ValueError):
        return fallback_id, True


def dict_to_card(payload: Dict, fallback_id: int = 900000) -> Card:
    """Build a Card from UI payload."""
    up, right, down, left = card_sides_from_payload(payload)
    card_id, synthetic = parse_card_id(payload, fallback_id)
    return Card(
        up,
        right,
        down,
        left,
        card_id=card_id,
        card_type=payload.get("type") if payload.get("type") != "无类型" else None,
    )


def mark_draft_card(card: Card, payload: Dict, fallback_label: str, synthetic: bool = False) -> Card:
    """Attach UI-only draft metadata to a card."""
    raw_id = payload_value(payload, "id", "cardId", "card_id")
    card._draft_label = str(payload.get("label") or payload.get("node") or raw_id or fallback_label)
    if synthetic:
        card._draft_synthetic = True
    if payload.get("star") not in (None, ""):
        card._draft_star = int(payload["star"])
    return card


def draft_payload_to_card(payload: Dict, fallback_id: int, fallback_label: str) -> Card:
    """Build a Card and keep draft-specific metadata."""
    card_id, synthetic = parse_card_id(payload, fallback_id)
    card = dict_to_card(payload, fallback_id)
    card.card_id = card_id
    return mark_draft_card(card, payload, fallback_label, synthetic)


def deck_is_legal(deck: Sequence[Card], star_map: Dict[int, int]) -> Tuple[bool, List[str]]:
    """Validate official deck star limits."""
    high_count = sum(1 for card in deck if card_star(card, star_map) >= 4)
    five_count = sum(1 for card in deck if card_star(card, star_map) == 5)
    warnings = []
    if high_count > 2:
        warnings.append(f"★4以上卡牌有{high_count}张，超过2张限制")
    if five_count > 1:
        warnings.append(f"★5卡牌有{five_count}张，超过1张限制")
    return not warnings, warnings


def single_card_base_score(card: Card, star_map: Dict[int, int]) -> float:
    """Score raw card strength before rule adaptation."""
    values = [card.base_up, card.base_right, card.base_down, card.base_left]
    avg = sum(values) / 4
    high_edges = sum(1 for value in values if value >= 8)
    low_edges = sum(1 for value in values if value <= 3)
    adjacent_high = (
        (card.base_up >= 8 and card.base_right >= 8)
        or (card.base_right >= 8 and card.base_down >= 8)
        or (card.base_down >= 8 and card.base_left >= 8)
        or (card.base_left >= 8 and card.base_up >= 8)
    )
    score = avg * 6.0 + high_edges * 8.0 - low_edges * 4.0
    score += card_star(card, star_map) * 2.5
    if adjacent_high:
        score += 6.0
    if high_edges >= 3:
        score += 8.0
    return clamp(score)


def rule_score_for_card(card: Card, rules: Sequence[str]) -> Tuple[float, List[str]]:
    """Score one card against selected rules and return compact reasons."""
    values = [card.base_up, card.base_right, card.base_down, card.base_left]
    avg = sum(values) / 4
    score = 45.0
    reasons = []

    if "同数" in rules:
        duplicate_count = 4 - len(set(values))
        score += duplicate_count * 10.0
        if duplicate_count:
            reasons.append("重复边值适合同数")
    if "加算" in rules:
        pair_sums = [a + b for a, b in itertools.combinations(values, 2)]
        common_hits = sum(1 for value in pair_sums if value in (8, 9, 10, 11, 12, 13))
        score += min(common_hits * 3.5, 18.0)
        if common_hits >= 3:
            reasons.append("有多组常见加算和")
    if "逆转" in rules:
        if avg <= 5:
            score += 24.0
            reasons.append("低均值适合逆转")
        elif avg >= 8:
            score -= 20.0
    if "王牌杀手" in rules:
        special_edges = sum(1 for value in values if value in (1, 10))
        score += special_edges * 9.0
        if special_edges:
            reasons.append("含1或A可利用王牌杀手")
    if "同类强化" in rules and card.card_type:
        score += 7.0
    if "同类弱化" in rules and card.card_type:
        score += 3.0

    return clamp(score), reasons


def deck_base_score(deck: Sequence[Card], star_map: Dict[int, int]) -> float:
    """Score the base strength of a deck."""
    return sum(single_card_base_score(card, star_map) for card in deck) / max(len(deck), 1)


def deck_rule_score(deck: Sequence[Card], rules: Sequence[str]) -> Tuple[float, List[str]]:
    """Score rule fit for a whole deck."""
    scores = []
    reasons = []
    for card in deck:
        score, card_reasons = rule_score_for_card(card, rules)
        scores.append(score)
        reasons.extend(card_reasons)

    values_by_side = [
        [card.base_up for card in deck],
        [card.base_right for card in deck],
        [card.base_down for card in deck],
        [card.base_left for card in deck],
    ]
    coverage_bonus = sum(1 for side_values in values_by_side if max(side_values) >= 8) * 3.0
    result = (sum(scores) / max(len(scores), 1)) + coverage_bonus
    return clamp(result), sorted(set(reasons))[:4]


def deck_synergy_score(deck: Sequence[Card], rules: Sequence[str]) -> Tuple[float, List[str]]:
    """Score how well five cards cooperate."""
    reasons = []
    score = 50.0
    all_values = [value for card in deck for value in [card.base_up, card.base_right, card.base_down, card.base_left]]
    high_by_side = {
        "上": max(card.base_up for card in deck),
        "右": max(card.base_right for card in deck),
        "下": max(card.base_down for card in deck),
        "左": max(card.base_left for card in deck),
    }
    covered_sides = sum(1 for value in high_by_side.values() if value >= 8)
    score += covered_sides * 6.0
    if covered_sides == 4:
        reasons.append("四向都有高边覆盖")

    weak_edges = sum(1 for value in all_values if value <= 3)
    score -= max(0, weak_edges - 5) * 2.5

    if "同数" in rules:
        duplicates = sum(count - 1 for count in {v: all_values.count(v) for v in set(all_values)}.values() if count >= 3)
        score += min(duplicates * 3.0, 18.0)
        if duplicates >= 3:
            reasons.append("牌组有同数连携素材")
    if "加算" in rules:
        popular_sums = 0
        for card_a, card_b in itertools.combinations(deck, 2):
            for value_a in [card_a.base_up, card_a.base_right, card_a.base_down, card_a.base_left]:
                for value_b in [card_b.base_up, card_b.base_right, card_b.base_down, card_b.base_left]:
                    if value_a + value_b in (8, 10, 12):
                        popular_sums += 1
        score += min(popular_sums * 0.45, 18.0)
        if popular_sums >= 18:
            reasons.append("牌组加算组合密度高")

    type_counts: Dict[str, int] = {}
    for card in deck:
        if card.card_type:
            type_counts[card.card_type] = type_counts.get(card.card_type, 0) + 1
    if "同类强化" in rules and type_counts:
        best_type, best_count = max(type_counts.items(), key=lambda item: item[1])
        score += best_count * 6.0
        if best_count >= 2:
            reasons.append(f"{best_type}类型可形成同类强化")
    if "同类弱化" in rules and type_counts:
        repeats = sum(count - 1 for count in type_counts.values() if count > 1)
        score -= repeats * 8.0
        if repeats == 0:
            reasons.append("类型分散，适合同类弱化")

    return clamp(score), reasons[:4]


def deck_stability_score(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> Tuple[float, List[str]]:
    """Estimate opening and draw-order stability."""
    per_card = [single_card_base_score(card, star_map) for card in deck]
    sorted_scores = sorted(per_card, reverse=True)
    top_three_avg = sum(sorted_scores[:3]) / max(min(3, len(sorted_scores)), 1)
    bottom_two_avg = sum(sorted_scores[-2:]) / max(min(2, len(sorted_scores)), 1)
    score = top_three_avg * 0.65 + bottom_two_avg * 0.35
    reasons = []

    stars = [card_star(card, star_map) for card in deck]
    if max(stars) - min(stars) <= 2:
        score += 5.0
        reasons.append("星级曲线平稳")
    if "秩序" in rules or "混乱" in rules:
        score += bottom_two_avg * 0.08
        reasons.append("可用顺序受限时下限更重要")
    if "选拔" in rules:
        score += 4.0

    return clamp(score), reasons


def deck_risk_penalty(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int]) -> Tuple[float, List[str]]:
    """Calculate risk penalty; higher means riskier."""
    warnings = []
    penalty = 0.0
    legal, legal_warnings = deck_is_legal(deck, star_map)
    if not legal:
        penalty += 45.0
        warnings.extend(legal_warnings)

    all_values = [value for card in deck for value in [card.base_up, card.base_right, card.base_down, card.base_left]]
    weak_edges = sum(1 for value in all_values if value <= 3)
    if weak_edges >= 8 and "逆转" not in rules:
        penalty += (weak_edges - 7) * 3.0
        warnings.append("弱边偏多，非逆转规则下容易被反吃")

    if "逆转" in rules:
        high_edges = sum(1 for value in all_values if value >= 8)
        if high_edges >= 8:
            penalty += (high_edges - 7) * 4.0
            warnings.append("高边偏多，逆转规则下风险较高")

    return clamp(penalty), warnings


def copy_deck_for_owner(deck: Sequence[Card], owner: str) -> List[Card]:
    """Copy cards and assign them to one player."""
    copied = []
    for card in deck:
        item = card.copy()
        item.owner = owner
        for attr in ("_draft_label", "_draft_synthetic", "_draft_star"):
            if hasattr(card, attr):
                setattr(item, attr, getattr(card, attr))
        copied.append(item)
    return copied


def build_random_opponent_deck(all_cards: Sequence[Card], star_map: Dict[int, int], avoid_ids: set) -> List[Card]:
    """Build a random legal opponent deck for draft verification."""
    available = [card for card in all_cards if real_card_id(card) not in avoid_ids]
    deck = generate_legal_deck(available or all_cards, star_map)
    return copy_deck_for_owner(deck, "red")


def opponent_card_threat_score(card: Card, my_deck: Sequence[Card], rules: Sequence[str],
                               star_map: Dict[int, int]) -> float:
    """Estimate how threatening one opponent card is against a selected deck."""
    score, _ = rule_score_for_card(card, rules)
    score += single_card_base_score(card, star_map) * 0.35
    directions = [
        ("up", "down"),
        ("right", "left"),
        ("down", "up"),
        ("left", "right"),
    ]
    for opp_dir, my_dir in directions:
        for my_card in my_deck:
            result = card.compare_values(opp_dir, my_card, my_dir, list(rules))
            if result == 1:
                score += 4.0
            elif result == 0:
                score += 0.7
            opp_value = card.get_effective_value(opp_dir, list(rules))
            my_value = my_card.get_effective_value(my_dir, list(rules))
            if "同数" in rules and opp_value == my_value:
                score += 2.4
            if "加算" in rules and opp_value + my_value in (8, 10, 12):
                score += 1.8
    return score


def build_threat_opponent_deck(all_cards: Sequence[Card], star_map: Dict[int, int],
                               avoid_ids: set, my_deck: Sequence[Card],
                               rules: Sequence[str]) -> List[Card]:
    """Build a legal opponent deck biased toward cards that threaten my deck."""
    available = [card for card in all_cards if real_card_id(card) not in avoid_ids]
    ranked = sorted(
        available or list(all_cards),
        key=lambda card: opponent_card_threat_score(card, my_deck, rules, star_map),
        reverse=True,
    )
    return build_threat_opponent_deck_from_ranked(ranked, star_map)


def build_threat_opponent_deck_from_ranked(ranked: Sequence[Card], star_map: Dict[int, int]) -> List[Card]:
    """Build a threat-biased opponent deck from a cached ranking."""
    top_pool = ranked[: min(36, len(ranked))]
    deck = []
    attempts = 0
    while len(deck) < 5 and attempts < 80:
        attempts += 1
        pick_pool = top_pool[: max(8, min(len(top_pool), 16 + attempts // 8))]
        candidate = random.choice(pick_pool)
        if any(real_card_id(card) == real_card_id(candidate) for card in deck):
            continue
        if deck_is_legal(deck + [candidate], star_map)[0]:
            deck.append(candidate)

    if len(deck) < 5:
        for candidate in ranked:
            if any(real_card_id(card) == real_card_id(candidate) for card in deck):
                continue
            if deck_is_legal(deck + [candidate], star_map)[0]:
                deck.append(candidate)
            if len(deck) == 5:
                break
    return copy_deck_for_owner(deck[:5], "red")


def rank_threat_opponent_cards(all_cards: Sequence[Card], avoid_ids: set, my_deck: Sequence[Card],
                               rules: Sequence[str], star_map: Dict[int, int]) -> List[Card]:
    """Rank opponent cards once for repeated threat sampling against one deck."""
    available = [card for card in all_cards if real_card_id(card) not in avoid_ids]
    return sorted(
        available or list(all_cards),
        key=lambda card: opponent_card_threat_score(card, my_deck, rules, star_map),
        reverse=True,
    )


def positional_score(row: int, col: int) -> float:
    """Return a small board-position value."""
    if (row, col) in ((0, 0), (0, 2), (2, 0), (2, 2)):
        return 1.5
    if (row, col) == (1, 1):
        return 1.15
    return 0.9


def state_score(state: GameState, ai_player_idx: int, star_map: Dict[int, int]) -> float:
    """Evaluate a simulated board from the selected deck's side."""
    red_count, blue_count = state.count_cards()
    score = (blue_count - red_count) if ai_player_idx == 1 else (red_count - blue_count)
    for row in range(3):
        for col in range(3):
            card = state.board.get_card(row, col)
            if not card:
                continue
            owner_sign = 1 if (
                (ai_player_idx == 1 and card.owner == "blue")
                or (ai_player_idx == 0 and card.owner == "red")
            ) else -1
            values = [card.base_up, card.base_right, card.base_down, card.base_left]
            score += owner_sign * positional_score(row, col) * 0.35
            score += owner_sign * max(values) * 0.035
            score += owner_sign * card_star(card, star_map) * 0.08
    return score


def choose_greedy_move(state: GameState, ai_player_idx: int, star_map: Dict[int, int]) -> Tuple:
    """Pick a one-ply tactical move for simulation."""
    best_move = None
    best_score = float("-inf") if state.current_player_idx == ai_player_idx else float("inf")
    for move in state.get_available_moves():
        card, (row, col) = move
        record = state.make_move(row, col, card)
        if record is None:
            continue
        try:
            score = state_score(state, ai_player_idx, star_map)
        finally:
            state.undo_move(record)
        score += random.uniform(-0.08, 0.08)
        if state.current_player_idx == ai_player_idx:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move
    return best_move


def simulate_match(deck: Sequence[Card], opponent_deck: Sequence[Card], rules: Sequence[str],
                   star_map: Dict[int, int], my_first: bool) -> float:
    """Play one lightweight tactical simulation and return normalized result."""
    my_hand = copy_deck_for_owner(deck, "blue")
    opp_hand = copy_deck_for_owner(opponent_deck, "red")
    state = GameState(
        Board(),
        [Player("opp", opp_hand), Player("me", my_hand)],
        current_player_idx=1 if my_first else 0,
        rules=list(rules),
    )
    if "同类强化" in rules or "同类弱化" in rules:
        state.recalculate_type_modifiers()

    move_guard = 0
    while not state.is_game_over() and move_guard < 9:
        move_guard += 1
        move = choose_greedy_move(state, 1, star_map)
        if not move:
            break
        card, (row, col) = move
        state.make_move(row, col, card)

    score = state_score(state, 1, star_map)
    return 100.0 / (1.0 + math.exp(-score / 3.0))


def collect_verification_samples(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int],
                                 all_cards: Sequence[Card], deadline: float,
                                 min_rounds: int = 2) -> Dict:
    """Collect lightweight matchup samples until the deck budget is exhausted."""
    scores = []
    avoid_ids = {real_card_id(card) for card in deck}
    threat_ranked = rank_threat_opponent_cards(all_cards, avoid_ids, deck, rules, star_map)
    rounds = 0
    threat_rounds = 0
    while (time.perf_counter() < deadline or rounds < min_rounds) and rounds < 2000:
        use_threat = rounds % 3 == 2
        if use_threat:
            opponent = build_threat_opponent_deck_from_ranked(threat_ranked, star_map)
            threat_rounds += 1
        else:
            opponent = build_random_opponent_deck(all_cards, star_map, avoid_ids)
        scores.append(simulate_match(deck, opponent, rules, star_map, my_first=True))
        scores.append(simulate_match(deck, opponent, rules, star_map, my_first=False))
        rounds += 1
        if time.perf_counter() >= deadline and rounds >= min_rounds:
            break

    return {
        "scores": scores,
        "rounds": rounds,
        "threat_rounds": threat_rounds,
    }


def summarize_verification_samples(samples: Dict) -> Tuple[float, Dict]:
    """Summarize one or more verification sample batches into a score tuple."""
    scores = samples.get("scores", [])
    rounds = int(samples.get("rounds", 0))
    threat_rounds = int(samples.get("threat_rounds", 0))
    if not scores:
        return 50.0, {"rounds": 0, "average": 50.0, "floor": 50.0, "min": 50.0}
    average = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    floor_index = min(len(sorted_scores) - 1, max(0, int(len(sorted_scores) * 0.10)))
    floor = sorted_scores[floor_index]
    minimum = sorted_scores[0]
    verified = average * 0.65 + floor * 0.35
    return clamp(verified), {
        "rounds": rounds,
        "threat_rounds": threat_rounds,
        "simulations": len(scores),
        "average": round(average, 2),
        "floor": round(floor, 2),
        "min": round(minimum, 2),
    }


def merge_verification_samples(*sample_batches: Dict) -> Dict:
    """Merge staged verification samples before scoring."""
    merged = {"scores": [], "rounds": 0, "threat_rounds": 0}
    for batch in sample_batches:
        merged["scores"].extend(batch.get("scores", []))
        merged["rounds"] += int(batch.get("rounds", 0))
        merged["threat_rounds"] += int(batch.get("threat_rounds", 0))
    return merged


def verification_score(deck: Sequence[Card], rules: Sequence[str], star_map: Dict[int, int],
                       all_cards: Sequence[Card], deadline: float, min_rounds: int = 2) -> Tuple[float, Dict]:
    """Run repeated lightweight matchups until the deck budget is exhausted."""
    return summarize_verification_samples(
        collect_verification_samples(deck, rules, star_map, all_cards, deadline, min_rounds)
    )


def score_deck(deck: Sequence[Card], rules: Sequence[str], weights: Dict[str, float],
               star_map: Dict[int, int], verification: Tuple[float, Dict] = None,
               selection: List[str] = None) -> DeckScore:
    """Score one draft deck and produce explanation details."""
    base = deck_base_score(deck, star_map)
    rule, rule_reasons = deck_rule_score(deck, rules)
    synergy, synergy_reasons = deck_synergy_score(deck, rules)
    stability, stability_reasons = deck_stability_score(deck, rules, star_map)
    risk, warnings = deck_risk_penalty(deck, rules, star_map)
    verification_value, verification_meta = verification or (
        50.0,
        {"rounds": 0, "simulations": 0, "average": 50.0, "floor": 50.0, "min": 50.0},
    )
    weight_sum = max(
        weights["base"]
        + weights["rule"]
        + weights["synergy"]
        + weights["stability"]
        + weights["verification"],
        0.001,
    )
    final = (
        base * weights["base"]
        + rule * weights["rule"]
        + synergy * weights["synergy"]
        + stability * weights["stability"]
        + verification_value * weights["verification"]
    ) / weight_sum
    final -= risk * weights["risk"]
    reasons = (rule_reasons + synergy_reasons + stability_reasons) or ["整体表现均衡"]
    if verification_meta.get("simulations", 0):
        reasons.append(
            f"深算{verification_meta['simulations']}局，威胁采样{verification_meta.get('threat_rounds', 0)}轮，"
            f"均值{verification_meta['average']}，P10下限{verification_meta['floor']}"
        )
    return DeckScore(
        deck=[card_to_dict(card, star_map) for card in deck],
        final_score=round(clamp(final), 2),
        components={
            "base": round(base, 2),
            "rule": round(rule, 2),
            "synergy": round(synergy, 2),
            "stability": round(stability, 2),
            "verification": round(verification_value, 2),
            "risk": round(risk, 2),
        },
        reasons=reasons[:6],
        warnings=warnings,
        selection=selection or [],
    )


def verify_deck_worker(payload: Tuple[List[Card], List[str], Dict[int, int], List[Card], float]) -> Tuple[float, Dict]:
    """Process-pool worker for one deck verification."""
    deck, rules, star_map, all_cards, seconds = payload
    deadline = time.perf_counter() + seconds
    return verification_score(deck, rules, star_map, all_cards, deadline)


def collect_samples_worker(payload: Tuple[List[Card], List[str], Dict[int, int], List[Card], float]) -> Dict:
    """Process-pool worker that returns mergeable verification samples."""
    deck, rules, star_map, all_cards, seconds = payload
    deadline = time.perf_counter() + seconds
    return collect_verification_samples(deck, rules, star_map, all_cards, deadline)


def collect_samples_parallel(candidates: List[Tuple[DeckScore, List[Card], List[str]]],
                             rules: Sequence[str], star_map: Dict[int, int],
                             all_cards: Sequence[Card], wall_budget: float) -> Tuple[List[Dict], int, float]:
    """Collect verification samples for candidates with bounded process count.

    Args:
        candidates: Decks to verify in this stage.
        rules: Active draft rules.
        star_map: Card star lookup.
        all_cards: Database cards used to sample opponents.
        wall_budget: Target wall-clock budget for the whole stage.
    Returns:
        Sample batches, worker count, and seconds assigned to each deck.
    """
    total = len(candidates)
    if total == 0:
        return [], 0, 0.0

    worker_count = min(total, os.cpu_count() or total, MAX_DRAFT_VERIFY_WORKERS)
    batch_count = max(1, math.ceil(total / worker_count))
    seconds_per_deck = max(MIN_DRAFT_VERIFY_SECONDS, float(wall_budget) / batch_count)
    tasks = [
        (deck, list(rules), star_map, all_cards, seconds_per_deck)
        for _, deck, _ in candidates
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        samples = list(executor.map(collect_samples_worker, tasks))
    return samples, worker_count, seconds_per_deck


def normalize_weights(payload: Dict) -> Dict[str, float]:
    """Read UI weight payload with defaults."""
    weights = dict(DEFAULT_WEIGHTS)
    for key in weights:
        if key in payload:
            weights[key] = max(0.0, float(payload[key]))
    return weights


def generate_legal_deck(cards: Sequence[Card], star_map: Dict[int, int]) -> List[Card]:
    """Generate one random deck that satisfies star limits."""
    shuffled = list(cards)
    random.shuffle(shuffled)
    deck = []
    for card in shuffled:
        candidate = deck + [card]
        if len(candidate) <= 5 and deck_is_legal(candidate, star_map)[0]:
            deck.append(card)
        if len(deck) == 5:
            return deck
    return list(shuffled[:5])


def build_selection_payload(cards: Sequence[Card], star_map: Dict[int, int]) -> List:
    """Build the UI draft payload from 15 ordered cards."""
    payloads = [card_to_dict(card, star_map) for card in cards]
    return [
        [
            {"left": payloads[0], "right": payloads[1]},
            {"left": payloads[2], "right": payloads[3]},
            {"left": payloads[4], "right": payloads[5]},
        ],
        [
            {"left": payloads[6], "right": payloads[7]},
            {"left": payloads[8], "right": payloads[9]},
            {"left": payloads[10], "right": payloads[11]},
        ],
        [payloads[12], payloads[13], payloads[14]],
    ]


def selection_payload_is_legal(selection_payload: List, star_map: Dict[int, int]) -> bool:
    """Return whether every possible selection result is a legal five-card deck."""
    expanded = expand_selection_candidates(selection_payload)
    return len(expanded) == 27 and all(deck_is_legal(deck, star_map)[0] for deck, _labels in expanded)


def generate_random_selection_candidates(star_map: Dict[int, int], attempts: int = 200) -> List:
    """Generate random three-stage draft choices that always expand to legal decks.

    Args:
        star_map: Card star lookup.
        attempts: Maximum full-pool sampling attempts before falling back to low-star cards.
    Returns:
        A DraftStage payload: stage1 and stage2 have three two-card choices, stage3 has three single cards.
    """
    cards = valid_database_cards(star_map)
    if len(cards) < 15:
        raise ValueError("可用卡牌不足15张，无法生成选拔候选")

    for _attempt in range(attempts):
        sampled = random.sample(cards, 15)
        payload = build_selection_payload(sampled, star_map)
        if selection_payload_is_legal(payload, star_map):
            return payload

    low_star_cards = [card for card in cards if card_star(card, star_map) <= 3]
    if len(low_star_cards) < 15:
        raise ValueError("低星可用卡牌不足15张，无法生成必定合法的选拔候选")
    sampled = random.sample(low_star_cards, 15)
    return build_selection_payload(sampled, star_map)


def normalize_selection_choice(choice, fallback_id: int, fallback_label: str) -> List[Card]:
    """Normalize one UI selection choice into one or more cards."""
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
        cards.append(draft_payload_to_card(card_payload, fallback_id + card_index - 1, label))
    return cards


def expand_selection_candidates(raw_candidates) -> List[Tuple[List[Card], List[str]]]:
    """Expand three UI draft stages into complete five-card decks."""
    stages = raw_candidates
    if isinstance(raw_candidates, dict):
        stages = raw_candidates.get("stages") or raw_candidates.get("groups") or raw_candidates.get("choices") or []

    normalized_stages: List[List[Tuple[List[Card], str]]] = []
    for stage_index, stage in enumerate(stages, start=1):
        if isinstance(stage, dict):
            choices = stage.get("choices") or stage.get("candidates") or stage.get("groups") or []
        else:
            choices = stage

        normalized_choices = []
        for choice_index, choice in enumerate(choices, start=1):
            label = f"备选{stage_index}-候选{choice_index}"
            cards = normalize_selection_choice(choice, 900000 + stage_index * 100 + choice_index * 10, label)
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


def analyze_candidates(mode: str, raw_candidates: List, rules: List[str], weights: Dict[str, float],
                       time_budget: float = DEFAULT_TIME_BUDGET) -> Dict:
    """Analyze the three-stage draft selection choices."""
    star_map = get_card_star_map()
    all_cards = valid_database_cards(star_map)
    time_budget = max(5.0, min(20.0, float(time_budget)))
    started = time.perf_counter()
    candidates = raw_candidates or []
    process = [
        f"规则：{', '.join(rules)}",
        f"权重：{weights}",
        f"深算预算：{time_budget:.1f}秒",
        "对手采样：约2/3随机合法选拔牌组 + 1/3高威胁合法选拔牌组",
    ]
    preliminary: List[Tuple[DeckScore, List[Card], List[str]]] = []

    expanded = expand_selection_candidates(candidates)
    process.append(f"选拔段：{len(candidates) if isinstance(candidates, list) else 0}段")
    process.append(f"展开完整卡组：{len(expanded)}套")
    preliminary = [
        (score_deck(deck, rules, weights, star_map, selection=labels), deck, labels)
        for deck, labels in expanded
        if deck_is_legal(deck, star_map)[0]
    ]
    process.append(f"合法完整卡组：{len(preliminary)}套")
    preliminary.sort(key=lambda item: item[0].final_score, reverse=True)

    if not preliminary:
        return {"process": process + ["没有找到合法卡组"], "results": [], "best": None}

    total_to_verify = len(preliminary)
    stage_one_budget = max(MIN_DRAFT_VERIFY_SECONDS, time_budget * DRAFT_STAGE_ONE_RATIO)
    stage_two_budget = max(MIN_DRAFT_VERIFY_SECONDS, time_budget * DRAFT_STAGE_TWO_RATIO)
    process.append(
        f"步进深算：第一段全量{stage_one_budget:.1f}秒，第二段TopK追加{stage_two_budget:.1f}秒"
    )

    stage_one_samples, worker_count, seconds_per_deck = collect_samples_parallel(
        preliminary,
        rules,
        star_map,
        all_cards,
        stage_one_budget,
    )
    stage_one_total_simulations = sum(len(samples.get("scores", [])) for samples in stage_one_samples)
    process.append(
        f"第一段全量深算：{worker_count}个进程，{total_to_verify}套，每套约{seconds_per_deck:.2f}秒，"
        f"共{stage_one_total_simulations}局"
    )

    stage_one_records = []
    for preliminary_item, samples in zip(preliminary, stage_one_samples):
        _, deck, labels = preliminary_item
        verification = summarize_verification_samples(samples)
        result = score_deck(deck, rules, weights, star_map, verification=verification, selection=labels)
        stage_one_records.append({
            "preliminary": preliminary_item,
            "samples": samples,
            "result": result,
        })
    stage_one_records.sort(key=lambda item: item["result"].final_score, reverse=True)

    refine_count = min(DRAFT_REFINE_TOP_K, total_to_verify)
    refine_records = stage_one_records[:refine_count]
    refine_candidates = [record["preliminary"] for record in refine_records]
    process.append(f"第二段追加深算：第一段前{refine_count}套进入追加验证")

    stage_two_samples, stage_two_workers, stage_two_seconds = collect_samples_parallel(
        refine_candidates,
        rules,
        star_map,
        all_cards,
        stage_two_budget,
    )
    process.append(f"第二段有界多进程：{stage_two_workers}个进程，每套约{stage_two_seconds:.2f}秒")

    refined_by_selection = {}
    for index, (record, extra_samples) in enumerate(zip(refine_records, stage_two_samples), start=1):
        _, deck, labels = record["preliminary"]
        merged_samples = merge_verification_samples(record["samples"], extra_samples)
        verification = summarize_verification_samples(merged_samples)
        result = score_deck(deck, rules, weights, star_map, verification=verification, selection=labels)
        refined_by_selection[tuple(labels)] = result
        process.append(
            f"追加深算 {index}/{refine_count}: {verification[1]['simulations']}局，"
            f"威胁{verification[1].get('threat_rounds', 0)}轮，"
            f"均值{verification[1]['average']}，P10下限{verification[1]['floor']}"
        )

    scored = []
    for record in stage_one_records:
        labels_key = tuple(record["result"].selection)
        scored.append(refined_by_selection.get(labels_key, record["result"]))
    scored.sort(key=lambda item: item.final_score, reverse=True)
    process.append("排序策略：27套均已深算，TopK候选使用两段合并样本")
    best = scored[0]
    elapsed = time.perf_counter() - started
    process.append(f"实际耗时：{elapsed:.2f}秒")
    process.append(f"最高分：{best.final_score}")
    if best.selection:
        process.append(f"推荐选择：{' + '.join(best.selection)}")
    process.extend([f"推荐理由：{reason}" for reason in best.reasons])
    if best.warnings:
        process.extend([f"风险提示：{warning}" for warning in best.warnings])
    return {
        "process": process,
        "best": best.__dict__,
        "results": [item.__dict__ for item in scored],
    }


@app.route("/")
def index():
    """Render the standalone analyzer page."""
    return HTML_PAGE


@app.route("/api/random", methods=["POST"])
def random_selection():
    """Generate random draft choices following the three-stage selection shape."""
    star_map = get_card_star_map()
    try:
        candidates = generate_random_selection_candidates(star_map)
    except ValueError as exc:
        return jsonify({"error": str(exc), "candidates": [], "rules": ["选拔"]}), 400
    return jsonify({
        "mode": "selection",
        "rules": ["选拔"],
        "candidates": candidates,
        "constraints": {
            "stage1": "3组选1，每组2张卡",
            "stage2": "3组选1，每组2张卡",
            "stage3": "3张选1",
            "expandedDecks": 27,
            "deckSize": 5,
            "starLimit": "★4以上≤2，★5≤1",
        },
    })


@app.route("/api/analyze", methods=["POST"])
@app.route("/api/draft/analyze", methods=["POST"])
def analyze():
    """Analyze draft candidates from the UI."""
    payload = request.get_json() or {}
    rule_payload = payload.get("rules")
    if rule_payload is not None:
        if isinstance(rule_payload, str):
            rule_payload = [item.strip() for item in rule_payload.replace("，", ",").split(",")]
        extra_rules = [rule for rule in rule_payload if rule != "选拔"]
    else:
        extra_rules = payload.get("extraRules", [])
        if isinstance(extra_rules, str):
            extra_rules = [extra_rules]
    rules = ["选拔"] + [rule for rule in extra_rules if rule in EXTRA_RULES][:2]
    weights = normalize_weights(payload.get("weights", {}))
    candidates = payload.get("candidates", payload.get("draftChoices", payload.get("choices", [])))
    result = analyze_candidates(
        "selection",
        candidates,
        rules,
        weights,
        payload.get("timeBudget", DEFAULT_TIME_BUDGET),
    )
    return jsonify(result)


def self_test() -> Dict:
    """Run a CLI smoke test for the draft analyzer flow."""
    star_map = get_card_star_map()
    candidates = generate_random_selection_candidates(star_map)
    return analyze_candidates("selection", candidates, ["选拔", "加算"], DEFAULT_WEIGHTS, time_budget=5.0)


HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>选拔分析器</title>
  <style>
    :root { color-scheme: light; --ink:#1d2433; --muted:#687386; --line:#d7dee8; --blue:#2267c7; --green:#1f8a5b; --red:#c74646; --bg:#f7f8fb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:18px 24px; background:#fff; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:16px; }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    main { display:grid; grid-template-columns: 390px 1fr; min-height:calc(100vh - 62px); }
    aside { padding:18px; border-right:1px solid var(--line); background:#fff; overflow:auto; }
    section { padding:18px; overflow:auto; }
    h2 { font-size:15px; margin:0 0 10px; }
    .block { padding:14px 0; border-bottom:1px solid var(--line); }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:9px 12px; cursor:pointer; font-weight:600; }
    button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
    button:hover { filter:brightness(0.98); }
    label { display:flex; align-items:center; gap:7px; color:var(--muted); font-size:14px; margin:7px 0; }
    input[type="number"] { width:76px; padding:7px; border:1px solid var(--line); border-radius:6px; }
    input[type="range"] { width:170px; }
    .rule-grid { display:grid; grid-template-columns: 1fr 1fr; gap:4px 12px; }
    .deck-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
    .stage-grid { display:grid; gap:14px; }
    .choice-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:10px; }
    .deck { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }
    .deck.best { border-color:var(--green); box-shadow:0 0 0 2px rgba(31,138,91,.12); }
    .deck-title { display:flex; justify-content:space-between; font-weight:700; margin-bottom:8px; }
    .cards { display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:8px; }
    .choice .cards { grid-template-columns: repeat(auto-fit, minmax(82px, 1fr)); }
    .card { min-height:92px; border:1px solid var(--line); border-radius:7px; padding:7px; background:#fbfcfe; text-align:center; font-size:12px; }
    .num { display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:2px; margin:3px 0; font-weight:700; }
    .num .top { grid-column:2; }
    .num .left { grid-column:1; }
    .num .right { grid-column:3; }
    .num .bottom { grid-column:2; }
    .meta { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .score { font-size:22px; color:var(--green); }
    .components { display:grid; grid-template-columns:repeat(5, 1fr); gap:6px; margin-top:10px; }
    .component { background:#f0f3f8; border-radius:6px; padding:7px; font-size:12px; }
    .component b { display:block; font-size:15px; margin-top:2px; }
    .process { background:#121722; color:#dfe7f3; border-radius:8px; padding:12px; min-height:180px; white-space:pre-wrap; font-family: Consolas, monospace; font-size:13px; }
    .warn { color:var(--red); }
    @media (max-width: 900px) { main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .cards { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header>
    <h1>选拔分析器</h1>
    <div class="row">
      <button onclick="generateRandom()">随机生成</button>
      <button class="primary" onclick="analyze()">分析</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="block">
        <h2>规则</h2>
        <label><input type="checkbox" checked disabled> 选拔</label>
        <div class="rule-grid" id="rules"></div>
      </div>
      <div class="block">
        <h2>权重</h2>
        <div id="weights"></div>
        <label>深算秒数 <input id="timeBudget" type="range" min="5" max="20" step="0.5" value="7" oninput="document.getElementById('timeValue').textContent=this.value"><span id="timeValue">7</span></label>
      </div>
    </aside>
    <section>
      <div class="stage-grid" id="candidateArea"></div>
      <h2 style="margin-top:18px;">分析过程</h2>
      <div class="process" id="process">正在随机生成候选卡牌...</div>
      <h2 style="margin-top:18px;">分析结果</h2>
      <div class="deck-grid" id="resultArea"></div>
    </section>
  </main>
<script>
const ruleNames = ["同数","加算","逆转","王牌杀手","同类强化","同类弱化","秩序","混乱"];
const weightDefs = [["base","基础强度",0.18],["rule","规则适配",0.24],["synergy","组合协同",0.16],["stability","稳定性",0.10],["verification","深算验证",0.32],["risk","风险惩罚",0.12]];
const componentLabels = {
  base: "基础强度",
  rule: "规则适配",
  synergy: "组合协同",
  stability: "稳定性",
  verification: "深算验证",
  risk: "风险惩罚"
};
let selectionCandidates = [];

document.getElementById("rules").innerHTML = ruleNames.map(r => `<label><input type="checkbox" name="extraRule" value="${r}" onchange="limitRules(this)"> ${r}</label>`).join("");
document.getElementById("weights").innerHTML = weightDefs.map(([id,label,val]) => `<label>${label}<input id="w_${id}" type="range" min="0" max="1" step="0.01" value="${val}" oninput="document.getElementById('v_${id}').textContent=this.value"><span id="v_${id}">${val}</span></label>`).join("");

function selectedRules(){ return [...document.querySelectorAll('input[name="extraRule"]:checked')].map(x => x.value).slice(0, 2); }
function weights(){ const o={}; weightDefs.forEach(([id]) => o[id] = Number(document.getElementById("w_"+id).value)); return o; }
function limitRules(changed){
  const checked = [...document.querySelectorAll('input[name="extraRule"]:checked')];
  if (checked.length > 2) changed.checked = false;
}
function cardHtml(c){
  const star = c.star == null ? "" : ` ★${c.star}`;
  return `<div class="card"><div class="meta">#${c.id}${star}</div><div class="num"><span class="top">${c.up}</span><span class="left">${c.left}</span><span class="right">${c.right}</span><span class="bottom">${c.down}</span></div><div class="meta">${c.type || "无类型"}</div></div>`;
}
function deckHtml(deck, title, score, components, reasons, best){
  const comp = components ? `<div class="components">${Object.entries(components).map(([k,v]) => `<div class="component">${componentLabels[k] || k}<b>${v}</b></div>`).join("")}</div>` : "";
  const reasonHtml = reasons ? `<div class="meta" style="margin-top:8px;white-space:normal">${reasons.join("；")}</div>` : "";
  return `<div class="deck ${best ? "best" : ""}"><div class="deck-title"><span>${title}</span>${score == null ? "" : `<span class="score">${score}</span>`}</div><div class="cards">${deck.map(cardHtml).join("")}</div>${comp}${reasonHtml}</div>`;
}
function choiceCards(choice){
  if (choice.cards) return choice.cards;
  if (typeof choice.left === "object" || typeof choice.right === "object") return [choice.left, choice.right].filter(Boolean);
  return [choice];
}
function choiceHtml(choice, title){
  return `<div class="deck choice"><div class="deck-title"><span>${title}</span></div><div class="cards">${choiceCards(choice).map(cardHtml).join("")}</div></div>`;
}
function renderCandidates(){
  const area = document.getElementById("candidateArea");
  if (!selectionCandidates.length) {
    area.innerHTML = "";
    return;
  }
  area.innerHTML = selectionCandidates.map((stage, stageIndex) => `
    <div>
      <h2>备选${stageIndex + 1}</h2>
      <div class="choice-grid">${stage.map((choice, choiceIndex) => choiceHtml(choice, `候选 ${choiceIndex + 1}`)).join("")}</div>
    </div>
  `).join("");
}
async function generateRandom(){
  document.getElementById("process").textContent = "正在随机生成候选卡牌...";
  document.getElementById("resultArea").innerHTML = "";
  const res = await fetch("/api/random", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({})});
  const data = await res.json();
  if (!res.ok) {
    document.getElementById("process").textContent = data.error || "随机生成失败";
    return;
  }
  selectionCandidates = data.candidates || [];
  renderCandidates();
  document.getElementById("process").textContent = "已随机生成候选卡牌。";
}
async function analyze(){
  if (!selectionCandidates.length) await generateRandom();
  document.getElementById("process").textContent = "正在深算，请稍等...";
  const res = await fetch("/api/analyze", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({mode:"selection", candidates:selectionCandidates, rules:["选拔"].concat(selectedRules()), weights:weights(), timeBudget:Number(document.getElementById("timeBudget").value)})});
  const data = await res.json();
  document.getElementById("process").textContent = (data.process || []).map((x,i) => `${i+1}. ${x}`).join("\n");
  document.getElementById("resultArea").innerHTML = (data.results || []).map((r,i) => {
    const title = i === 0 ? `推荐卡组：${(r.selection || []).join(" + ")}` : `${(r.selection || []).join(" + ")}`;
    return deckHtml(r.deck, title, r.final_score, r.components, r.reasons.concat(r.warnings || []), i === 0);
  }).join("");
}
generateRandom();
</script>
</body>
</html>
"""


def main() -> None:
    """Run self-test or start the standalone web app."""
    parser = argparse.ArgumentParser(description="Standalone draft analyzer")
    parser.add_argument("--self-test", action="store_true", help="run one analyzer smoke test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    print(f"Draft analyzer running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
