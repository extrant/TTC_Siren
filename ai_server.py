from flask import Flask, request, jsonify
import pandas as pd
from core.card import Card
from core.player import Player
from core.board import Board
from core.game_state import GameState
from ai.ai import find_best_move_parallel, evaluate_state
from ai.monte_carlo import monte_carlo_best_move
from ai.unknown_card_handler import initialize_unknown_card_handler, get_unknown_card_handler
import os
import codecs
import threading
import random
import itertools
import time

# 全局缓存和唯一ID查找表
_card_db = None
_all_cards = None
_card_lookup = None
_card_lock = threading.Lock()
_card_star_map = None
_card_type_map = None
_handler_initialized = False

def get_card_db():
    global _card_db
    if _card_db is None:
        with _card_lock:
            if _card_db is None:
                _card_db = pd.read_csv('data/幻卡数据库.csv')
    return _card_db

def get_all_cards():
    global _all_cards
    if _all_cards is None:
        db = get_card_db()
        _all_cards = [
            Card(
                up=int(row['Top']),
                right=int(row['Right']),
                down=int(row['Bottom']),
                left=int(row['Left']),
                owner=None,
                card_id=int(row['序号']),
                card_type=row['TripleTriadCardType'] if pd.notna(row['TripleTriadCardType']) and row['TripleTriadCardType'] != '' else None
            )
            for _, row in db.iterrows()
        ]
    return _all_cards

def get_card_lookup():
    global _card_lookup
    if _card_lookup is None:
        db = get_card_db()
        # 用四面数值唯一确定卡牌ID
        _card_lookup = {}
        for _, row in db.iterrows():
            key = (int(row['Top']), int(row['Right']), int(row['Bottom']), int(row['Left']))
            _card_lookup[key] = int(row['序号'])
    return _card_lookup

def get_card_star_map():
    global _card_star_map
    if _card_star_map is None:
        db = get_card_db()
        _card_star_map = {int(row['序号']): int(row['星级']) for _, row in db.iterrows()}
    return _card_star_map

def get_card_type_map():
    global _card_type_map
    if _card_type_map is None:
        db = get_card_db()
        _card_type_map = {}
        for _, row in db.iterrows():
            card_id = int(row['序号'])
            card_type = row['TripleTriadCardType'] if pd.notna(row['TripleTriadCardType']) and row['TripleTriadCardType'] != '' else None
            _card_type_map[card_id] = card_type
    return _card_type_map

def ensure_handler_initialized():
    """确保未知卡牌处理器已初始化"""
    global _handler_initialized
    if not _handler_initialized:
        all_cards = get_all_cards()
        card_type_map = get_card_type_map()
        card_star_map = get_card_star_map()
        initialize_unknown_card_handler(all_cards, card_type_map, card_star_map)
        _handler_initialized = True

def parse_owner(owner):
    # 1=蓝方, 2=红方
    return 'blue' if owner == 1 else 'red'

def find_card_id_by_stats(up, right, down, left):
    lookup = get_card_lookup()
    return lookup.get((up, right, down, left), None)

def parse_board(board_json):
    board = Board()
    type_map = get_card_type_map()
    for item in board_json:
        r, c = item['pos']
        up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
        card_id = find_card_id_by_stats(up, right, down, left)
        if card_id is None:
            raise ValueError(f"Board card not found in database: U{up} R{right} D{down} L{left}")
        card_type = type_map.get(card_id)
        card = Card(
            up=up,
            right=right,
            down=down,
            left=left,
            owner=parse_owner(item['owner']),
            card_id=card_id,
            card_type=card_type
        )
        board.place_card(r, c, card)
    return board

def _board_occupancy(board_state):
    """返回棋盘已占用格数。"""
    if not board_state:
        return 0

    occupied = 0
    for r in range(3):
        for c in range(3):
            if board_state.get_card(r, c) is not None:
                occupied += 1
    return occupied

def _score_unknown_candidate(card, board_state, rules, owner):
    """评估未知候选牌在当前棋盘上的最大即时威胁。"""
    if not board_state:
        return sum([card.up, card.right, card.down, card.left])

    directions = [
        (-1, 0, 'up', 'down'),
        (1, 0, 'down', 'up'),
        (0, -1, 'left', 'right'),
        (0, 1, 'right', 'left'),
    ]
    best_score = 0.0

    for row in range(3):
        for col in range(3):
            if not board_state.is_empty(row, col):
                continue

            score = 0.0
            same_matches = []
            plus_sums = {}

            for dr, dc, my_dir, opp_dir in directions:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < 3 and 0 <= nc < 3):
                    continue

                target = board_state.get_card(nr, nc)
                if not target:
                    continue

                target_is_enemy = target.owner != owner
                if target_is_enemy and card.compare_values(my_dir, target, opp_dir, rules) == 1:
                    score += 6.0

                my_value = card.get_effective_value(my_dir, rules)
                target_value = target.get_effective_value(opp_dir, rules)
                if my_value == target_value:
                    same_matches.append((target, target_is_enemy))

                plus_sum = my_value + target_value
                plus_sums.setdefault(plus_sum, []).append((target, target_is_enemy))

            if '同数' in rules and len(same_matches) >= 2 and any(is_enemy for _, is_enemy in same_matches):
                score += 8.0 * sum(1 for _, is_enemy in same_matches if is_enemy)

            if '加算' in rules:
                for matches in plus_sums.values():
                    if len(matches) >= 2 and any(is_enemy for _, is_enemy in matches):
                        score += 8.0 * sum(1 for _, is_enemy in matches if is_enemy)

            best_score = max(best_score, score)

    value_pressure = max(card.up, card.right, card.down, card.left) + sum([card.up, card.right, card.down, card.left]) / 40.0
    return best_score * 10.0 + value_pressure

def _select_unknown_cards_for_slots(candidates, slot_count, board_state, rules, owner, is_opponent):
    """从未知候选池中选择实际填入手牌槽位的卡牌。"""
    if len(candidates) <= slot_count:
        return list(candidates)

    occupied = _board_occupancy(board_state)
    should_risk_select = is_opponent and (occupied >= 4 or (slot_count <= 2 and occupied >= 3))
    if not should_risk_select:
        return random.sample(candidates, slot_count)

    ranked = sorted(
        candidates,
        key=lambda card: (
            _score_unknown_candidate(card, board_state, rules or [], owner),
            card.card_id or 0,
        ),
        reverse=True,
    )
    selected = ranked[:slot_count]
    print(
        f"Endgame risk selection: {len(candidates)} candidates → {slot_count} slots "
        f"(occupied={occupied}/9)"
    )
    return selected

def _is_generated_unknown_card(card):
    return bool(
        getattr(card, '_is_generated', False) or
        getattr(card, '_is_prediction', False) or
        getattr(card, 'card_id', None) is None or
        (getattr(card, 'card_id', 0) is not None and getattr(card, 'card_id', 0) >= 1000)
    )

def _count_unknown_slots_from_hand(hand_cards):
    return sum(1 for card in hand_cards if _is_generated_unknown_card(card))

def _get_unknown_hand_indices(hand_cards):
    return [idx for idx, card in enumerate(hand_cards) if _is_generated_unknown_card(card)]

def _base_card_id(card):
    """返回推测牌对应的真实卡牌 ID。"""
    card_id = getattr(card, 'card_id', None)
    if card_id is None:
        return None
    if card_id >= 1000:
        return card_id - 1000
    return card_id

def _known_card_ids_on_board(board_state):
    """收集棋盘上已出现的真实卡牌 ID。"""
    known_ids = set()
    if not board_state:
        return known_ids

    for row in range(3):
        for col in range(3):
            card = board_state.get_card(row, col)
            card_id = _base_card_id(card) if card else None
            if card_id is not None:
                known_ids.add(card_id)
    return known_ids

def _known_card_ids_in_hands(*hands):
    """收集明牌手牌中的真实卡牌 ID。"""
    known_ids = set()
    for hand in hands:
        for card in hand:
            if _is_generated_unknown_card(card):
                continue
            card_id = _base_card_id(card)
            if card_id is not None:
                known_ids.add(card_id)
    return known_ids

def _high_star_usage(cards, star_map):
    """统计一组已知卡牌占用的高星配额。"""
    high_star_count = 0
    five_star_count = 0
    for card in cards:
        if _is_generated_unknown_card(card):
            continue
        star = star_map.get(_base_card_id(card), 1)
        if star >= 4:
            high_star_count += 1
        if star == 5:
            five_star_count += 1
    return high_star_count, five_star_count

def _is_legal_unknown_assignment(assignment, known_opp_cards, star_map):
    """检查未知牌组合是否满足对手卡组星级限制。"""
    high_star_count, five_star_count = _high_star_usage(known_opp_cards, star_map)
    seen_ids = set()

    for card in assignment:
        card_id = _base_card_id(card)
        if card_id in seen_ids:
            return False
        seen_ids.add(card_id)

        star = star_map.get(card_id, 1)
        if star >= 4:
            high_star_count += 1
        if star == 5:
            five_star_count += 1

    return high_star_count <= 2 and five_star_count <= 1

def _build_legal_unknown_candidates(base_state, opp_hand, board_state, opp_owner):
    """枚举符合当前已知信息和卡组限制的对手未知牌候选。

    Args:
        base_state: 当前局面，用于读取双方手牌。
        opp_hand: 对手当前手牌。
        board_state: 当前棋盘。
        opp_owner: 对手颜色。

    Returns:
        按数据库顺序生成的合法未知候选卡牌列表。
    """
    star_map = get_card_star_map()
    type_map = get_card_type_map()
    known_global_ids = _known_card_ids_on_board(board_state)
    for player in base_state.players:
        known_global_ids.update(_known_card_ids_in_hands(player.hand))

    known_opp_cards = [card for card in opp_hand if not _is_generated_unknown_card(card)]
    candidate_cards = []
    for card in get_all_cards():
        card_id = _base_card_id(card)
        if card_id in known_global_ids:
            continue

        candidate = Card(
            card.base_up,
            card.base_right,
            card.base_down,
            card.base_left,
            owner=opp_owner,
            card_id=card_id,
            card_type=type_map.get(card_id),
            can_use=True,
        )
        if not _is_legal_unknown_assignment([candidate], known_opp_cards, star_map):
            continue
        candidate._is_generated = True
        candidate._is_prediction = True
        candidate_cards.append(candidate)

    return candidate_cards

def _unique_unknown_candidates(candidates):
    """按卡牌身份去重，保留候选池中的首个副本。"""
    unique = []
    seen = set()
    for card in candidates:
        key = (
            card.card_id,
            card.base_up,
            card.base_right,
            card.base_down,
            card.base_left,
            card.card_type,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(card)
    return unique

def _copy_state_with_unknown_assignment(base_state, opp_player_idx, unknown_indices, sampled_cards):
    """复制局面，并把一组未知牌候选填入对手手牌槽位。"""
    scenario = base_state.copy()
    opp_hand_copy = scenario.players[opp_player_idx].hand
    for idx, sampled_card in zip(unknown_indices, sampled_cards):
        copied_card = sampled_card.copy()
        copied_card.owner = opp_hand_copy[idx].owner
        copied_card.can_use = opp_hand_copy[idx].can_use
        opp_hand_copy[idx] = copied_card
    return scenario

def _build_endgame_scenarios(base_state, opp_hand, used_cards, rules, board_state, opp_owner, opp_player_idx, sample_count):
    """构建残局信息集场景，优先覆盖对手高威胁未知牌组合。

    Args:
        base_state: 当前搜索局面。
        opp_hand: 当前对手手牌，未知槽位已由预测牌占位。
        used_cards: 已知使用过的卡牌 ID。
        rules: 当前规则列表。
        board_state: 当前棋盘。
        opp_owner: 对手颜色。
        opp_player_idx: 对手在 GameState.players 中的索引。
        sample_count: 最多构建的可能世界数量。

    Returns:
        一组 GameState 副本；无未知牌时仅返回当前局面。
    """
    handler = get_unknown_card_handler()
    unknown_indices = _get_unknown_hand_indices(opp_hand)
    if not unknown_indices:
        return [base_state]
    if not handler:
        return [base_state]

    known_opp_hand = [card for card in opp_hand if not _is_generated_unknown_card(card)]
    legal_candidates = _build_legal_unknown_candidates(base_state, opp_hand, board_state, opp_owner)
    if legal_candidates:
        candidates = legal_candidates
        print(
            f"Endgame legal enumeration: {len(legal_candidates)} candidates "
            f"for {len(unknown_indices)} unknown slots"
        )
    else:
        candidates = handler.generate_opponent_cards(
            count=len(unknown_indices),
            rules=rules,
            used_cards=used_cards.copy(),
            board_state=board_state,
            known_hand=known_opp_hand,
            owner=opp_owner,
            can_use=True
        )
    candidates = _unique_unknown_candidates(candidates)
    if not candidates:
        return [base_state]

    ranked_candidates = sorted(
        candidates,
        key=lambda card: (
            _score_unknown_candidate(card, board_state, rules or [], opp_owner),
            card.up + card.right + card.down + card.left,
            -(card.card_id or 0),
        ),
        reverse=True,
    )

    unknown_count = len(unknown_indices)
    candidate_limit = min(len(ranked_candidates), max(sample_count * unknown_count, unknown_count))
    scenario_limit = max(1, sample_count)
    scenarios = []
    seen_assignments = set()
    star_map = get_card_star_map()

    for assignment in itertools.permutations(ranked_candidates[:candidate_limit], unknown_count):
        assignment_key = tuple(card.card_id for card in assignment)
        if assignment_key in seen_assignments:
            continue
        if not _is_legal_unknown_assignment(assignment, known_opp_hand, star_map):
            continue
        seen_assignments.add(assignment_key)
        scenarios.append(_copy_state_with_unknown_assignment(
            base_state,
            opp_player_idx,
            unknown_indices,
            assignment,
        ))
        if len(scenarios) >= scenario_limit:
            break

    return scenarios or [base_state]

def _endgame_score(state, ai_player_idx):
    """返回终局或近终局时 AI 视角的牌数差。"""
    red_count, blue_count = state.count_cards()
    return (red_count - blue_count) if ai_player_idx == 0 else (blue_count - red_count)

def _endgame_state_key(state):
    """为残局完全搜索构建稳定缓存键。"""
    board_key = []
    for r in range(3):
        for c in range(3):
            card = state.board.get_card(r, c)
            if card is None:
                board_key.append(None)
            else:
                board_key.append((
                    card.card_id,
                    card.owner,
                    card.type_modifier,
                    card.base_up,
                    card.base_right,
                    card.base_down,
                    card.base_left,
                ))

    hand_key = []
    for player in state.players:
        hand_key.append(tuple(
            (
                card.card_id,
                card.owner,
                card.can_use,
                card.type_modifier,
                card.base_up,
                card.base_right,
                card.base_down,
                card.base_left,
            )
            for card in player.hand
        ))

    return (state.current_player_idx, tuple(board_key), tuple(hand_key), tuple(state.rules))

def _solve_endgame_exact(state, ai_player_idx, cache=None):
    """对空位很少的残局做完整 Minimax 终局搜索。

    Args:
        state: 当前局面，会在搜索中原地落子并撤销。
        ai_player_idx: AI 玩家索引。
        cache: 本次信息集搜索的局面缓存。

    Returns:
        AI 视角终局牌数差；正数更好，负数更差。
    """
    if cache is None:
        cache = {}
    if state.is_game_over():
        return _endgame_score(state, ai_player_idx)

    key = _endgame_state_key(state)
    if key in cache:
        return cache[key]

    moves = state.get_available_moves()
    if not moves:
        score = _endgame_score(state, ai_player_idx)
        cache[key] = score
        return score

    maximizing = state.current_player_idx == ai_player_idx
    if maximizing:
        best_score = float('-inf')
        for card, (row, col) in moves:
            move_record = state.make_move(row, col, card)
            if move_record is None:
                continue
            try:
                best_score = max(best_score, _solve_endgame_exact(state, ai_player_idx, cache))
            finally:
                state.undo_move(move_record)
    else:
        best_score = float('inf')
        for card, (row, col) in moves:
            move_record = state.make_move(row, col, card)
            if move_record is None:
                continue
            try:
                best_score = min(best_score, _solve_endgame_exact(state, ai_player_idx, cache))
            finally:
                state.undo_move(move_record)

    cache[key] = best_score
    return best_score

class ConsoleSearchReporter:
    """节流输出服务端搜索速度，避免刷屏拖慢搜索。"""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.last_print = 0.0
        self.endgame_nodes = 0
        self.endgame_start = None

    def on_minimax_progress(self, progress_info):
        """打印 Minimax 实时速度。"""
        now = time.time()
        if now - self.last_print < self.interval:
            return
        self.last_print = now

        stats = progress_info.get('stats', {})
        elapsed = max(progress_info.get('time_elapsed', 0.0), 1e-6)
        nodes = progress_info.get('nodes_searched', 0)
        nps = nodes / elapsed
        print(
            f"[Search] depth={progress_info.get('depth')} "
            f"nodes={nodes:,} nps={nps:,.0f} "
            f"tt={stats.get('tt_hit_rate', 0) * 100:.1f}% "
            f"cutoff={stats.get('cutoff_rate', 0) * 100:.1f}% "
            f"elapsed={elapsed:.2f}s"
        )

    def start_endgame(self):
        """开始记录信息集残局速度。"""
        self.endgame_nodes = 0
        self.endgame_start = time.time()
        self.last_print = 0.0

    def on_endgame_node(self, move_index, move_count, scenario_index, scenario_count):
        """打印信息集残局实时速度。"""
        self.endgame_nodes += 1
        now = time.time()
        if now - self.last_print < self.interval:
            return
        self.last_print = now
        elapsed = max(now - (self.endgame_start or now), 1e-6)
        print(
            f"[Endgame] move={move_index}/{move_count} "
            f"scenario={scenario_index}/{scenario_count} "
            f"nodes={self.endgame_nodes:,} nps={self.endgame_nodes / elapsed:,.0f} "
            f"elapsed={elapsed:.2f}s"
        )

def _best_immediate_reply_score(state_after_my_move, ai_player_idx):
    """计算对手对当前局面的最佳即时回应分数。"""
    reply_moves = state_after_my_move.get_available_moves()
    if not reply_moves:
        return evaluate_state(state_after_my_move, ai_player_idx)

    best_reply = float('inf')
    for reply_card, (row, col) in reply_moves:
        reply_state = state_after_my_move.copy()
        scenario_card = next(
            (c for c in reply_state.current_player.hand if c.card_id == reply_card.card_id),
            None
        )
        if scenario_card is None:
            continue
        if reply_state.make_move(row, col, scenario_card) is None:
            continue
        best_reply = min(best_reply, evaluate_state(reply_state, ai_player_idx))

    return best_reply if best_reply != float('inf') else evaluate_state(state_after_my_move, ai_player_idx)

def _evaluate_endgame_move_robustly(base_state, move, scenario_states, ai_player_idx,
                                   safety_margin=0.75, progress_reporter=None,
                                   move_index=1, move_count=1):
    """对单个走法进行信息集残局评分。"""
    card, (row, col) = move
    scenario_scores = []
    safety_votes = 0
    corner_risk = _calculate_corner_safety_risk(card, row, col, base_state.board, base_state.rules)
    exact_cache = {}
    use_exact_solver = _board_occupancy(base_state.board) >= 5

    scenario_count = len(scenario_states)
    for scenario_index, scenario in enumerate(scenario_states, start=1):
        if progress_reporter:
            progress_reporter.on_endgame_node(move_index, move_count, scenario_index, scenario_count)
        scenario_eval = evaluate_state(scenario, ai_player_idx)
        scenario_after = scenario.copy()
        scenario_card = next((c for c in scenario_after.current_player.hand if c.card_id == card.card_id), None)
        if scenario_card is None:
            continue
        if scenario_after.make_move(row, col, scenario_card) is None:
            continue

        if use_exact_solver:
            reply_score = _solve_endgame_exact(scenario_after, ai_player_idx, cache=exact_cache)
        else:
            reply_score = _best_immediate_reply_score(scenario_after, ai_player_idx)
        scenario_scores.append(reply_score)
        if (use_exact_solver and reply_score >= 0) or (
            not use_exact_solver and reply_score >= scenario_eval - safety_margin
        ):
            safety_votes += 1

    if not scenario_scores:
        return float('-inf'), 0.0, float('-inf'), corner_risk

    avg_score = sum(scenario_scores) / len(scenario_scores)
    worst_score = min(scenario_scores)
    robust_score = avg_score * 0.45 + worst_score * 0.55
    safety_ratio = safety_votes / len(scenario_scores)
    if not use_exact_solver and _is_corner_position(row, col) and corner_risk >= 5.0 and safety_ratio < 0.8:
        return float('-inf'), safety_ratio, float('-inf'), corner_risk

    final_score = robust_score + safety_ratio * 2.0 - corner_risk * 0.15
    return final_score, safety_ratio, robust_score, corner_risk

def select_endgame_robust_move(base_state, scenario_states, ai_player_idx, progress_reporter=None):
    """在残局场景中选择鲁棒性更强的走法。"""
    moves = base_state.get_available_moves()
    if not moves:
        return None, []

    scored_moves = []
    if progress_reporter:
        progress_reporter.start_endgame()

    move_count = len(moves)
    for move_index, move in enumerate(moves, start=1):
        final_score, safety_ratio, robust_score, corner_risk = _evaluate_endgame_move_robustly(
            base_state,
            move,
            scenario_states,
            ai_player_idx,
            progress_reporter=progress_reporter,
            move_index=move_index,
            move_count=move_count,
        )
        scored_moves.append({
            'move': move,
            'final_score': final_score,
            'safety_ratio': safety_ratio,
            'robust_score': robust_score,
            'corner_risk': corner_risk,
        })

    fully_safe_moves = [item for item in scored_moves if item['safety_ratio'] >= 1.0]
    safe_moves = fully_safe_moves or [
        item for item in scored_moves
        if item['safety_ratio'] >= 0.75 and (
            item['corner_risk'] < 5.0 or item['safety_ratio'] >= 0.8
        )
    ]
    ranked_moves = safe_moves if safe_moves else scored_moves
    ranked_moves.sort(
        key=lambda item: (item['safety_ratio'], item['final_score'], item['robust_score'], -item['corner_risk']),
        reverse=True,
    )

    best_item = ranked_moves[0]
    return best_item['move'], scored_moves

def _should_use_endgame_robust_mode(board_state, opp_unknown_count):
    """判断是否启用信息集残局求解。"""
    occupied = _board_occupancy(board_state)
    return occupied >= 5 and opp_unknown_count <= 2

def _move_key(move):
    card, (row, col) = move
    return (card.card_id, row, col)

def _is_corner_position(row, col):
    return (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]

def _calculate_corner_safety_risk(card, row, col, board_state, rules):
    """计算残局角落安全风险。"""
    occupied = _board_occupancy(board_state)
    if occupied < 5:
        return 0.0

    stage_weight = 1.0 + min((occupied - 4) / 4.0, 1.0)
    risk = 0.0
    attackable_sides = 0
    weak_sides = 0
    directions = [
        (-1, 0, 'up', 'down'),
        (1, 0, 'down', 'up'),
        (0, -1, 'left', 'right'),
        (0, 1, 'right', 'left'),
    ]

    for dr, dc, my_dir, opp_dir in directions:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < 3 and 0 <= nc < 3):
            continue

        attackable_sides += 1
        my_value = card.get_effective_value(my_dir, rules or [])
        if my_value <= 3:
            weak_sides += 1
            risk += (4 - my_value) * 2.2
        elif my_value == 4:
            risk += 1.0

        adj_card = board_state.get_card(nr, nc) if board_state else None
        if adj_card and adj_card.owner != card.owner:
            opp_value = adj_card.get_effective_value(opp_dir, rules or [])
            if opp_value > my_value:
                risk += (opp_value - my_value) * 1.5
            elif opp_value == my_value:
                risk += 1.0

    if _is_corner_position(row, col) and weak_sides > 0:
        risk += weak_sides * 1.5
    elif attackable_sides >= 3 and weak_sides > 0:
        risk += weak_sides * 1.0

    return risk * stage_weight

def parse_hand(hand_json, owner, used_cards, rules=None, board_state=None, is_opponent=False, id_offset=1000, skip_sampling=False):
    """
    解析手牌，智能处理未知卡牌

    Args:
        hand_json: 手牌JSON数据
        owner: 卡牌所有者
        used_cards: 已使用的卡牌ID集合
        rules: 当前游戏规则（用于智能采样）
        board_state: 当前棋盘状态（用于智能采样）
        is_opponent: 是否为对手手牌（启用行为建模）
        id_offset: ID偏移量（已弃用）
        skip_sampling: 跳过智能采样，保留未知卡牌为占位符（蒙特卡洛求解器用）
    """
    hand_slots = []
    known_cards = []
    unknown_count = 0
    type_map = get_card_type_map()

    # 第一遍：解析所有槽位，保留原始顺序
    for item in hand_json:
        if all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
            hand_slots.append(("unknown", item))
            unknown_count += 1
        else:
            up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
            card_id = find_card_id_by_stats(up, right, down, left)
            if card_id is None:
                raise ValueError(f"Hand card not found in database: U{up} R{right} D{down} L{left}")
            card_type = type_map.get(card_id)
            c = Card(up, right, down, left, owner, card_id, card_type,
                    item.get('canUse', True))
            hand_slots.append(("known", c))
            known_cards.append(c)
            used_cards.add(card_id)

    generated_unknown_cards = []

    # 第二遍：智能处理未知卡牌（或跳过采样保留占位符）
    if unknown_count > 0:
        if skip_sampling:
            print(f"Skipped sampling: kept {unknown_count} unknown cards as placeholders for Monte Carlo")
        else:
            card_type_label = "opponent" if is_opponent else "own"
            print(f"Processing {unknown_count} unknown {card_type_label} cards for {owner}")
            ensure_handler_initialized()
            handler = get_unknown_card_handler()

            if handler and rules:
                if is_opponent:
                    unknown_cards = handler.generate_opponent_cards(
                        count=unknown_count,
                        rules=rules,
                        used_cards=used_cards.copy(),
                        board_state=board_state,
                        known_hand=known_cards,
                        owner=owner,
                        can_use=True
                    )
                else:
                    unknown_cards = handler.generate_unknown_cards(
                        count=unknown_count,
                        rules=rules,
                        used_cards=used_cards.copy(),
                        board_state=board_state,
                        known_hand=known_cards,
                        owner=owner,
                        can_use=True
                    )

                generated_unknown_cards = _select_unknown_cards_for_slots(
                    unknown_cards,
                    unknown_count,
                    board_state,
                    rules,
                    owner,
                    is_opponent,
                )
                print(f"Generated {len(generated_unknown_cards)} {card_type_label} cards for {unknown_count} unknown slots")
            else:
                print("Using fallback sampling for unknown cards")
                all_cards = get_all_cards()
                sample_size = min(unknown_count * 5, len(all_cards))
                sampled_cards = random.sample(all_cards, sample_size)
                if len(sampled_cards) > unknown_count:
                    sampled_cards = random.sample(sampled_cards, unknown_count)

                generated_unknown_cards = [
                    Card(card.up, card.right, card.down, card.left, owner,
                         card.card_id, card.card_type, True)
                    for card in sampled_cards
                ]
                print(f"Fallback generated {len(generated_unknown_cards)} cards for {unknown_count} unknown slots")

    # 第三遍：按原始顺序重建手牌
    hand = []
    unknown_idx = 0
    for kind, payload in hand_slots:
        if kind == "known":
            hand.append(payload)
        else:
            if unknown_idx < len(generated_unknown_cards):
                card = generated_unknown_cards[unknown_idx]
            else:
                card = Card(0, 0, 0, 0, owner, None, None, payload.get('canUse', True))
            card.can_use = payload.get('canUse', True)
            hand.append(card)
            unknown_idx += 1

    return hand

def parse_rules_and_open_mode(rules_str):
    # 解码unicode
    if isinstance(rules_str, str):
        try:
            rules_str = rules_str
        except Exception:
            pass
    rules = []
    open_mode = 'none'
    # 规则识别
    print(rules_str)
    if '全明牌' in rules_str:
        open_mode = 'all'
    elif '三明牌' in rules_str:
        open_mode = 'three'
    if '同数' in rules_str:
        rules.append('同数')
    if '加算' in rules_str:
        rules.append('加算')
    if '逆转' in rules_str:
        rules.append('逆转')
    if '王牌杀手' in rules_str:
        rules.append('王牌杀手')
    if '同类强化' in rules_str:
        rules.append('同类强化')
    if '同类弱化' in rules_str:
        rules.append('同类弱化')
    if '秩序' in rules_str:
        rules.append('秩序')
    if '混乱' in rules_str:
        rules.append('混乱')
    if '选拔' in rules_str:
        rules.append('选拔')

    # 允许最多4条规则
    # 可按逗号分割后去重
    return rules, open_mode

def complete_unknown_hand(hand, all_cards, used_cards):
    # hand: List[Card or None]
    completed = []
    for c in hand:
        if c is not None:
            completed.append(c)
            if c.card_id is not None:
                used_cards.add(c.card_id)
        else:
            # 用全牌池未用过的牌补全
            for card in all_cards:
                if card.card_id not in used_cards and card.owner is None:
                    completed.append(Card(card.up, card.right, card.down, card.left, None, card.card_id, card.card_type))
                    used_cards.add(card.card_id)
                    break
    return completed

def analyze_opponent_hand(opp_hand, rules, board):
    """分析对手手牌"""
    if not opp_hand:
        return {
            'predicted_cards': [],
            'total_unknown': 0,
            'strategy_analysis': '无对手手牌信息'
        }
    
    # 统计已知和未知卡牌
    known_cards = []
    predicted_cards = []
    unknown_count = 0
    
    star_map = get_card_star_map()
    
    for card in opp_hand:
        # 检查是否是原始未知卡牌（所有数值为0）
        if card.up == 0 and card.right == 0 and card.down == 0 and card.left == 0:
            unknown_count += 1
            continue
        
        # 检查是否是智能采样生成的卡牌
        star = star_map.get(card.card_id, '?')
        confidence = 1.0  # 默认已知卡牌
        reasoning = '已知卡牌'
        
        # 判断是否为预测卡牌的更准确方法
        is_predicted = False
        if hasattr(card, 'card_id') and card.card_id:
            # 如果card_id >= 1000，这是预测卡牌
            if card.card_id >= 1000:
                is_predicted = True
                unknown_count += 1  # 统计为未知卡牌
                confidence = get_prediction_confidence(card, rules)
                reasoning = get_prediction_reasoning(card, rules)
            # 或者，如果这是从unknown_card_handler生成的卡牌
            elif hasattr(card, '_is_generated') and card._is_generated:
                is_predicted = True
                unknown_count += 1  # 统计为未知卡牌
                confidence = get_prediction_confidence(card, rules)
                reasoning = get_prediction_reasoning(card, rules)
        
        # 生成卡牌显示信息
        card_display = format_card_display(card, star)
        card_info = {
            'card': card_display,
            'star': star,
            'confidence': confidence,
            'reasoning': reasoning,
            'is_predicted': is_predicted
        }
        
        if is_predicted or confidence < 1.0:
            predicted_cards.append(card_info)
        else:
            known_cards.append(card_info)
    
    # 分析未知卡牌策略
    strategy_analysis = analyze_opponent_strategy(rules, board, known_cards)
    
    # 选拔规则特殊分析
    draft_analysis = None
    if '选拔' in rules:
        draft_analysis = analyze_draft_mode_constraints(opp_hand, board, rules)
        if draft_analysis:
            strategy_analysis += f" | 选拔分析：{draft_analysis}"
    
    # 如果还有未处理的未知卡牌，生成通用预测
    remaining_unknown = unknown_count - len(predicted_cards)
    if remaining_unknown > 0:
        generic_predictions = predict_unknown_cards(rules, board, remaining_unknown)
        predicted_cards.extend(generic_predictions)
    
    # 合并所有卡牌信息
    all_cards = known_cards + predicted_cards
    
    result = {
        'predicted_cards': all_cards[:10],  # 最多显示10张
        'total_unknown': unknown_count,
        'strategy_analysis': strategy_analysis
    }
    
    # 如果有选拔规则，添加星级分析
    if '选拔' in rules and draft_analysis:
        result['draft_analysis'] = draft_analysis
    
    return result

def get_prediction_confidence(card, rules):
    """根据规则和卡牌特征计算预测置信度（支持多规则组合）"""
    confidence = 0.6  # 基础置信度
    rule_bonuses = []  # 记录各规则的置信度加成
    
    # 选拔规则优先级最高
    if '选拔' in rules:
        # 选拔规则下，星级匹配的置信度很高
        rule_bonuses.append(0.85)
    
    # 连携类规则优先级最高
    if '同数' in rules and has_same_number_potential(card):
        if '加算' in rules and has_addition_potential(card):
            # 同时适合同数和加算的卡牌置信度最高
            rule_bonuses.append(0.9)
        else:
            rule_bonuses.append(0.8)
    elif '加算' in rules and has_addition_potential(card):
        rule_bonuses.append(0.7)
    
    # 逆转规则（使用有效数值）
    if '逆转' in rules:
        effective_values = [card.get_effective_value('up', []), card.get_effective_value('right', []), 
                           card.get_effective_value('down', []), card.get_effective_value('left', [])]
        avg_value = sum(effective_values) / 4
        if avg_value <= 5:  # 低数值卡牌在逆转规则下更有用
            rule_bonuses.append(0.8)
        elif avg_value >= 8:  # 高数值卡牌在逆转规则下不利
            rule_bonuses.append(0.4)
        else:
            rule_bonuses.append(0.6)
    
    # 王牌杀手规则（使用修正后数值）
    if '王牌杀手' in rules:
        effective_values = [card.get_modified_value('up'), card.get_modified_value('right'), 
                           card.get_modified_value('down'), card.get_modified_value('left')]
        ace_killer_count = sum(1 for v in effective_values if v in [1, 10])
        if ace_killer_count >= 2:
            rule_bonuses.append(0.85)  # 多个1或A的卡牌
        elif ace_killer_count == 1:
            rule_bonuses.append(0.75)  # 单个1或A
        else:
            rule_bonuses.append(0.5)   # 无1或A
    
    # 同类强化/弱化规则
    if ('同类强化' in rules or '同类弱化' in rules) and card.card_type:
        if '同类强化' in rules:
            rule_bonuses.append(0.7)  # 同类强化通常是好事
        else:  # 同类弱化
            rule_bonuses.append(0.5)  # 同类弱化是风险
    
    # 秩序/混乱规则影响
    if '秩序' in rules or '混乱' in rules:
        if hasattr(card, 'can_use') and not card.can_use:
            rule_bonuses.append(0.3)  # 不可使用的卡牌置信度很低
        else:
            rule_bonuses.append(0.6)  # 可使用的卡牌正常置信度
    
    # 计算最终置信度：取最高规则置信度，但有多规则加成
    if rule_bonuses:
        base_confidence = max(rule_bonuses)
        # 多规则组合加成：每多一条规则+0.05，最大不超过0.95
        multi_rule_bonus = min(0.05 * (len(rule_bonuses) - 1), 0.15)
        confidence = min(0.95, base_confidence + multi_rule_bonus)
    
    return confidence

def get_prediction_reasoning(card, rules):
    """根据规则和卡牌特征生成预测原因"""
    reasons = []
    
    # 选拔规则分析
    if '选拔' in rules:
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, '?')
        if star == 5:
            reasons.append('选拔模式下的5星王牌，战略价值极高')
        elif star == 4:
            reasons.append('选拔模式下的4星强力卡牌，关键时刻的选择')
        elif star in [1, 2]:
            reasons.append('选拔模式下的低星级卡牌，早期使用较安全')
        else:
            reasons.append('选拔模式下的中等星级卡牌，平衡性较好')
    
    # 连携类规则分析
    if '同数' in rules and has_same_number_potential(card):
        if '加算' in rules and has_addition_potential(card):
            reasons.append('同时适合同数和加算连携的复合战术卡牌')
        else:
            reasons.append('同数规则下可能用于设置连携陷阱')
    elif '加算' in rules and has_addition_potential(card):
        reasons.append('加算规则下适合数值组合连携')
    
    # 其他规则分析
    if '逆转' in rules:
        effective_values = [card.get_effective_value('up', []), card.get_effective_value('right', []), 
                           card.get_effective_value('down', []), card.get_effective_value('left', [])]
        avg_value = sum(effective_values) / 4
        if avg_value <= 5:
            reasons.append('逆转规则下的优势低数值卡牌')
        else:
            reasons.append('虽然数值较高，但可能用于特殊战术')
    
    if '王牌杀手' in rules:
        effective_values = [card.get_modified_value('up'), card.get_modified_value('right'), 
                           card.get_modified_value('down'), card.get_modified_value('left')]
        if 1 in effective_values or 10 in effective_values:
            reasons.append('王牌杀手规则下含1或A具有特殊效果')
        else:
            reasons.append('王牌杀手规则下的支援卡牌')
    
    # 如果没有特殊规则匹配，使用通用分析
    if not reasons:
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, 3)
        if star <= 2:
            reasons.append('低星级卡牌，可能用于早期控场')
        elif star >= 4:
            reasons.append('高星级卡牌，关键时刻的强力选择')
        else:
            reasons.append('中等星级卡牌，平衡性较好')
    
    return '；'.join(reasons)

def analyze_global_star_usage(board, my_hand_json, opp_hand_json):
    """分析全局星级使用情况（包括棋盘和所有已知手牌）"""
    star_map = get_card_star_map()
    type_map = get_card_type_map()
    global_usage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # 统计棋盘卡牌
    if board:
        for r in range(3):
            for c in range(3):
                card = board.get_card(r, c)
                if card and hasattr(card, 'card_id') and card.card_id:
                    star = star_map.get(card.card_id, 1)
                    global_usage[star] += 1
    
    # 统计己方已知手牌
    for item in my_hand_json:
        if not all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
            up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
            card_id = find_card_id_by_stats(up, right, down, left)
            if card_id:
                star = star_map.get(card_id, 1)
                global_usage[star] += 1
    
    # 统计对手已知手牌
    for item in opp_hand_json:
        if not all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
            up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
            card_id = find_card_id_by_stats(up, right, down, left)
            if card_id:
                star = star_map.get(card_id, 1)
                global_usage[star] += 1
    
    return global_usage

def analyze_draft_mode_constraints(all_hand, board, rules):
    """分析选拔模式下的星级约束"""
    star_map = get_card_star_map()
    
    # 统计全局星级使用情况
    global_star_usage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # 统计棋盘上的卡牌
    if board:
        for r in range(3):
            for c in range(3):
                card = board.get_card(r, c)
                if card and hasattr(card, 'card_id') and card.card_id:
                    star = star_map.get(card.card_id, 1)
                    global_star_usage[star] += 1
    
    # 统计手牌中的已知卡牌
    if all_hand:
        for card in all_hand:
            # 只统计有效的已知卡牌
            if (hasattr(card, 'card_id') and card.card_id and card.card_id > 0 and
                not (card.up == 0 and card.right == 0 and card.down == 0 and card.left == 0)):
                star = star_map.get(card.card_id, 1)
                global_star_usage[star] += 1
    
    # 计算剩余配额
    star_limits = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    remaining_quota = {}
    for star, limit in star_limits.items():
        used = global_star_usage.get(star, 0)
        remaining = max(0, limit - used)
        remaining_quota[star] = remaining
    
    # 生成分析报告
    analysis_parts = []
    
    # 检查是否有星级用完
    exhausted_stars = [str(star) for star, quota in remaining_quota.items() if quota == 0]
    if exhausted_stars:
        analysis_parts.append(f"{','.join(exhausted_stars)}星已用完")
    
    # 检查剩余高价值星级
    high_value_remaining = remaining_quota.get(4, 0) + remaining_quota.get(5, 0)
    if high_value_remaining > 0:
        analysis_parts.append(f"剩余{high_value_remaining}张高星卡牌")
    
    # 检查可用星级分布
    available_stars = [str(star) for star, quota in remaining_quota.items() if quota > 0]
    if available_stars:
        analysis_parts.append(f"可用星级:{','.join(available_stars)}")
    
    # 战略建议
    if remaining_quota.get(5, 0) > 0:
        analysis_parts.append("对手可能保留5星卡牌作为杀招")
    elif remaining_quota.get(4, 0) > 0:
        analysis_parts.append("对手可能依赖4星卡牌")
    
    return ' | '.join(analysis_parts) if analysis_parts else "星级分布正常"

def analyze_opponent_strategy(rules, board, known_cards):
    """分析对手策略"""
    strategies = []
    
    # 选拔规则优先分析
    if '选拔' in rules:
        strategies.append('严格控制星级配额，注意高星卡牌的战略性使用')
    
    # 检查连携类规则（最重要）
    if '同数' in rules and '加算' in rules:
        strategies.append('对手可能会同时利用同数和加算连携，设置复合陷阱')
    elif '同数' in rules:
        strategies.append('对手可能会设置同数连携陷阱，注意重复数值卡牌')
    elif '加算' in rules:
        strategies.append('对手可能会利用加算规则设置连携，注意互补数值组合')
    
    # 检查其他特殊规则
    if '逆转' in rules:
        strategies.append('偏好低数值卡牌')
    if '王牌杀手' in rules:
        strategies.append('重点使用含1或A的卡牌')
    if '同类强化' in rules:
        strategies.append('集中使用同类型卡牌以触发强化效果')
    if '同类弱化' in rules:
        strategies.append('避免使用相同类型，采用多样化策略')
    
    if strategies:
        return '对手策略：' + '，'.join(strategies)
    else:
        return '对手采用常规策略，平衡攻防'

def predict_unknown_cards(rules, board, count):
    """预测未知卡牌"""
    predictions = []
    
    # 选拔规则优先处理
    if '选拔' in rules:
        predictions.append({
            'card': '根据星级配额的战略性卡牌',
            'confidence': 0.85,
            'reasoning': '选拔模式下严格按照星级配额进行卡牌选择'
        })
    
    # 基于规则生成预测（支持多规则组合）
    if '同数' in rules and '加算' in rules:
        predictions.append({
            'card': '同数+加算复合连携卡牌',
            'confidence': 0.8,
            'reasoning': '同时适合同数和加算规则的复合战术卡牌'
        })
    elif '同数' in rules:
        predictions.append({
            'card': '重复数值卡牌',
            'confidence': 0.7,
            'reasoning': '同数规则下偏好设置连携陷阱'
        })
    elif '加算' in rules:
        predictions.append({
            'card': '互补数值卡牌',
            'confidence': 0.6,
            'reasoning': '加算规则下偏好和数组合'
        })
    
    if '逆转' in rules and len(predictions) < count:
        predictions.append({
            'card': '低数值卡牌',
            'confidence': 0.8,
            'reasoning': '逆转规则下低数值更有优势'
        })
    
    if '王牌杀手' in rules and len(predictions) < count:
        predictions.append({
            'card': '含1或A的卡牌',
            'confidence': 0.75,
            'reasoning': '王牌杀手规则下1和A具有特殊效果'
        })
    
    # 补充通用预测
    while len(predictions) < count:
        predictions.append({
            'card': '中等星级卡牌',
            'confidence': 0.5,
            'reasoning': '玩家通常使用平衡的卡组配置'
        })
    
    return predictions[:count]

def calculate_win_probability(game_state, move, my_owner):
    """计算胜率"""
    # 当前局面评估
    current_eval = evaluate_current_position(game_state, my_owner)
    
    # 执行移动后的评估（使用make_move/undo_move）
    card, (row, col) = move
    move_record = game_state.make_move(row, col, card)
    if move_record is None:
        return {
            'current': round(sigmoid(current_eval), 3),
            'after_move': round(sigmoid(current_eval), 3),
            'confidence': 0.5
        }
    
    try:
        after_move_eval = evaluate_current_position(game_state, my_owner)
        
        # 转换为胜率（简化计算）
        current_prob = sigmoid(current_eval)
        after_move_prob = sigmoid(after_move_eval)
        
        return {
            'current': round(current_prob, 3),
            'after_move': round(after_move_prob, 3),
            'confidence': min(0.9, abs(after_move_eval) / 10 + 0.6)  # 基于评估差值的置信度
        }
    finally:
        # 撤销移动
        game_state.undo_move(move_record)

def sigmoid(x):
    """将评估值转换为0-1之间的概率"""
    import math
    return 1 / (1 + math.exp(-x / 5))  # 调整缩放因子

def evaluate_current_position(game_state, my_owner):
    """评估当前局面（简化版）"""
    red_count, blue_count = game_state.count_cards()
    
    if my_owner == 'blue':
        basic_score = blue_count - red_count
    else:  # my_owner == 'red'
        basic_score = red_count - blue_count
    
    # 考虑位置因素
    position_bonus = 0
    for r in range(3):
        for c in range(3):
            card = game_state.board.get_card(r, c)
            if card:
                weight = 1.5 if (r, c) in [(0,0), (0,2), (2,0), (2,2)] else 1.2 if (r, c) == (1,1) else 1.0
                if (card.owner == 'blue' and my_owner == 'blue') or (card.owner == 'red' and my_owner == 'red'):
                    position_bonus += weight
                else:
                    position_bonus -= weight
    
    return basic_score * 2 + position_bonus

def generate_move_recommendation(game_state, move, my_owner):
    """生成移动建议"""
    card, (row, col) = move
    
    # 分析移动的价值
    move_reasoning = analyze_move_value(game_state, move, my_owner)
    strategic_value = analyze_strategic_value(game_state, move, my_owner)
    
    # 生成备选方案（简化）
    alternative_moves = []
    current_player = game_state.players[game_state.current_player_idx]
    playable_cards = current_player.get_playable_cards(game_state.rules)
    available_positions = game_state.board.available_positions()
    
    # 评估几个备选移动
    for alt_card in playable_cards[:2]:  # 只检查前2张卡
        for alt_pos in available_positions[:2]:  # 只检查前2个位置
            if alt_card.card_id != card.card_id or alt_pos != (row, col):
                alt_eval = evaluate_alternative_move(game_state, (alt_card, alt_pos), my_owner)
                star_map = get_card_star_map()
                alt_star = star_map.get(alt_card.card_id, '?')
                alternative_moves.append({
                    'card': f"U{alt_card.up} R{alt_card.right} D{alt_card.down} L{alt_card.left} 星级:{alt_star}",
                    'pos': list(alt_pos),
                    'value': round(alt_eval, 3)
                })
    
    # 按价值排序
    alternative_moves.sort(key=lambda x: x['value'], reverse=True)
    
    return {
        'move_reasoning': move_reasoning,
        'strategic_value': strategic_value,
        'alternative_moves': alternative_moves[:3]  # 最多3个备选
    }

def analyze_move_value(game_state, move, my_owner):
    """分析移动价值"""
    card, (row, col) = move
    
    # 检查能否吃掉对手卡牌（包括连携）
    captured_cards = count_captured_cards(game_state, move, my_owner)
    combo_analysis = analyze_combo_potential(game_state, move, my_owner)
    
    if captured_cards > 0:
        if combo_analysis:
            return f"此移动可以吃掉{captured_cards}张对手卡牌（{combo_analysis}）"
        else:
            return f"此移动可以吃掉{captured_cards}张对手卡牌"
    
    # 如果没有直接吃子，检查连携潜力
    if combo_analysis:
        return f"设置连携陷阱：{combo_analysis}"
    
    # 检查位置价值
    if (row, col) in [(0,0), (0,2), (2,0), (2,2)]:
        return "占据角落位置，具有防御优势"
    elif (row, col) == (1,1):
        return "控制中心位置，影响四个方向"
    else:
        return "稳定发展，保持场面控制"

def analyze_combo_potential(game_state, move, my_owner):
    """分析连携潜力和规则组合效果"""
    card, (row, col) = move
    rules = game_state.rules
    board = game_state.board
    analysis = []
    
    directions = [(-1, 0, 'up', 'down'), (1, 0, 'down', 'up'), 
                 (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
    
    # 同数连携分析
    if '同数' in rules:
        same_values = []
        for dr, dc, my_dir, opp_dir in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                neighbor_card = board.get_card(nr, nc)
                if neighbor_card:
                    # 使用原始数值进行同数判断
                    my_value = card.get_base_value(my_dir)
                    neighbor_value = neighbor_card.get_base_value(opp_dir)
                    if my_value == neighbor_value:
                        same_values.append((nr, nc, my_value))
        
        if len(same_values) >= 2:
            analysis.append(f"同数连携：{len(same_values)}个数值{same_values[0][2]}的相邻卡牌")
        elif len(same_values) == 1:
            analysis.append(f"部分同数匹配：与({same_values[0][0]},{same_values[0][1]})数值{same_values[0][2]}相同")
    
    # 加算连携分析
    if '加算' in rules:
        sum_groups = {}
        for dr, dc, my_dir, opp_dir in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                neighbor_card = board.get_card(nr, nc)
                if neighbor_card:
                    # 使用原始数值进行加算判断
                    my_value = card.get_base_value(my_dir)
                    neighbor_value = neighbor_card.get_base_value(opp_dir)
                    sum_val = my_value + neighbor_value
                    if sum_val not in sum_groups:
                        sum_groups[sum_val] = []
                    sum_groups[sum_val].append((nr, nc))
        
        for sum_val, positions in sum_groups.items():
            if len(positions) >= 2:
                analysis.append(f"加算连携：{len(positions)}个和为{sum_val}的相邻卡牌")
            elif len(positions) == 1:
                analysis.append(f"部分加算匹配：与({positions[0][0]},{positions[0][1]})和为{sum_val}")
    
    # 逆转规则分析（使用有效数值，考虑同类修正）
    if '逆转' in rules:
        effective_values = [card.get_effective_value('up', []), card.get_effective_value('right', []), 
                           card.get_effective_value('down', []), card.get_effective_value('left', [])]
        card_avg = sum(effective_values) / 4
        if card_avg <= 5:
            analysis.append("逆转优势：低数值卡牌在逆转规则下更有竞争力")
        elif card_avg >= 8:
            analysis.append("逆转劣势：高数值卡牌在逆转规则下处于劣势")
    
    # 王牌杀手规则分析（使用修正后数值）
    if '王牌杀手' in rules:
        effective_values = [card.get_modified_value('up'), card.get_modified_value('right'), 
                           card.get_modified_value('down'), card.get_modified_value('left')]
        ace_killer_count = sum(1 for v in effective_values if v in [1, 10])
        if ace_killer_count >= 2:
            analysis.append(f"王牌杀手强势：卡牌有{ace_killer_count}个1或A，具有特殊攻击力")
        elif ace_killer_count == 1:
            analysis.append("王牌杀手效果：卡牌包含1或A，部分方向具有特殊效果")
    
    # 同类强化/弱化分析
    if ('同类强化' in rules or '同类弱化' in rules) and card.card_type:
        same_type_count = 0
        for r in range(3):
            for c in range(3):
                board_card = board.get_card(r, c)
                if board_card and board_card.card_type == card.card_type:
                    same_type_count += 1
        
        if '同类强化' in rules and same_type_count > 0:
            analysis.append(f"同类强化：棋盘上已有{same_type_count}张{card.card_type}类型卡牌，将获得数值提升")
        elif '同类弱化' in rules and same_type_count > 0:
            analysis.append(f"同类弱化：棋盘上已有{same_type_count}张{card.card_type}类型卡牌，将受到数值削弱")
    
    # 复合规则效果分析
    combo_effects = analyze_rule_combinations(rules, analysis)
    if combo_effects:
        analysis.extend(combo_effects)
    
    return '；'.join(analysis) if analysis else None

def analyze_rule_combinations(rules, existing_analysis):
    """分析复合规则效果"""
    combo_effects = []
    
    # 同数+加算复合连携
    if '同数' in rules and '加算' in rules:
        has_same = any('同数连携' in a for a in existing_analysis)
        has_addition = any('加算连携' in a for a in existing_analysis)
        if has_same and has_addition:
            combo_effects.append("复合连携：同时触发同数和加算效果")
    
    # 逆转+王牌杀手组合
    if '逆转' in rules and '王牌杀手' in rules:
        combo_effects.append("逆转王牌杀手：1和A在逆转规则下具有双重优势")
    
    # 逆转+连携组合
    if '逆转' in rules and ('同数' in rules or '加算' in rules):
        combo_effects.append("逆转连携：连携触发时使用逆转比较规则")
    
    # 同类规则+连携组合
    if ('同类强化' in rules or '同类弱化' in rules) and ('同数' in rules or '加算' in rules):
        combo_effects.append("同类连携：卡牌数值变化影响连携计算")
    
    # 秩序/混乱规则影响
    if '秩序' in rules and ('同数' in rules or '加算' in rules):
        combo_effects.append("秩序连携：可用卡牌受限，连携策略需要重新评估")
    elif '混乱' in rules and ('同数' in rules or '加算' in rules):
        combo_effects.append("混乱连携：可用卡牌受限，连携策略需要重新评估")
    
    return combo_effects

def analyze_strategic_value(game_state, move, my_owner):
    """分析战略价值（支持多规则组合）"""
    card, (row, col) = move
    rules = game_state.rules
    board = game_state.board
    strategies = []
    
    # 连携类规则策略
    if '同数' in rules:
        if has_same_number_potential(card):
            strategies.append("为同数连携做准备")
    
    if '加算' in rules:
        if has_addition_potential(card):
            strategies.append("为加算连携做准备")
    
    # 逆转规则策略
    if '逆转' in rules:
        card_avg = (card.up + card.right + card.down + card.left) / 4
        if card_avg <= 5:
            strategies.append("逆转优势策略：使用低数值卡牌")
        elif card_avg >= 8:
            strategies.append("逆转风险策略：高数值卡牌需谨慎使用")
    
    # 王牌杀手规则策略
    if '王牌杀手' in rules:
        values = [card.up, card.right, card.down, card.left]
        ace_killer_count = sum(1 for v in values if v in [1, 10])
        if ace_killer_count >= 1:
            strategies.append(f"王牌杀手战术：利用{ace_killer_count}个特殊数值")
    
    # 同类强化/弱化策略
    if ('同类强化' in rules or '同类弱化' in rules) and card.card_type:
        same_type_on_board = sum(1 for r in range(3) for c in range(3) 
                                if board.get_card(r, c) and board.get_card(r, c).card_type == card.card_type)
        
        if '同类强化' in rules:
            if same_type_on_board > 0:
                strategies.append(f"同类强化策略：与{same_type_on_board}张同类卡牌协同")
            else:
                strategies.append("同类强化先锋：率先建立类型优势")
        elif '同类弱化' in rules:
            if same_type_on_board > 0:
                strategies.append(f"同类弱化规避：避免与{same_type_on_board}张同类卡牌聚集")
            else:
                strategies.append("同类弱化安全：首张同类卡牌不受影响")
    
    # 秩序/混乱规则策略
    if '秩序' in rules:
        if card.can_use:
            strategies.append("秩序策略：珍惜可用卡牌机会")
        else:
            strategies.append("秩序限制：当前卡牌不可使用")
    elif '混乱' in rules:
        if card.can_use:
            strategies.append("混乱策略：把握可用卡牌时机")
        else:
            strategies.append("混乱限制：当前卡牌不可使用")
    
    # 位置战略价值
    if (row, col) == (1, 1):
        strategies.append("控制中心战略位置")
    elif (row, col) in [(0,0), (0,2), (2,0), (2,2)]:
        # 增强的边角战略评估
        corner_analysis = analyze_corner_strategy(card, (row, col), board)
        if corner_analysis:
            strategies.append(f"占据防御要塞：{corner_analysis}")
        else:
            strategies.append("占据防御要塞")
    elif (row, col) in [(0,1), (1,0), (1,2), (2,1)]:
        strategies.append("控制边缘要道")
    
    # 多规则组合策略
    combo_strategies = analyze_multi_rule_strategies(rules, card, row, col)
    if combo_strategies:
        strategies.extend(combo_strategies)
    
    if strategies:
        return '；'.join(strategies)
    else:
        return "维持场面平衡，为后续发展铺路"

def analyze_multi_rule_strategies(rules, card, row, col):
    """分析多规则组合策略"""
    combo_strategies = []
    
    # 逆转+连携组合策略
    if '逆转' in rules and ('同数' in rules or '加算' in rules):
        card_avg = (card.up + card.right + card.down + card.left) / 4
        if card_avg <= 5 and (has_same_number_potential(card) or has_addition_potential(card)):
            combo_strategies.append("逆转连携双重优势：低数值+连携潜力")
    
    # 王牌杀手+连携组合策略
    if '王牌杀手' in rules and ('同数' in rules or '加算' in rules):
        values = [card.up, card.right, card.down, card.left]
        has_ace_killer = any(v in [1, 10] for v in values)
        if has_ace_killer and (has_same_number_potential(card) or has_addition_potential(card)):
            combo_strategies.append("王牌杀手连携：特殊数值+连携双重威胁")
    
    # 逆转+王牌杀手组合策略
    if '逆转' in rules and '王牌杀手' in rules:
        values = [card.up, card.right, card.down, card.left]
        has_ace_killer = any(v in [1, 10] for v in values)
        if has_ace_killer:
            combo_strategies.append("逆转王牌杀手：1和A在逆转规则下无敌")
    
    # 同类规则+其他规则组合
    if ('同类强化' in rules or '同类弱化' in rules) and card.card_type:
        if '逆转' in rules:
            combo_strategies.append("同类逆转策略：数值变化影响逆转效果")
        if '王牌杀手' in rules:
            values = [card.up, card.right, card.down, card.left]
            if any(v in [1, 10] for v in values):
                combo_strategies.append("同类王牌杀手：数值变化不影响特殊效果")
    
    # 中心位置的特殊组合价值
    if (row, col) == (1, 1):
        rule_count = len(rules)
        if rule_count >= 3:
            combo_strategies.append(f"多规则中心：{rule_count}条规则在中心位置发挥最大效果")
    
    return combo_strategies

def count_captured_cards(game_state, move, my_owner):
    """计算能吃掉的对手卡牌数量（包括连携效果）"""
    card, (row, col) = move
    
    # 记录原始卡牌归属
    original_owners = {}
    for r in range(3):
        for c in range(3):
            board_card = game_state.board.get_card(r, c)
            if board_card:
                original_owners[(r, c)] = board_card.owner
    
    # 执行移动（会自动处理翻转和同类效果）
    card_copy = card.copy()
    card_copy.owner = 'red' if my_owner == 'red' else 'blue'
    move_record = game_state.make_move(row, col, card_copy)
    if move_record is None:
        return 0
    
    try:
        # 计算被翻转的对手卡牌数量
        captured = 0
        for r in range(3):
            for c in range(3):
                if (r, c) in original_owners:
                    board_card = game_state.board.get_card(r, c)
                    if board_card and original_owners[(r, c)] != card_copy.owner and board_card.owner == card_copy.owner:
                        captured += 1
        
        return captured
    finally:
        # 撤销移动
        game_state.undo_move(move_record)

def has_same_number_potential(card):
    """检查是否有同数潜力"""
    values = [card.get_base_value('up'), card.get_base_value('right'), 
             card.get_base_value('down'), card.get_base_value('left')]
    return len(set(values)) < 4  # 有重复数值

def has_addition_potential(card):
    """检查是否有加算潜力"""
    values = [card.get_base_value('up'), card.get_base_value('right'), 
             card.get_base_value('down'), card.get_base_value('left')]
    # 检查是否有常见的和数组合
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if i != j and v1 + v2 in [8, 10, 12]:
                return True
    return False

def evaluate_alternative_move(game_state, move, my_owner):
    """评估备选移动"""
    card, (row, col) = move
    move_record = game_state.make_move(row, col, card)
    if move_record is None:
        return evaluate_current_position(game_state, my_owner)
    
    try:
        return evaluate_current_position(game_state, my_owner)
    finally:
        game_state.undo_move(move_record)

def format_card_display(card, star):
    """
    格式化卡牌显示信息，包含原始和修正后的数值
    """
    if card.type_modifier != 0:
        # 有同类修正的情况
        modifier_str = f"{card.type_modifier:+d}"
        base_display = f"U{card.base_up} R{card.base_right} D{card.base_down} L{card.base_left}"
        modified_display = f"U{card.get_modified_value('up')} R{card.get_modified_value('right')} D{card.get_modified_value('down')} L{card.get_modified_value('left')}"
        type_name = card.card_type or "无类型"
        return f"{base_display} → {modified_display} ({type_name}{modifier_str}) 星级:{star}"
    else:
        # 无修正的情况
        return f"U{card.base_up} R{card.base_right} D{card.base_down} L{card.base_left} 星级:{star}"

def _print_type_analysis(game_state):
    """
    打印同类强化/弱化的详细分析
    """
    # 统计棋盘上已设置的各类型数量。手牌会受到修正影响，但不增加修正层数。
    board_type_counts = {}
    
    # 棋盘卡牌
    print("棋盘卡牌类型分布：")
    for r in range(3):
        for c in range(3):
            card = game_state.board.get_card(r, c)
            if card and card.card_type:
                board_type_counts[card.card_type] = board_type_counts.get(card.card_type, 0) + 1
                modifier_info = f"(修正{card.type_modifier:+d})" if card.type_modifier != 0 else ""
                print(f"  位置({r},{c}): {card.card_type} {modifier_info}")
    
    # 手牌类型
    print("\n手牌类型分布：")
    for i, player in enumerate(game_state.players):
        player_name = "红方" if i == 0 else "蓝方"
        print(f"  {player_name}手牌：")
        for hand_card in player.hand:
            if hand_card.card_type:
                modifier_info = f"(修正{hand_card.type_modifier:+d})" if hand_card.type_modifier != 0 else ""
                is_unknown = getattr(hand_card, '_is_prediction', False) or \
                           (hand_card.up == 0 and hand_card.right == 0 and hand_card.down == 0 and hand_card.left == 0)
                unknown_info = " [推测]" if is_unknown else " [已知]"
                print(f"    {hand_card.card_type}{modifier_info}{unknown_info}")
    
    # 总计
    print(f"\n类型总计：")
    for card_type, count in board_type_counts.items():
        rule_type = "强化" if '同类强化' in game_state.rules else "弱化"
        modifier = count if '同类强化' in game_state.rules else -count
        print(f"  {card_type}: 场上{count}张 → {rule_type}{modifier:+d}")

def analyze_corner_strategy(card, position, board):
    """
    分析边角放置战略价值
    针对高数值边角落放置的特殊评估
    """
    row, col = position
    values = [card.up, card.right, card.down, card.left]  # U, R, D, L
    analysis_parts = []
    
    # 识别高数值边
    high_values = [v for v in values if v >= 8]
    very_high_values = [v for v in values if v >= 9]
    ace_values = [v for v in values if v == 10]  # A值
    low_values = [v for v in values if v <= 4]
    
    # 特殊AA组合检测
    ace_positions = [i for i, v in enumerate(values) if v == 10]
    if len(ace_positions) >= 2:
        # 检查是否是最优AA组合
        if (row == 2 and col == 2 and 1 in ace_positions and 3 in ace_positions):  # 右下角: R+L=AA
            analysis_parts.append("最优AA右下角组合")
        elif (row == 2 and col == 0 and 1 in ace_positions and 0 in ace_positions):  # 左下角: R+U=AA  
            analysis_parts.append("次优AA左下角组合")
        elif (row == 0 and col == 2 and 2 in ace_positions and 3 in ace_positions):  # 右上角: D+L=AA
            analysis_parts.append("优质AA右上角组合")
        elif (row == 0 and col == 0 and 1 in ace_positions and 2 in ace_positions):  # 左上角: R+D=AA
            analysis_parts.append("平衡AA左上角组合")
        elif len(ace_positions) >= 2:
            analysis_parts.append(f"{len(ace_positions)}边AA优势组合")
    
    # 三边高数值组合分析 (如右下左为9或8的卡牌)
    if len(high_values) >= 3:
        if len(very_high_values) >= 3:
            analysis_parts.append("三边9+超强角落控制")
        else:
            analysis_parts.append("三边8+强力角落控制")
        
        # 检查弱势边保护
        if len(low_values) >= 1:
            weak_sides = [['上', '右', '下', '左'][i] for i, v in enumerate(values) if v <= 4]
            analysis_parts.append(f"保护{'/'.join(weak_sides)}弱势边")
    
    # 双边高数值分析
    elif len(high_values) >= 2:
        high_positions = [i for i, v in enumerate(values) if v >= 8]
        
        # 检查高数值边的相邻性
        if are_adjacent_positions(high_positions):
            side_names = get_side_names(high_positions)
            analysis_parts.append(f"相邻{'/'.join(side_names)}高数值控制")
        else:
            analysis_parts.append("双边高数值防护")
    
    # 分析当前位置的边暴露情况
    exposed_sides = get_exposed_sides(row, col)
    if exposed_sides:
        exposed_values = [values[i] for i in exposed_sides]
        if any(v <= 3 for v in exposed_values):
            # 有弱势边暴露的警告
            weak_exposed = [['上', '右', '下', '左'][i] for i in exposed_sides if values[i] <= 3]
            analysis_parts.append(f"警告：{'/'.join(weak_exposed)}边存在弱点")
        elif all(v >= 8 for v in exposed_values):
            # 所有暴露边都是高值
            strong_exposed = [['上', '右', '下', '左'][i] for i in exposed_sides]
            analysis_parts.append(f"{'/'.join(strong_exposed)}边强力防护")
    
    # 特殊情况：极端数值差异卡牌的角落适配性
    min_val, max_val = min(values), max(values)
    if max_val - min_val >= 7:  # 如1,9,9,9这样的卡牌
        if len(low_values) == 1:
            weak_side = ['上', '右', '下', '左'][values.index(min_val)]
            analysis_parts.append(f"隐藏{weak_side}边弱点(值{min_val})")
    
    # 分析相邻已放置卡牌的协同效果
    adjacent_synergy = analyze_adjacent_synergy(card, row, col, board)
    if adjacent_synergy:
        analysis_parts.append(adjacent_synergy)
    
    return '，'.join(analysis_parts) if analysis_parts else None

def are_adjacent_positions(positions):
    """检查位置列表中是否有相邻的位置"""
    if len(positions) < 2:
        return False
    # 边的相邻关系: 0(上)-1(右), 1(右)-2(下), 2(下)-3(左), 3(左)-0(上)
    adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            if pos2 in adjacency[pos1]:
                return True
    return False

def get_side_names(positions):
    """获取位置对应的边名称"""
    side_map = {0: '上', 1: '右', 2: '下', 3: '左'}
    return [side_map[pos] for pos in positions]

def get_exposed_sides(row, col):
    """获取在该位置会暴露的边（不靠墙的边）"""
    exposed = []
    
    # 检查每条边是否暴露
    if row > 0:  # 上边暴露
        exposed.append(0)
    if col < 2:  # 右边暴露  
        exposed.append(1)
    if row < 2:  # 下边暴露
        exposed.append(2)
    if col > 0:  # 左边暴露
        exposed.append(3)
        
    return exposed

def analyze_adjacent_synergy(card, row, col, board):
    """分析与相邻卡牌的协同效果"""
    synergy_effects = []
    values = [card.up, card.right, card.down, card.left]
    
    # 检查四个方向的相邻卡牌
    directions = [(-1, 0, 0, 2), (0, 1, 1, 3), (1, 0, 2, 0), (0, -1, 3, 1)]  # (dr, dc, my_side, adj_side)
    
    for dr, dc, my_side, adj_side in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            adj_card = board.get_card(nr, nc)
            if adj_card:
                my_value = values[my_side]
                adj_value = getattr(adj_card, ['up', 'right', 'down', 'left'][adj_side])
                
                # 检查数值匹配度
                if my_value == adj_value and my_value >= 8:
                    side_name = ['上', '右', '下', '左'][my_side]
                    synergy_effects.append(f"{side_name}边与邻牌高值匹配({my_value})")
                elif my_value + adj_value == 10 and min(my_value, adj_value) >= 4:
                    side_name = ['上', '右', '下', '左'][my_side]
                    synergy_effects.append(f"{side_name}边与邻牌互补({my_value}+{adj_value}=10)")
    
    return '，'.join(synergy_effects) if synergy_effects else None

app = Flask(__name__)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    try:
        data = request.get_json()
        if not data or 'board' not in data or 'myHand' not in data or 'oppHand' not in data or 'myOwner' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        used_cards = set()
        board = parse_board(data['board'])
        print("收到客户端消息的棋盘：")
        print(board)
        my_owner = parse_owner(data['myOwner'])
        if my_owner == 'red':
            opp_owner = 'blue'
        else:
            opp_owner = 'red'
        print(data['oppHand'])
        
        # 先解析规则，然后用于智能手牌处理
        rules, open_mode = parse_rules_and_open_mode(data.get('rules', ''))
        
        # 选拔规则特殊处理：需要全局统计星级使用情况
        if '选拔' in rules:
            print("检测到选拔规则，启动全局星级约束分析")
            # 先统计棋盘和已知手牌的星级使用情况
            global_star_usage = analyze_global_star_usage(board, data['myHand'], data['oppHand'])
            print(f"全局星级使用情况: {global_star_usage}")
        
        # 使用智能手牌解析，对对手手牌启用行为建模
        # 蒙特卡洛模式下跳过对手手牌采样（求解器内部自行处理未知卡牌）
        solver_type = data.get('solver', 'minimax')
        mc_skip_sampling = (solver_type == 'monte_carlo')
        my_hand = parse_hand(data['myHand'], my_owner, used_cards, rules, board, is_opponent=False)
        opp_hand = parse_hand(data['oppHand'], opp_owner, used_cards, rules, board,
                              is_opponent=True, skip_sampling=mc_skip_sampling)
        # 玩家顺序：遵循GameState约定 - players[0]=红方, players[1]=蓝方
        # currentPlayer: 1=蓝方回合, 2=红方回合, 0=未知(兼容旧客户端)
        from core.player import Player
        current_player = data.get('currentPlayer', 0)

        # 按照红蓝方约定创建玩家列表
        if my_owner == 'red':
            # 我是红方(players[0]), 对手是蓝方(players[1])
            players = [Player('me', my_hand), Player('opp', opp_hand)]
        else:  # my_owner == 'blue'
            # 对手是红方(players[0]), 我是蓝方(players[1])
            players = [Player('opp', opp_hand), Player('me', my_hand)]

        # 确定当前回合玩家
        if current_player == 2:  # 红方回合
            current_player_idx = 0
        elif current_player == 1:  # 蓝方回合
            current_player_idx = 1
        else:
            # 兼容旧客户端：从棋盘卡牌数量推断回合
            red_count = 0
            blue_count = 0
            for r in range(3):
                for c in range(3):
                    board_card = board.get_card(r, c)
                    if board_card:
                        if board_card.owner == 'red':
                            red_count += 1
                        elif board_card.owner == 'blue':
                            blue_count += 1
            if red_count < blue_count:
                current_player_idx = 0  # 红方落后, 红方回合
            elif blue_count < red_count:
                current_player_idx = 1  # 蓝方落后, 蓝方回合
            else:
                current_player_idx = 0  # 平局, 默认红方(先手)回合

        is_my_turn = (my_owner == 'red' and current_player_idx == 0) or (my_owner == 'blue' and current_player_idx == 1)
        print(f"[Turn] my_owner={my_owner}, currentPlayer={current_player}, current_player_idx={current_player_idx}, is_my_turn={is_my_turn}")
        game_state = GameState(board, players, current_player_idx=current_player_idx, rules=rules)
        
        # 如果有同类强化/弱化规则，立即处理
        if '同类强化' in rules or '同类弱化' in rules:
            game_state.recalculate_type_modifiers()
            print(f"应用同类规则后的类型分析：")
            _print_type_analysis(game_state)
        
        # 检查是否请求详细搜索进度 (默认关闭以提升性能)
        show_search_progress = data.get('show_search_progress', False)
        search_progress_data = []
        console_reporter = ConsoleSearchReporter(interval=0.5)
        opp_unknown_count = _count_unknown_slots_from_hand(opp_hand)
        use_endgame_robust = solver_type == 'minimax' and _should_use_endgame_robust_mode(board, opp_unknown_count)
        
        def progress_callback(progress_info):
            """轻量级搜索进度回调函数"""
            nonlocal search_progress_data
            console_reporter.on_minimax_progress(progress_info)
            # 只在需要时才做复杂的数据处理
            if show_search_progress:
                search_progress_data.append({
                    'depth': progress_info['depth'],
                    'max_depth': progress_info['max_depth'],
                    'best_move': format_move_for_display(progress_info['best_move']),
                    'best_score': progress_info['best_score'],
                    'nodes_searched': progress_info['nodes_searched'],
                    'time_elapsed': round(progress_info['time_elapsed'], 3),
                    'time_remaining': round(progress_info['time_remaining'], 3),
                    'nodes_per_second': round(progress_info['stats']['nodes_per_second'], 0),
                    'tt_hit_rate': round(progress_info['stats']['tt_hit_rate'] * 100, 1),
                    'cutoff_rate': round(progress_info['stats']['cutoff_rate'] * 100, 1),
                    'branching_factor': round(progress_info['stats']['avg_branching_factor'], 2)
                })
                
                # 输出详细进度（仅在明确请求时）
                print(f"搜索进度更新 - 深度 {progress_info['depth']}: "
                      f"评分={progress_info['best_score']:.3f}, "
                      f"节点={progress_info['nodes_searched']:,}, "
                      f"时间={progress_info['time_elapsed']:.2f}秒")
        
        # 选择求解器（solver_type 已在上面读取）
        mc_simulations = data.get('mc_simulations', 150)  # 蒙特卡洛模拟次数

        if solver_type == 'monte_carlo':
            print(f"[Solver] 使用蒙特卡洛求解器 (simulations={mc_simulations})")
            move, _ = monte_carlo_best_move(
                game_state,
                all_cards=get_all_cards(),
                my_owner=my_owner,
                time_limit=8,
                base_simulations=mc_simulations,
                verbose=True
            )
        else:
            print("[Solver] 使用 Minimax 求解器")
            move, _ = find_best_move_parallel(
                game_state,
                max_depth=10,
                verbose=False,
                all_cards=get_all_cards(),
                open_mode=open_mode,
                max_time=10,
                progress_callback=progress_callback
            )

            if use_endgame_robust:
                opponent_player_idx = 0 if my_owner == 'blue' else 1
                scenario_sample_count = 1 if opp_unknown_count == 0 else min(24, max(16, opp_unknown_count * 12))
                scenario_states = _build_endgame_scenarios(
                    game_state,
                    opp_hand,
                    used_cards,
                    rules,
                    board,
                    opp_owner,
                    opponent_player_idx,
                    scenario_sample_count
                )
                robust_move, robust_candidates = select_endgame_robust_move(
                    game_state,
                    scenario_states,
                    game_state.current_player_idx,
                    progress_reporter=console_reporter
                )
                robust_lookup = {_move_key(item['move']): item for item in robust_candidates}

                if move is not None and robust_move is not None:
                    standard_item = robust_lookup.get(_move_key(move))
                    robust_item = robust_lookup.get(_move_key(robust_move))
                    if standard_item and robust_item:
                        should_override = (
                            standard_item['safety_ratio'] < 0.5 and robust_item['safety_ratio'] >= 0.5
                        ) or (
                            robust_item['safety_ratio'] > standard_item['safety_ratio'] and
                            robust_item['final_score'] >= standard_item['final_score'] - 1.0
                        ) or (
                            robust_item['final_score'] > standard_item['final_score'] + 0.05
                        )
                        if should_override and _move_key(robust_move) != _move_key(move):
                            print(
                                f"[Solver] 信息集残局覆盖: 标准={standard_item['final_score']:.3f}/"
                                f"{standard_item['safety_ratio']:.2f}, "
                                f"信息集={robust_item['final_score']:.3f}/{robust_item['safety_ratio']:.2f}, "
                                f"场景={len(scenario_states)}"
                            )
                            move = robust_move

        if move is None:
            return jsonify({'move': None, 'msg': '无可用动作'})
            
        card, (row, col) = move
        
        # 分析对手手牌
        opponent_analysis = analyze_opponent_hand(opp_hand, rules, board)
        
        # 调试信息：输出对手手牌分析结果
        print(f"对手手牌分析: 总计{len(opp_hand)}张卡牌, 其中{opponent_analysis.get('total_unknown', 0)}张未知")
        
        # 计算胜率
        win_prob = calculate_win_probability(game_state, move, my_owner)
        
        # 生成AI建议
        recommendation = generate_move_recommendation(game_state, move, my_owner)
        
        # 打印AI给出结果后的棋盘
        new_state = game_state.copy()
        new_state.play_move(row, col, card)
        print("AI给出结果后的棋盘：")
        print(new_state.board)
        
        # 性能统计
        from ai.ai import SEARCH_STATS
        performance_stats = {
            'nodes_searched': SEARCH_STATS.nodes_searched,
            'search_depth': SEARCH_STATS.depth_completed,
            'tt_hit_rate': SEARCH_STATS.tt_hits / max(SEARCH_STATS.nodes_searched, 1) * 100,
            'cutoff_rate': SEARCH_STATS.alpha_beta_cutoffs / max(SEARCH_STATS.nodes_searched, 1) * 100,
            'unknown_cards_processed': opponent_analysis.get('total_unknown', 0),
            'performance_optimizations_active': True  # 标记优化已激活
        }
        
        print(f"性能统计:")
        print(f"  搜索节点: {performance_stats['nodes_searched']:,}")
        print(f"  搜索深度: {performance_stats['search_depth']}")
        print(f"  置换表命中率: {performance_stats['tt_hit_rate']:.1f}%")
        print(f"  α-β剪枝率: {performance_stats['cutoff_rate']:.1f}%")
        print(f"  未知卡牌处理: {performance_stats['unknown_cards_processed']} 张")
        
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, '?')
        
        # 生成卡牌显示信息（包含原始和修正后的数值）
        card_display = format_card_display(card, star)
        
        # 准备返回结果
        result = {
            'card': card_display,
            'card_id': card.card_id,
            'pos': [row, col],
            'opponent_hand_analysis': opponent_analysis,
            'win_probability': win_prob,
            'recommendation': recommendation,
            'performance_stats': performance_stats  # 添加性能统计
        }
        
        # 如果请求了搜索进度，添加到结果中
        if show_search_progress and search_progress_data:
            result['search_progress'] = search_progress_data
            result['search_summary'] = generate_search_summary(search_progress_data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_move_for_display(move):
    """格式化移动用于显示"""
    if move is None:
        return "无移动"
    
    card, (row, col) = move
    star_map = get_card_star_map()
    star = star_map.get(card.card_id, '?')
    return f"U{card.up}R{card.right}D{card.down}L{card.left}(★{star}) → ({row},{col})"

def generate_search_summary(progress_data):
    """生成搜索摘要"""
    if not progress_data:
        return "无搜索数据"
    
    final_data = progress_data[-1]
    max_depth = final_data['depth']
    total_nodes = final_data['nodes_searched']
    total_time = final_data['time_elapsed']
    final_score = final_data['best_score']
    
    # 计算评分变化趋势
    score_trend = "稳定"
    if len(progress_data) >= 2:
        score_changes = []
        for i in range(1, len(progress_data)):
            change = progress_data[i]['best_score'] - progress_data[i-1]['best_score']
            score_changes.append(change)
        
        avg_change = sum(score_changes) / len(score_changes)
        if avg_change > 0.1:
            score_trend = "上升"
        elif avg_change < -0.1:
            score_trend = "下降"
    
    # 计算搜索效率
    efficiency = "高效" if final_data['nodes_per_second'] > 1000 else "正常" if final_data['nodes_per_second'] > 500 else "较慢"
    
    return {
        'max_depth_reached': max_depth,
        'total_nodes_searched': total_nodes,
        'total_time_seconds': total_time,
        'final_evaluation_score': final_score,
        'score_trend': score_trend,
        'search_efficiency': efficiency,
        'average_branching_factor': final_data['branching_factor'],
        'transposition_table_hit_rate': f"{final_data['tt_hit_rate']}%",
        'alpha_beta_cutoff_rate': f"{final_data['cutoff_rate']}%"
    }

@app.route('/search_progress', methods=['POST'])
def get_search_progress():
    """获取搜索进度的专用端点"""
    try:
        data = request.get_json()
        # 这里可以实现实时搜索进度查询
        # 目前返回简单的状态信息
        return jsonify({
            'status': 'searching',
            'message': '搜索进行中，请等待完成'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    import os
    import sys
    
    
    # 清除之前的输出并显示字符画
    os.system('title TTC_Siren - Triple Triad AI Server')
    os.system('cls' if os.name == 'nt' else 'clear')
    print(render(0))
    
    # 抑制Flask的启动信息
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # 启动应用，使用threaded=True减少警告
    try:
        print(" * Running on http://127.0.0.1:5000")
        print("Press CTRL+C to quit")
        app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
