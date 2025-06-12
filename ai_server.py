from flask import Flask, request, jsonify
import pandas as pd
from core.card import Card
from core.player import Player
from core.board import Board
from core.game_state import GameState
from ai.ai import find_best_move_parallel
from ai.unknown_card_handler import initialize_unknown_card_handler, get_unknown_card_handler
import os
import codecs
import threading
import random

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

def parse_hand(hand_json, owner, used_cards, rules=None, board_state=None, is_opponent=False, id_offset=1000):
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
    """
    hand = []
    known_cards = []
    unknown_count = 0
    type_map = get_card_type_map()
    
    # 第一遍：处理已知卡牌，统计未知卡牌数量
    for idx, item in enumerate(hand_json):
        if all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
            # 未知手牌
            unknown_count += 1
        else:
            # 已知手牌
            up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
            card_id = find_card_id_by_stats(up, right, down, left)
            if card_id is None:
                raise ValueError(f"Hand card not found in database: U{up} R{right} D{down} L{left}")
            card_type = type_map.get(card_id)
            c = Card(up, right, down, left, owner, card_id, card_type, 
                    item.get('canUse', True))  # 支持canUse参数
            hand.append(c)
            known_cards.append(c)
            used_cards.add(card_id)
    
    # 第二遍：智能处理未知卡牌
    if unknown_count > 0:
        card_type = "opponent" if is_opponent else "own"
        print(f"Processing {unknown_count} unknown {card_type} cards for {owner}")
        ensure_handler_initialized()
        handler = get_unknown_card_handler()
        
        if handler and rules:
            # 使用智能采样，对对手启用行为建模
            if is_opponent:
                # 对手手牌使用行为建模采样
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
                # 己方手牌使用常规采样
                unknown_cards = handler.generate_unknown_cards(
                    count=unknown_count,
                    rules=rules,
                    used_cards=used_cards.copy(),
                    board_state=board_state,
                    known_hand=known_cards,
                    owner=owner,
                    can_use=True
                )
            
            # 设置正确的canUse值
            unknown_idx = 0
            for idx, item in enumerate(hand_json):
                if all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
                    if unknown_idx < len(unknown_cards):
                        unknown_cards[unknown_idx].can_use = item.get('canUse', True)
                        unknown_idx += 1
            
            hand.extend(unknown_cards)
            print(f"Generated {len(unknown_cards)} {card_type} cards, hand now has {len(hand)} total cards")
        else:
            # 回退到简化的处理方式
            print("Using fallback sampling for unknown cards")
            all_cards = get_all_cards()
            sample_size = min(unknown_count * 5, len(all_cards))  # 限制采样数量
            sampled_cards = random.sample(all_cards, sample_size)
            
            for idx, item in enumerate(hand_json):
                if all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
                    for card in sampled_cards:
                        c = Card(card.up, card.right, card.down, card.left, owner, 
                               card.card_id, card.card_type, item.get('canUse', True))
                        hand.append(c)
            print(f"Fallback generated cards, hand now has {len(hand)} total cards")
    
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
    
    # 如果还有未处理的未知卡牌，生成通用预测
    remaining_unknown = unknown_count - len(predicted_cards)
    if remaining_unknown > 0:
        generic_predictions = predict_unknown_cards(rules, board, remaining_unknown)
        predicted_cards.extend(generic_predictions)
    
    # 合并所有卡牌信息
    all_cards = known_cards + predicted_cards
    
    return {
        'predicted_cards': all_cards[:10],  # 最多显示10张
        'total_unknown': unknown_count,
        'strategy_analysis': strategy_analysis
    }

def get_prediction_confidence(card, rules):
    """根据规则和卡牌特征计算预测置信度（支持多规则组合）"""
    confidence = 0.6  # 基础置信度
    rule_bonuses = []  # 记录各规则的置信度加成
    
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

def analyze_opponent_strategy(rules, board, known_cards):
    """分析对手策略"""
    strategies = []
    
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
    
    if my_owner == 1:  # 蓝方
        basic_score = blue_count - red_count
    else:  # 红方
        basic_score = red_count - blue_count
    
    # 考虑位置因素
    position_bonus = 0
    for r in range(3):
        for c in range(3):
            card = game_state.board.get_card(r, c)
            if card:
                weight = 1.5 if (r, c) in [(0,0), (0,2), (2,0), (2,2)] else 1.2 if (r, c) == (1,1) else 1.0
                if (card.owner == 'blue' and my_owner == 1) or (card.owner == 'red' and my_owner == 2):
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
    card_copy.owner = 'red' if my_owner == 2 else 'blue'
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
    # 统计各类型数量
    type_counts = {}
    
    # 棋盘卡牌
    print("棋盘卡牌类型分布：")
    for r in range(3):
        for c in range(3):
            card = game_state.board.get_card(r, c)
            if card and card.card_type:
                type_counts[card.card_type] = type_counts.get(card.card_type, 0) + 1
                modifier_info = f"(修正{card.type_modifier:+d})" if card.type_modifier != 0 else ""
                print(f"  位置({r},{c}): {card.card_type} {modifier_info}")
    
    # 手牌类型
    print("\n手牌类型分布：")
    for i, player in enumerate(game_state.players):
        player_name = "红方" if i == 0 else "蓝方"
        print(f"  {player_name}手牌：")
        for hand_card in player.hand:
            if hand_card.card_type:
                type_counts[hand_card.card_type] = type_counts.get(hand_card.card_type, 0) + 1
                modifier_info = f"(修正{hand_card.type_modifier:+d})" if hand_card.type_modifier != 0 else ""
                is_unknown = getattr(hand_card, '_is_prediction', False) or \
                           (hand_card.up == 0 and hand_card.right == 0 and hand_card.down == 0 and hand_card.left == 0)
                unknown_info = " [推测]" if is_unknown else " [已知]"
                print(f"    {hand_card.card_type}{modifier_info}{unknown_info}")
    
    # 总计
    print(f"\n类型总计：")
    for card_type, count in type_counts.items():
        rule_type = "强化" if '同类强化' in game_state.rules else "弱化"
        modifier = count - 1 if '同类强化' in game_state.rules else -(count - 1)
        print(f"  {card_type}: {count}张 → {rule_type}{modifier:+d}")

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
        
        # 使用智能手牌解析，对对手手牌启用行为建模
        my_hand = parse_hand(data['myHand'], my_owner, used_cards, rules, board, is_opponent=False)
        opp_hand = parse_hand(data['oppHand'], opp_owner, used_cards, rules, board, is_opponent=True)
        # 玩家顺序
        from core.player import Player
        if my_owner == 'red':
            players = [Player('me', my_hand), Player('opp', opp_hand)]
            current_player_idx = 0
        else:
            players = [Player('opp', opp_hand), Player('me', my_hand)]
            current_player_idx = 1
        game_state = GameState(board, players, current_player_idx=current_player_idx, rules=rules)
        
        # 如果有同类强化/弱化规则，立即处理
        if '同类强化' in rules or '同类弱化' in rules:
            game_state.recalculate_type_modifiers()
            print(f"应用同类规则后的类型分析：")
            _print_type_analysis(game_state)
        
        # 检查是否请求详细搜索进度 (默认关闭以提升性能)
        show_search_progress = data.get('show_search_progress', False)
        search_progress_data = []
        
        def progress_callback(progress_info):
            """轻量级搜索进度回调函数"""
            nonlocal search_progress_data
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
        
        move, _ = find_best_move_parallel(
            game_state,
            max_depth=10,  # 降低到8层以提升性能
            verbose=False,  # 关闭详细输出以提升性能
            all_cards=get_all_cards(),
            open_mode=open_mode,
            max_time=10,  # 降低到5秒以提升响应速度
            progress_callback=progress_callback if show_search_progress else None
        )

        # MCTS
        #ai_hand = my_hand if my_owner == 'red' else opp_hand
        #move, best_path, _, main_root = mcts_best_move(
        #    game_state,
        #    all_cards=get_all_cards(),
        #    my_hand=ai_hand,
        #    n_simulations=None,
        #    max_seconds=5,
        #    progress_callback=None,
        #    parallel=12
        #)

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

if __name__ == '__main__':
    import os
    import sys
    
    # 定义字符画 - 你可以在这里替换为你准备好的字符画
    # ASCII_ART = """TTC_Siren"""
    # 例如，你可以替换为类似这样的字符画：
    ASCII_ART = '''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⠁⠈⣿⠾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⣿⠃⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⡿⣧⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⣿⣻⣿⣼⠤⣤⠼⠵⣿⠟⠻⢿⢾⡿⠧⢽⣿⣿⣿⣿⡇⠙⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢿⣿⡟⣿⡏⢓⣟⣶⣶⣾⠗⠀⠀⠀⠄⠺⢿⣶⣶⡞⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣌⣇⠀⠈⠉⠉⠀⠀⢀⠠⠀⢀⠀⠈⠉⠉⠀⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠰⠻⣿⣿⣿⣄⠀⠀⠐⠀⠀⠂⠀⠠⠀⢀⠠⠀⠀⢠⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡿⢿⡿⢧⡀⠄⠀⠁⢀⣙⣀⡁⠀⠀⢀⡼⠛⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠁⢰⣿⣶⣦⣄⣀⠀⠀⡀⣠⣼⣾⣷⠄⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠴⣾⣿⣿⣿⣿⣿⣶⣯⣷⣿⣿⣿⣿⣿⡗⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⣫⠦⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡷⣿⣿⡣⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⡠⠒⠉⠀⠀⢹⡜⡥⣚⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡙⣿⣿⠏⠀⠉⠢⢄⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⡸⠤⠤⢤⣀⣀⡸⢎⡵⣘⠮⣝⣿⣿⣿⣿⣿⣿⣿⣿⡿⢥⢛⣼⡏⠀⢀⣠⠴⠊⠙⢄⠀⠀⠀⠀
⠀⠀⢀⠎⠀⠰⣆⠀⠀⠈⠉⠚⡦⢷⢾⣔⣭⣿⣿⣿⣿⣿⣿⣟⣬⠷⢮⢾⠓⠊⠉⠀⠀⢠⠆⠘⡆⠀⠀⠀
⠀⠀⡎⠀⠀⡀⠘⢆⠀⠀⠀⠄⠻⣜⠲⣎⢼⣿⣿⣿⣿⣿⣿⣾⣥⢛⡜⡾⠀⢀⠐⠀⠀⡞⠀⠀⢷⠀⠀⠀
⠀⠀⡇⠀⠠⠀⠀⠈⢷⡀⠀⢀⠀⠹⡗⡜⣎⠶⣹⣿⣿⣿⣿⢎⡱⣋⡼⠃⠀⢀⠀⠀⣰⠃⢀⠠⢸⡄⠀⠀
⠀⠀⡇⠀⠀⠄⠂⠀⠀⢳⠀⠀⠀⠀⠘⢷⡘⢮⣽⣿⣿⣿⣿⣌⢧⡝⠁⠀⠀⠄⠀⢠⠏⠀⠀⡀⠀⡇⠀⠀
⠀⠀⡇⠀⠂⡀⠐⢀⠀⠘⡆⠀⠁⠀⠄⠀⠙⠛⠻⣿⣿⣿⠋⠛⠋⠀⠀⠈⠀⢀⠀⡞⠀⠀⠠⠀⠀⢳⠀⠀
⠀⠀⡇⠀⠠⠀⢠⡀⠀⠀⢻⠀⠀⠁⡀⠌⠀⠀⠀⢿⣿⡟⠀⠀⠠⠀⠐⠀⠁⠀⡼⠁⠀⠠⠁⠀⠀⣹⠀⠀
⠀⢠⠇⠀⠠⠀⠀⢣⡐⠀⠘⡇⠀⠐⠀⠀⠄⠀⠾⢿⣿⡇⢠⣤⠀⠀⠂⠀⡀⣸⠁⠀⠀⠂⠀⠁⠀⡜⠀⠀
⠀⢸⠀⠀⠐⢀⠀⠈⣧⠀⠀⢻⡀⠀⠐⠀⠠⢀⠀⢸⣿⡇⠈⠁⠀⢀⠂⠀⢠⠏⠀⠠⠐⠀⠈⢠⠆⣛⠀⠀
⠀⢸⠀⠈⢀⠀⠠⠀⠸⣆⠀⠈⣧⠐⠀⢈⠀⠀⡀⢼⣿⡇⣠⣀⠀⢀⠠⠀⡟⠀⠀⠠⠀⠠⢠⠎⡗⢸⠀⠀
⠀⣼⠀⢁⠂⡀⠂⠠⠀⢻⡄⠀⠸⡇⠀⠀⠀⠻⠟⣿⣿⣿⠈⠋⠀⠀⡀⣼⢡⡇⠀⠀⠀⢤⠏⢸⠃⢸⡀⠀
⠀⣯⠀⢂⠲⣄⠂⡁⠂⢈⣧⠀⢀⢻⡀⠂⢁⠠⢸⣿⣿⣿⣷⡀⣠⣁⠀⢻⣼⠀⠠⠀⣬⠏⢠⠏⠀⡈⡇⠀ ████████ ████████  ██████     ███████ ██ ██████  ███████ ███    ██ 
⠀⡇⡐⢀⠂⠌⢷⠀⠡⢀⠘⣇⠠⢸⡇⠈⠄⠠⣹⣿⣿⣿⣿⣷⣉⠡⠀⢢⠏⠀⢠⡼⠃⢀⡞⢀⠐⡀⢷⠀    ██       ██    ██          ██      ██ ██   ██ ██      ████   ██ 
⠠⡗⡈⠷⣈⠰⢈⠻⣆⠂⠌⠹⣆⢼⢋⠐⡘⢛⣿⣿⣿⣿⣿⣿⣿⠠⠁⣾⠀⣼⠋⠄⡐⣸⠃⠄⠂⠌⢸⠀    ██       ██    ██          ███████ ██ ██████  █████   ██ ██  ██ 
⠀⣷⡈⡔⠩⠳⣦⢈⠹⢿⣬⡐⠙⣾⢀⠣⠠⢭⣿⣿⣿⣿⣿⣿⣿⠻⢃⣧⠟⢡⠈⡐⣰⠏⡐⣈⠐⡨⣼⡀    ██       ██    ██               ██ ██ ██   ██ ██      ██  ██ ██ 
⠀⠸⡇⠬⣁⠣⡐⠢⢌⠢⡙⢿⣔⡹⣇⠴⠿⣿⣿⣿⣿⣿⣿⣿⣿⢡⣿⠁⢎⠠⡑⢠⡯⠐⡰⠀⠎⣡⣿⡇    ██       ██     ██████     ███████ ██ ██   ██ ███████ ██   ████ 
A Final Fantasy 14 Triple Triad Solver, Using MinMax algorithm and Alpha-Beta pruning. 
If you are willing to provide assistance for my project, welcome!
https://github.com/extrant/TTC_Siren
     '''
    
    # 清除之前的输出并显示字符画
    os.system('title TTC_Siren - Triple Triad AI Server')
    os.system('cls' if os.name == 'nt' else 'clear')
    print(ASCII_ART)
    
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