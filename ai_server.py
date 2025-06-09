from flask import Flask, request, jsonify
import pandas as pd
from card import Card
from player import Player
from board import Board
from game_state import GameState
from ai import find_best_move_parallel
from unknown_card_handler import initialize_unknown_card_handler, get_unknown_card_handler
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
                _card_db = pd.read_csv('幻卡数据库.csv')
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
        if card.up == 0 and card.right == 0 and card.down == 0 and card.left == 0:
            unknown_count += 1
        else:
            # 检查是否是智能采样生成的卡牌（有具体数值但可能是预测的）
            star = star_map.get(card.card_id, '?')
            confidence = 1.0  # 默认已知卡牌
            reasoning = '已知卡牌'
            
            # 如果卡牌有card_id但数值不为0，可能是智能采样的结果
            if hasattr(card, 'card_id') and card.card_id and card.card_id >= 1000:
                # 这是生成的预测卡牌
                confidence = get_prediction_confidence(card, rules)
                reasoning = get_prediction_reasoning(card, rules)
            
            card_info = {
                'card': f"U{card.up} R{card.right} D{card.down} L{card.left}",
                'star': star,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
            if confidence < 1.0:
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
    """根据规则和卡牌特征计算预测置信度"""
    if '同数' in rules and has_same_number_potential(card):
        return 0.8
    elif '加算' in rules and has_addition_potential(card):
        return 0.7
    elif '逆转' in rules:
        avg_value = (card.up + card.right + card.down + card.left) / 4
        if avg_value <= 5:  # 低数值卡牌在逆转规则下更有用
            return 0.8
        else:
            return 0.6
    elif '王牌杀手' in rules:
        values = [card.up, card.right, card.down, card.left]
        if 1 in values or 10 in values:  # 包含1或A(10)
            return 0.75
        else:
            return 0.5
    else:
        return 0.6  # 通用预测置信度

def get_prediction_reasoning(card, rules):
    """根据规则和卡牌特征生成预测原因"""
    if '同数' in rules and has_same_number_potential(card):
        return '同数规则下，此卡可能用于设置连携陷阱'
    elif '加算' in rules and has_addition_potential(card):
        return '加算规则下，此卡适合数值组合连携'
    elif '逆转' in rules:
        avg_value = (card.up + card.right + card.down + card.left) / 4
        if avg_value <= 5:
            return '逆转规则下偏好低数值卡牌'
        else:
            return '虽然数值较高，但可能用于特殊战术'
    elif '王牌杀手' in rules:
        values = [card.up, card.right, card.down, card.left]
        if 1 in values or 10 in values:
            return '王牌杀手规则下，含1或A的卡牌有特殊效果'
        else:
            return '可能作为支援卡牌使用'
    else:
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, 3)
        if star <= 2:
            return '低星级卡牌，可能用于早期控场'
        elif star >= 4:
            return '高星级卡牌，关键时刻的强力选择'
        else:
            return '中等星级卡牌，平衡性较好'

def analyze_opponent_strategy(rules, board, known_cards):
    """分析对手策略"""
    if '同数' in rules:
        return '对手可能会设置同数连携陷阱，注意重复数值卡牌'
    elif '加算' in rules:
        return '对手可能会利用加算规则设置连携，注意互补数值组合'
    elif '逆转' in rules:
        return '对手在逆转规则下可能偏好低数值卡牌'
    elif '王牌杀手' in rules:
        return '对手可能重点使用含1或A的卡牌'
    elif '同类强化' in rules:
        return '对手可能会集中使用同类型卡牌以触发强化效果'
    elif '同类弱化' in rules:
        return '对手可能会避免使用相同类型，采用多样化策略'
    else:
        return '对手采用常规策略，平衡攻防'

def predict_unknown_cards(rules, board, count):
    """预测未知卡牌"""
    predictions = []
    
    # 基于规则生成预测
    if '同数' in rules:
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
    elif '逆转' in rules:
        predictions.append({
            'card': '低数值卡牌',
            'confidence': 0.8,
            'reasoning': '逆转规则下低数值更有优势'
        })
    elif '王牌杀手' in rules:
        predictions.append({
            'card': '含1或A的卡牌',
            'confidence': 0.75,
            'reasoning': '王牌杀手规则下1和A具有特殊效果'
        })
    
    # 补充通用预测
    if len(predictions) < count:
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
    
    # 执行移动后的评估
    new_state = game_state.copy()
    card, (row, col) = move
    new_state.play_move(row, col, card)
    after_move_eval = evaluate_current_position(new_state, my_owner)
    
    # 转换为胜率（简化计算）
    current_prob = sigmoid(current_eval)
    after_move_prob = sigmoid(after_move_eval)
    
    return {
        'current': round(current_prob, 3),
        'after_move': round(after_move_prob, 3),
        'confidence': min(0.9, abs(after_move_eval) / 10 + 0.6)  # 基于评估差值的置信度
    }

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
    
    # 检查能否吃掉对手卡牌
    captured_cards = count_captured_cards(game_state, move, my_owner)
    if captured_cards > 0:
        return f"此移动可以吃掉{captured_cards}张对手卡牌"
    
    # 检查位置价值
    if (row, col) in [(0,0), (0,2), (2,0), (2,2)]:
        return "占据角落位置，具有防御优势"
    elif (row, col) == (1,1):
        return "控制中心位置，影响四个方向"
    else:
        return "稳定发展，保持场面控制"

def analyze_strategic_value(game_state, move, my_owner):
    """分析战略价值"""
    card, (row, col) = move
    
    # 检查规则特殊效果
    if '同数' in game_state.rules:
        if has_same_number_potential(card):
            return "为同数连携做准备"
    
    if '加算' in game_state.rules:
        if has_addition_potential(card):
            return "为加算连携做准备"
    
    return "维持场面平衡，为后续发展铺路"

def count_captured_cards(game_state, move, my_owner):
    """计算能吃掉的对手卡牌数量"""
    card, (row, col) = move
    captured = 0
    
    directions = [(-1, 0, 'up', 'down'), (1, 0, 'down', 'up'), 
                 (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
    
    for dr, dc, my_dir, opp_dir in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            opp_card = game_state.board.get_card(nr, nc)
            if opp_card and opp_card.owner != card.owner:
                result = card.compare_values(my_dir, opp_card, opp_dir, game_state.rules)
                if result == 1:
                    captured += 1
    
    return captured

def has_same_number_potential(card):
    """检查是否有同数潜力"""
    values = [card.up, card.right, card.down, card.left]
    return len(set(values)) < 4  # 有重复数值

def has_addition_potential(card):
    """检查是否有加算潜力"""
    values = [card.up, card.right, card.down, card.left]
    # 检查是否有常见的和数组合
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if i != j and v1 + v2 in [8, 10, 12]:
                return True
    return False

def evaluate_alternative_move(game_state, move, my_owner):
    """评估备选移动"""
    new_state = game_state.copy()
    card, (row, col) = move
    new_state.play_move(row, col, card)
    return evaluate_current_position(new_state, my_owner)

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
        from player import Player
        if my_owner == 'red':
            players = [Player('me', my_hand), Player('opp', opp_hand)]
            current_player_idx = 0
        else:
            players = [Player('opp', opp_hand), Player('me', my_hand)]
            current_player_idx = 1
        game_state = GameState(board, players, current_player_idx=current_player_idx, rules=rules)
        
        move, _ = find_best_move_parallel(
            game_state,
            max_depth=8,  # 增加到8层但会被100层限制
            verbose=False,  # 关闭详细输出减少日志
            all_cards=get_all_cards(),
            open_mode=open_mode
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
        
        # 计算胜率
        win_prob = calculate_win_probability(game_state, move, my_owner)
        
        # 生成AI建议
        recommendation = generate_move_recommendation(game_state, move, my_owner)
        
        # 打印AI给出结果后的棋盘
        new_state = game_state.copy()
        new_state.play_move(row, col, card)
        print("AI给出结果后的棋盘：")
        print(new_state.board)
        
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, '?')
        
        return jsonify({
            'card': f"U{card.up} R{card.right} D{card.down} L{card.left} 星级:{star}",
            'pos': [row, col],
            'opponent_hand_analysis': opponent_analysis,
            'win_probability': win_prob,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)