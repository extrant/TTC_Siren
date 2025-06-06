from flask import Flask, request, jsonify
import pandas as pd
from card import Card
from player import Player
from board import Board
from game_state import GameState
from ai import find_best_move_parallel
import os
import codecs
import threading

# 全局缓存和唯一ID查找表
_card_db = None
_all_cards = None
_card_lookup = None
_card_lock = threading.Lock()
_card_star_map = None

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
                card_id=int(row['序号'])
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

def parse_owner(owner):
    # 1=蓝方, 2=红方
    return 'blue' if owner == 1 else 'red'

def find_card_id_by_stats(up, right, down, left):
    lookup = get_card_lookup()
    return lookup.get((up, right, down, left), None)

def parse_board(board_json):
    board = Board()
    for item in board_json:
        r, c = item['pos']
        up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
        card_id = find_card_id_by_stats(up, right, down, left)
        if card_id is None:
            raise ValueError(f"Board card not found in database: U{up} R{right} D{down} L{left}")
        card = Card(
            up=up,
            right=right,
            down=down,
            left=left,
            owner=parse_owner(item['owner']),
            card_id=card_id
        )
        board.place_card(r, c, card)
    return board

def parse_hand(hand_json, owner, used_cards, id_offset=1000):
    hand = []
    all_cards = get_all_cards()
    for idx, item in enumerate(hand_json):
        print(f"hand_json: {hand_json} owner: {owner} used_cards: {used_cards}")
        if all([item[k] == 0 for k in ['numU', 'numR', 'numD', 'numL']]):
            # 未知手牌，用全牌池中的牌补全（不考虑used_cards限制，因为对手可能有相同的牌）
            # print('Filling unknown card with all possible cards')
            for card in all_cards:
                c = Card(card.up, card.right, card.down, card.left, owner, card.card_id)
                hand.append(c)
        else:
            up, right, down, left = item['numU'], item['numL'], item['numD'], item['numR']
            card_id = find_card_id_by_stats(up, right, down, left)
            if card_id is None:
                raise ValueError(f"Hand card not found in database: U{up} R{right} D{down} L{left}")
            c = Card(up, right, down, left, owner, card_id)
            hand.append(c)
            used_cards.add(card_id)
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
                    completed.append(Card(card.up, card.right, card.down, card.left, None, card.card_id))
                    used_cards.add(card.card_id)
                    break
    return completed

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
        my_hand = parse_hand(data['myHand'], my_owner, used_cards)
        opp_hand = parse_hand(data['oppHand'], opp_owner, used_cards)
        rules, open_mode = parse_rules_and_open_mode(data.get('rules', ''))
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
            max_depth=4,
            verbose=True,
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
        # 打印AI给出结果后的棋盘
        new_state = game_state.copy()
        new_state.play_move(row, col, card)
        print("AI给出结果后的棋盘：")
        print(new_state.board)
        

        
        star_map = get_card_star_map()
        star = star_map.get(card.card_id, '?')
        
        return jsonify({
            'card': f"U{card.up} R{card.right} D{card.down} L{card.left} 星级:{star}",
            'pos': [row, col]
            
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)