from typing import Tuple, Optional, Dict, List
from game_state import GameState
from card import Card
import copy
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import random
import math
import time


TRANSPOSITION_TABLE = {}

HISTORY_TABLE = {}

TIME_LIMIT = 5.0  # 默认5秒时间限制
START_TIME = 0

class SearchResult:
    def __init__(self, eval_score: float, best_move: Optional[Tuple], path: List = None):
        self.eval_score = eval_score
        self.best_move = best_move
        self.path = path or []

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 懒加载卡牌星级映射
def get_card_star_map():
    if not hasattr(get_card_star_map, '_cache'):
        df = pd.read_csv(resource_path('幻卡数据库.csv'))
        get_card_star_map._cache = {int(row['序号']): int(row['星级']) for _, row in df.iterrows()}
    return get_card_star_map._cache


def evaluate_state(state: GameState, ai_player_idx: int) -> float:
    """改进的评估函数"""
    red_count, blue_count = state.count_cards()
    
    # 基础分数
    if ai_player_idx == 0:
        base_score = red_count - blue_count
    else:
        base_score = blue_count - red_count
        
    # 位置权重
    position_weights = {
        (0,0): 1.5, (0,2): 1.5, (2,0): 1.5, (2,2): 1.5,  # 角落
        (1,1): 1.2  # 中心
    }
    
    position_score = 0
    for r in range(3):
        for c in range(3):
            card = state.board.get_card(r, c)
            if card:
                weight = position_weights.get((r,c), 1.0)
                if (card.owner == 'red' and ai_player_idx == 0) or \
                   (card.owner == 'blue' and ai_player_idx == 1):
                    position_score += weight
                else:
                    position_score -= weight
    
    return base_score * 0.7 + position_score * 0.3


"""
高级评估函数数学模型说明：

1. 综合评分函数 (Comprehensive Evaluation Function)
\[
E_{total} = \alpha H + \beta P + \gamma C + \delta T + \epsilon N
\]
其中：
- H: 历史启发得分 (History Heuristic Score)
- P: 位置评估得分 (Position Evaluation Score)
- C: 卡牌属性得分 (Card Attribute Score)
- T: 战术评估得分 (Tactical Evaluation Score)
- N: 邻近协同得分 (Neighborhood Synergy Score)
权重系数：\alpha = 0.1, \beta = 1.5, \gamma = 2.0, \delta = 3.0, \epsilon = 1.0

2. 历史启发评分 (History Heuristic)
\[
H(m) = min(\frac{h(m)}{10}, 1000)
\]
其中 h(m) 为历史表中的原始值

3. 位置权重矩阵 (Position Weight Matrix)
\[
W = \begin{bmatrix} 
1.5 & 1.0 & 1.5 \\
1.0 & 1.2 & 1.0 \\
1.5 & 1.0 & 1.5
\end{bmatrix}
\]

4. 卡牌战术价值 (Card Tactical Value)
\[
V_{card} = \sum_{i=1}^4 E_i + S \cdot M + \sum_{j=1}^k C_j
\]
其中：
- E_i: 边值评估 (Edge Values)
- S: 星级系数 (Star Rating)
- M: 位置乘数 (Position Multiplier)
- C_j: 组合加成 (Combination Bonus)

5. 吃子评估函数 (Capture Evaluation)
\[
Cap(x,y) = 30 + 5\Delta + 10(S_1 - S_2)
\]
其中：
- \Delta: 数值差异 (Value Difference)
- S_1: 己方星级 (Own Star Rating)
- S_2: 对方星级 (Opponent Star Rating)

6. 邻近协同系数 (Neighborhood Synergy)
\[
N(x,y) = \sum_{i,j \in Adj(x,y)} 10 \cdot I(owner_{i,j} = owner_{x,y})
\]
其中 I 为示性函数

7. 深度奖励因子 (Depth Reward Factor)
\[
R(d) = min(2^d, 1000000)
\]

8. 最终评分归一化 (Score Normalization)
\[
Score_{final} = min(\frac{Score_{raw}}{1000}, 1.0) \cdot 1000
\]

动态评估优化：
1. 早期游戏 (d ≤ 2): 增加位置权重 (\beta *= 1.2)
2. 中期游戏 (2 < d ≤ 6): 增加吃子权重 (\delta *= 1.3)
3. 晚期游戏 (d > 6): 增加协同权重 (\epsilon *= 1.4)

"""

def evaluate_move(move: Tuple, state: GameState, history_table: Dict) -> float:
    """
    综合评估移动的分数，融合历史启发和启发式评估
    """
    card, (row, col) = move
    score = 0.0
    
    # 1. 历史启发表分数 (基础权重 0.1)
    history_score = history_table.get((card.card_id, row, col), 0)
    score += min(history_score * 0.1, 1000.0)  # 限制历史分数的最大值
    
    # 2. 卡牌星级和边数值评估 (权重 2.0)
    try:
        star = card.star if hasattr(card, 'star') else None
    except Exception:
        star = None
    if hasattr(card, 'card_id'):
        star = get_card_star_map().get(card.card_id, 0)
        
    card_edges = [card.up, card.down, card.left, card.right]
    
    # 高星级卡牌在角落的评估
    if star == 3 and card_edges.count(8) >= 2 and (row, col) in [(0,0),(0,2),(2,0),(2,2)]:
        score += 40.0  # 三星双8角落
    if star == 4 and card_edges.count(9) >= 2:
        score += 30.0  # 四星双9
    if star == 5 and card_edges.count(9) >= 3:
        score += 50.0  # 五星三9
        
    # 3. 位置评估 (权重 1.5)
    if (row, col) in [(0,0), (0,2), (2,0), (2,2)]:
        score += 15.0  # 角落位置
    elif (row, col) == (1,1):
        score += 10.0  # 中心位置
        
    # 4. 吃子评估 (权重 3.0)
    directions = [(-1, 0, 'up', 'down'), (1, 0, 'down', 'up'), 
                 (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
                 
    for dr, dc, my_dir, opp_dir in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            opp_card = state.board.get_card(nr, nc)
            if opp_card and opp_card.owner != card.owner:
                my_value = getattr(card, my_dir)
                opp_value = getattr(opp_card, opp_dir)
                if my_value > opp_value:
                    diff = my_value - opp_value
                    score += 30.0 + diff * 5.0  # 基础吃子分30，每点数值差加5
                    
                    # 如果是高星级卡吃低星级卡，额外加分
                    if star and hasattr(opp_card, 'card_id'):
                        opp_star = get_card_star_map().get(opp_card.card_id, 0)
                        if star > opp_star:
                            score += (star - opp_star) * 10.0
    
    # 5. 边缘保护评估 (权重 1.0)
    # 检查是否有己方卡牌在相邻位置
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            adj_card = state.board.get_card(nr, nc)
            if adj_card and adj_card.owner == card.owner:
                score += 10.0  # 相邻己方卡牌
    
    return float(min(score, 1000.0))  # 确保最终分数不会过大

def order_moves(moves: List[Tuple], state: GameState, history_table: Dict) -> List[Tuple]:
    """
    使用综合评估函数对移动进行排序
    """
    return sorted(moves, key=lambda move: evaluate_move(move, state, history_table), reverse=True)

def get_state_hash(state: GameState) -> str:
    """生成状态的哈希值"""
    board_str = str(state.board)
    hands_str = "".join(str(card.card_id) for player in state.players for card in player.hand)
    return f"{board_str}_{hands_str}_{state.current_player_idx}"

def is_time_up() -> bool:
    """检查是否超时"""
    return time.time() - START_TIME > TIME_LIMIT

def minimax(state: GameState, depth: int, alpha: float, beta: float, maximizing: bool, 
           ai_player_idx: int, verbose: bool = False, is_root: bool = False, 
           history_table: Dict = None, path: List = None) -> SearchResult:
    """改进的极小极大搜索"""
    if path is None:
        path = []
    if history_table is None:
        history_table = {}
        
    # 检查时间限制
    if is_time_up():
        return SearchResult(evaluate_state(state, ai_player_idx), None, path)
        
    # 检查终止条件
    if depth == 0 or state.is_game_over():
        return SearchResult(evaluate_state(state, ai_player_idx), None, path)
        
    # 置换表查找
    state_hash = get_state_hash(state)
    if state_hash in TRANSPOSITION_TABLE:
        tt_entry = TRANSPOSITION_TABLE[state_hash]
        if tt_entry['depth'] >= depth:
            return SearchResult(float(tt_entry['score']), tt_entry['move'], path)
            
    current_player = state.players[state.current_player_idx]
    moves = [(card, pos) for card in current_player.hand 
             for pos in state.board.available_positions()]
             
    if not moves:
        return SearchResult(evaluate_state(state, ai_player_idx), None, path)
        
    # 移动排序
    moves = order_moves(moves, state, history_table)
    
    best_move = None
    best_path = []
    
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            if is_time_up():
                break
                
            card, (row, col) = move
            next_state = state.copy()
            next_state.play_move(row, col, card)
            
            result = minimax(next_state, depth-1, alpha, beta, False, ai_player_idx, 
                           verbose, False, history_table, path + [state.copy()])
                           
            if result.eval_score > max_eval:
                max_eval = result.eval_score
                best_move = move
                best_path = result.path
                
            alpha = max(alpha, result.eval_score)
            if beta <= alpha:
                # 更新历史启发表
                if best_move:
                    card, (row, col) = best_move
                    key = (card.card_id, row, col)
                    history_table[key] = min(history_table.get(key, 0) + 2 ** depth, 1000000)
                break
                
        # 更新置换表
        TRANSPOSITION_TABLE[state_hash] = {
            'depth': depth,
            'score': float(max_eval),
            'move': best_move
        }
        
        return SearchResult(max_eval, best_move, best_path)
    else:
        min_eval = float('inf')
        for move in moves:
            if is_time_up():
                break
                
            card, (row, col) = move
            next_state = state.copy()
            next_state.play_move(row, col, card)
            
            result = minimax(next_state, depth-1, alpha, beta, True, ai_player_idx,
                           verbose, False, history_table, path + [state.copy()])
                           
            if result.eval_score < min_eval:
                min_eval = result.eval_score
                best_move = move
                best_path = result.path
                
            beta = min(beta, result.eval_score)
            if beta <= alpha:
                # 更新历史启发表
                if best_move:
                    card, (row, col) = best_move
                    key = (card.card_id, row, col)
                    history_table[key] = min(history_table.get(key, 0) + 2 ** depth, 1000000)
                break
                
        # 更新置换表
        TRANSPOSITION_TABLE[state_hash] = {
            'depth': depth,
            'score': float(min_eval),
            'move': best_move
        }
        
        return SearchResult(min_eval, best_move, best_path)

def iterative_deepening_search(state: GameState, max_time: float, verbose: bool = False) -> Tuple:
    """迭代加深搜索"""
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = max_time
    
    history_table = {}
    best_move = None
    best_path = []
    depth = 1
    
    while not is_time_up():
        if verbose:
            print(f"\n开始深度 {depth} 的搜索...")
            
        result = minimax(state, depth, float('-inf'), float('inf'), True,
                        state.current_player_idx, verbose, True, history_table)
                        
        if is_time_up() and depth > 1:
            break
            
        best_move = result.best_move
        best_path = result.path
        
        if verbose:
            print(f"深度 {depth} 完成，最佳移动：{best_move}，评分：{result.eval_score}")
            
        depth += 1
        
        # 早停
        if result.eval_score > 10:
            break
            
    return best_move, best_path

def find_best_move_parallel(game_state: GameState, max_depth: int = 9, verbose: bool = False,
                          all_cards=None, n_jobs=None, progress_callback=None, open_mode='none'):
    """并行搜索入口"""
    # 清理全局状态
    global TRANSPOSITION_TABLE
    TRANSPOSITION_TABLE = {}
    
    # 使用迭代加深搜索
    best_move, best_path = iterative_deepening_search(
        game_state,
        max_time=10.0,  # 5秒时间限制
        verbose=verbose
    )
    
    return best_move, best_path 