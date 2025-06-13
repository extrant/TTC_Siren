from typing import Tuple, Optional, Dict, List, Callable
from core.game_state import GameState
from core.card import Card
import copy
import sys
import os
import pandas as pd
import random
import math
import time
# 新增导入: 获取全局 UnknownCardHandler 以复用其角落评分逻辑
try:
    from ai.unknown_card_handler import get_unknown_card_handler  # type: ignore
except Exception:
    # 兼容运行时尚未初始化或包路径问题
    def get_unknown_card_handler():
        return None

# 角落策略评分的全局权重，可根据实际效果调节
CORNER_STRATEGY_WEIGHT = 5.0

TRANSPOSITION_TABLE = {}

HISTORY_TABLE = {}

TIME_LIMIT = 5.0  # 默认5秒时间限制
START_TIME = 0

# 搜索统计信息
class SearchStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.alpha_beta_cutoffs = 0
        self.move_evaluations = 0
        self.depth_completed = 0
        self.best_score_history = []
        self.best_move_history = []
        self.search_time_per_depth = []
        self.nodes_per_depth = []
        self.branching_factors = []
        
    def add_depth_stats(self, depth: int, nodes: int, time_taken: float, best_score: float, best_move, branching_factor: float):
        self.depth_completed = depth
        self.nodes_per_depth.append(nodes)
        self.search_time_per_depth.append(time_taken)
        self.best_score_history.append(best_score)
        self.best_move_history.append(best_move)
        self.branching_factors.append(branching_factor)
    
    def get_summary(self) -> Dict:
        total_time = sum(self.search_time_per_depth)
        total_nodes = sum(self.nodes_per_depth)
        avg_branching = sum(self.branching_factors) / len(self.branching_factors) if self.branching_factors else 0
        
        return {
            'total_nodes': total_nodes,
            'total_time': total_time,
            'nodes_per_second': total_nodes / total_time if total_time > 0 else 0,
            'tt_hit_rate': self.tt_hits / max(self.nodes_searched, 1),
            'cutoff_rate': self.alpha_beta_cutoffs / max(self.nodes_searched, 1),
            'avg_branching_factor': avg_branching,
            'depths_completed': self.depth_completed,
            'score_trend': self.best_score_history[-3:] if len(self.best_score_history) >= 3 else self.best_score_history
        }

# 全局搜索统计
SEARCH_STATS = SearchStats()

class SearchResult:
    def __init__(self, eval_score: float, best_move: Optional[Tuple], path: List = None, stats: Dict = None):
        self.eval_score = eval_score
        self.best_move = best_move
        self.path = path or []
        self.stats = stats or {}

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 懒加载卡牌星级映射
def get_card_star_map():
    if not hasattr(get_card_star_map, '_cache'):
        df = pd.read_csv(resource_path('data/幻卡数据库.csv'))
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
    corner_edge_score = 0.0  # 新增：角落边值综合评分
    handler = None
    try:
        handler = get_unknown_card_handler()
    except Exception:
        pass
    for r in range(3):
        for c in range(3):
            card = state.board.get_card(r, c)
            if card:
                weight = position_weights.get((r,c), 1.0)
                if (card.owner == 'red' and ai_player_idx == 0) or \
                   (card.owner == 'blue' and ai_player_idx == 1):
                    position_score += weight
                    # 角落边值加分（仅己方卡牌）
                    if handler and (r, c) in [(0,0),(0,2),(2,0),(2,2)]:
                        try:
                            cs = handler._calculate_corner_strategy_score(card, state.board)
                            corner_edge_score += cs * (CORNER_STRATEGY_WEIGHT * 0.6)  # 在总评估中权重稍低
                        except Exception:
                            pass
                else:
                    position_score -= weight
                    # 对手角落高边则扣分
                    if handler and (r, c) in [(0,0),(0,2),(2,0),(2,2)]:
                        try:
                            cs = handler._calculate_corner_strategy_score(card, state.board)
                            corner_edge_score -= cs * (CORNER_STRATEGY_WEIGHT * 0.6)
                        except Exception:
                            pass
    
    return base_score * 0.7 + position_score * 0.3 + corner_edge_score * 0.1


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
        
    # 4. 吃子评估 (权重 3.0) - 支持新规则
    directions = [(-1, 0, 'up', 'down'), (1, 0, 'down', 'up'), 
                 (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
                 
    for dr, dc, my_dir, opp_dir in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            opp_card = state.board.get_card(nr, nc)
            if opp_card and opp_card.owner != card.owner:
                # 使用新的比较方法来支持逆转和王牌杀手规则
                result = card.compare_values(my_dir, opp_card, opp_dir, state.rules)
                if result == 1:  # 我方获胜
                    my_value = getattr(card, my_dir)
                    opp_value = getattr(opp_card, opp_dir)
                    diff = abs(my_value - opp_value)
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

    # 6. 角落/高边战略评分 (新增)
    try:
        handler = get_unknown_card_handler()
        if handler and hasattr(handler, "_calculate_corner_strategy_score"):
            corner_score = handler._calculate_corner_strategy_score(card, state.board)
            score += corner_score * CORNER_STRATEGY_WEIGHT
    except Exception:
        # 若处理器未初始化或计算失败，忽略该评分
        pass

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
           history_table: Dict = None, path: List = None, progress_callback: Callable = None) -> SearchResult:
    """改进的极小极大搜索，使用make_move/undo_move机制避免深拷贝"""
    global SEARCH_STATS
    SEARCH_STATS.nodes_searched += 1
    
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
        SEARCH_STATS.tt_hits += 1
        tt_entry = TRANSPOSITION_TABLE[state_hash]
        if tt_entry['depth'] >= depth:
            SEARCH_STATS.tt_cutoffs += 1
            return SearchResult(float(tt_entry['score']), tt_entry['move'], path)
            
    current_player = state.players[state.current_player_idx]
    playable_cards = current_player.get_playable_cards(state.rules)
    moves = [(card, pos) for card in playable_cards 
             for pos in state.board.available_positions()]
             
    if not moves:
        return SearchResult(evaluate_state(state, ai_player_idx), None, path)
        
    # 移动排序
    moves = order_moves(moves, state, history_table)
    
    best_move = None
    best_path = []
    moves_evaluated = 0
    
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            if is_time_up():
                break
                
            moves_evaluated += 1
            SEARCH_STATS.move_evaluations += 1
            
            card, (row, col) = move
            
            # 使用make_move代替深拷贝
            move_record = state.make_move(row, col, card)
            if move_record is None:
                continue  # 无效移动
            
            try:
                result = minimax(state, depth-1, alpha, beta, False, ai_player_idx, 
                               verbose, False, history_table, path + [move], progress_callback)
                               
                if result.eval_score > max_eval:
                    max_eval = result.eval_score
                    best_move = move
                    best_path = result.path
                    
                alpha = max(alpha, result.eval_score)
                
            finally:
                # 始终撤销移动
                state.undo_move(move_record)
                
            if beta <= alpha:
                SEARCH_STATS.alpha_beta_cutoffs += 1
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
        
        # 计算分支因子
        branching_factor = moves_evaluated if depth > 1 else len(moves)
        
        return SearchResult(max_eval, best_move, best_path, {'branching_factor': branching_factor})
    else:
        min_eval = float('inf')
        for move in moves:
            if is_time_up():
                break
                
            moves_evaluated += 1
            SEARCH_STATS.move_evaluations += 1
            
            card, (row, col) = move
            
            # 使用make_move代替深拷贝
            move_record = state.make_move(row, col, card)
            if move_record is None:
                continue  # 无效移动
            
            try:
                result = minimax(state, depth-1, alpha, beta, True, ai_player_idx,
                               verbose, False, history_table, path + [move], progress_callback)
                               
                if result.eval_score < min_eval:
                    min_eval = result.eval_score
                    best_move = move
                    best_path = result.path
                    
                beta = min(beta, result.eval_score)
                
            finally:
                # 始终撤销移动
                state.undo_move(move_record)
                
            if beta <= alpha:
                SEARCH_STATS.alpha_beta_cutoffs += 1
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
        
        # 计算分支因子
        branching_factor = moves_evaluated if depth > 1 else len(moves)
        
        return SearchResult(min_eval, best_move, best_path, {'branching_factor': branching_factor})

def iterative_deepening_search(state: GameState, max_time: float, verbose: bool = False, 
                             max_depth: int = 100, progress_callback: Callable = None) -> Tuple:
    """增强的迭代加深搜索，包含详细进度显示"""
    global START_TIME, TIME_LIMIT, SEARCH_STATS
    START_TIME = time.time()
    TIME_LIMIT = max_time
    
    # 重置搜索统计
    SEARCH_STATS.reset()
    
    history_table = {}
    best_move = None
    best_path = []
    depth = 1
    
    if verbose:
        print("="*60)
        print("开始迭代加深搜索")
        print(f"时间限制: {max_time}秒, 最大深度: {max_depth}")
        print("="*60)
    
    while not is_time_up() and depth <= max_depth:
        depth_start_time = time.time()
        nodes_before = SEARCH_STATS.nodes_searched
        
        if verbose:
            print(f"\n{'='*20} 深度 {depth} {'='*20}")
            print(f"开始时间: {time.strftime('%H:%M:%S')}")
            remaining_time = TIME_LIMIT - (time.time() - START_TIME)
            print(f"剩余时间: {remaining_time:.2f}秒")
            
        result = minimax(state, depth, float('-inf'), float('inf'), True,
                        state.current_player_idx, verbose, True, history_table, progress_callback=progress_callback)
                        
        depth_end_time = time.time()
        depth_time = depth_end_time - depth_start_time
        nodes_this_depth = SEARCH_STATS.nodes_searched - nodes_before
        
        # 检查是否超时
        if is_time_up() and depth > 1:
            if verbose:
                print(f"深度 {depth} 搜索超时，使用深度 {depth-1} 的结果")
            break
            
        best_move = result.best_move
        best_path = result.path
        
        # 计算分支因子
        branching_factor = result.stats.get('branching_factor', 0)
        
        # 记录深度统计
        SEARCH_STATS.add_depth_stats(depth, nodes_this_depth, depth_time, result.eval_score, best_move, branching_factor)
        
        if verbose:
            print(f"深度 {depth} 完成:")
            print(f"  最佳移动: {format_move_display(best_move)}")
            print(f"  评分: {result.eval_score:.3f}")
            print(f"  搜索节点: {nodes_this_depth:,}")
            print(f"  用时: {depth_time:.3f}秒")
            print(f"  节点/秒: {nodes_this_depth/depth_time:,.0f}" if depth_time > 0 else "  节点/秒: N/A")
            print(f"  分支因子: {branching_factor:.2f}")
            
            # 显示搜索统计
            tt_hit_rate = SEARCH_STATS.tt_hits / max(SEARCH_STATS.nodes_searched, 1) * 100
            cutoff_rate = SEARCH_STATS.alpha_beta_cutoffs / max(SEARCH_STATS.nodes_searched, 1) * 100
            print(f"  置换表命中率: {tt_hit_rate:.1f}%")
            print(f"  α-β剪枝率: {cutoff_rate:.1f}%")
            
            # 显示评分趋势
            if len(SEARCH_STATS.best_score_history) >= 2:
                score_change = SEARCH_STATS.best_score_history[-1] - SEARCH_STATS.best_score_history[-2]
                trend = "↑" if score_change > 0 else "↓" if score_change < 0 else "→"
                print(f"  评分变化: {score_change:+.3f} {trend}")
        
        # 回调函数更新
        if progress_callback:
            progress_info = {
                'depth': depth,
                'max_depth': max_depth,
                'best_move': best_move,
                'best_score': result.eval_score,
                'nodes_searched': SEARCH_STATS.nodes_searched,
                'time_elapsed': time.time() - START_TIME,
                'time_remaining': TIME_LIMIT - (time.time() - START_TIME),
                'stats': SEARCH_STATS.get_summary()
            }
            progress_callback(progress_info)
            
        depth += 1
        
        # 自适应深度限制
        if depth > max_depth:
            if verbose:
                print(f"达到最大深度限制({max_depth})，结束搜索")
            break
            
        # 预测下一深度所需时间（更宽松的判断）
        if depth_time > 0 and depth >= 3:  # 至少完成3层搜索再考虑时间限制
            # 考虑置换表命中率的影响，命中率高时搜索会更快
            tt_hit_rate = SEARCH_STATS.tt_hits / max(SEARCH_STATS.nodes_searched, 1)
            tt_speedup_factor = 1.0 - min(tt_hit_rate * 0.3, 0.4)  # 最多40%的加速
            
            # 更保守的分支因子估算，考虑剪枝效果
            effective_branching = max(branching_factor * 0.8, 2.0) if branching_factor > 1 else 2.5
            
            # 估算时间，考虑各种优化因素
            base_estimation = depth_time * (effective_branching ** 1.2)  # 降低指数从1.5到1.2
            estimated_next_time = base_estimation * tt_speedup_factor
            
            remaining_time = TIME_LIMIT - (time.time() - START_TIME)
            
            # 更宽松的缓冲时间：使用95%的剩余时间，且有最小时间保证
            time_threshold = max(remaining_time * 0.95, 0.1)  # 预留5%缓冲时间，或至少0.1秒
            
            if estimated_next_time > time_threshold:
                if verbose:
                    print(f"预计深度 {depth} 需要 {estimated_next_time:.2f}秒，超过可用时间 {time_threshold:.2f}秒")
                    print(f"  (分支因子: {branching_factor:.2f} → 有效: {effective_branching:.2f}, 置换表加速: {(1-tt_speedup_factor)*100:.1f}%)")

            
    if verbose:
        print("\n" + "="*60)
        print("搜索完成总结:")
        summary = SEARCH_STATS.get_summary()
        print(f"  最终深度: {SEARCH_STATS.depth_completed}")
        print(f"  总搜索节点: {summary['total_nodes']:,}")
        print(f"  总用时: {summary['total_time']:.3f}秒")
        print(f"  平均节点/秒: {summary['nodes_per_second']:,.0f}")
        print(f"  置换表命中率: {summary['tt_hit_rate']*100:.1f}%")
        print(f"  α-β剪枝率: {summary['cutoff_rate']*100:.1f}%")
        print(f"  平均分支因子: {summary['avg_branching_factor']:.2f}")
        print(f"  最终最佳移动: {format_move_display(best_move)}")
        print("="*60)
        
    return best_move, best_path

def format_move_display(move):
    """格式化移动显示"""
    if move is None:
        return "无移动"
    
    card, (row, col) = move
    star_map = get_card_star_map()
    star = star_map.get(card.card_id, '?')
    return f"卡牌U{card.up}R{card.right}D{card.down}L{card.left}(★{star}) → 位置({row},{col})"

def find_best_move_parallel(game_state: GameState, max_depth: int = 9, verbose: bool = False, max_time: float = 5,
                          all_cards=None, n_jobs=None, progress_callback=None, open_mode='none'):
    """并行搜索入口，增强进度显示"""
    # 清理全局状态
    global TRANSPOSITION_TABLE
    TRANSPOSITION_TABLE = {}
    
    # 使用迭代加深搜索，限制最大深度为100
    best_move, best_path = iterative_deepening_search(
        game_state,
        max_time=max_time,  # 5秒时间限制
        verbose=verbose,
        max_depth=min(max_depth, 100),  # 确保不超过100层
        progress_callback=progress_callback
    )
    
    return best_move, best_path 