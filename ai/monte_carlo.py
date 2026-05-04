"""
蒙特卡洛求解器 - 基于LinkRoss的PVE算法改编为PVP幻卡锦标赛

核心思想（来自LinkRoss SolverA）：
1. 对对手的未知手牌进行随机采样
2. 对每个候选走法，模拟大量随机对局到结束
3. 统计胜率，选择期望胜率最高的走法

相比Minimax的优势：
- 自然处理未知信息的不确定性
- 不假设对手完美发挥（更符合真实PVP场景）
- 算法简单，不易出现逻辑bug
- 通过大量采样自然平均各种可能性

参考: LinkRoss/Solvers/SolverA.py 的 get_steps_score 方法
"""

import random
import time
import copy
from typing import List, Tuple, Optional, Set, Dict
from core.card import Card
from core.player import Player
from core.game_state import GameState


class MonteCarloSolver:
    """
    蒙特卡洛求解器

    Parameters
    ----------
    game_state : GameState
        当前游戏状态（包含已知信息）
    all_cards : List[Card]
        完整卡牌数据库
    ai_player_idx : int
        AI玩家索引 (0=红方, 1=蓝方)
    time_limit : float
        时间预算（秒）
    """

    def __init__(self, game_state: GameState, all_cards: List[Card],
                 ai_player_idx: int, time_limit: float = 5.0):
        self.original_state = game_state
        self.all_cards = all_cards
        self.ai_player_idx = ai_player_idx
        self.time_limit = time_limit
        self.start_time = 0.0

        # 记录对手玩家的索引
        self.opp_idx = 1 - ai_player_idx

        # 构建卡牌ID到Card对象的快速查找表
        self._card_by_id: Dict[int, Card] = {c.card_id: c for c in all_cards if c.card_id}

        # 收集已确定使用的卡牌ID
        self.used_card_ids = self._collect_used_card_ids(game_state)

        # 可用卡池（用于采样对手未知手牌）
        card_star_map = {}
        card_type_map = {}
        for c in all_cards:
            if c.card_id:
                card_star_map[c.card_id] = getattr(c, 'star', 1)
                card_type_map[c.card_id] = c.card_type

        self.available_pool_ids = [
            cid for cid in card_star_map
            if cid not in self.used_card_ids
        ]

        # 统计对手未知手牌数量（同时检测真正的占位符和 AI 生成的预测卡牌）
        self.opp_unknown_count = 0
        self.opp_known_indices = []
        self.opp_hand_slots = 0  # 对手手牌槽位总数
        opp_player = game_state.players[self.opp_idx]
        for i, card in enumerate(opp_player.hand):
            if self._is_unknown_or_generated(card):
                self.opp_unknown_count += 1
            else:
                self.opp_known_indices.append(i)
            self.opp_hand_slots += 1

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _collect_used_card_ids(self, state: GameState) -> Set[int]:
        """收集所有已确定使用的卡牌ID（棋盘 + 已知手牌）"""
        used = set()
        for r in range(3):
            for c in range(3):
                card = state.board.get_card(r, c)
                if card and card.card_id:
                    used.add(card.card_id)
        for player in state.players:
            for card in player.hand:
                if card.card_id and not self._is_unknown_or_generated(card):
                    used.add(card.card_id)
        return used

    @staticmethod
    def _is_unknown_or_generated(card: Card) -> bool:
        """判断是否为未知卡牌（占位符 或 AI生成的预测卡）"""
        # 原始未知占位符（全0）
        if card.up == 0 and card.right == 0 and card.down == 0 and card.left == 0:
            return True
        # AI 生成的预测卡牌（UnknownCardHandler 标记）
        if getattr(card, '_is_generated', False) or getattr(card, '_is_prediction', False):
            return True
        # card_id >= 1000 是预测卡牌的 ID 偏移
        if card.card_id and card.card_id >= 1000:
            return True
        return False

    def _build_card_from_id(self, card_id: int, owner: str = None,
                            can_use: bool = True) -> Card:
        """从卡牌ID构建Card对象"""
        src = self._card_by_id.get(card_id)
        if src is None:
            # fallback: 从all_cards重建
            return None
        return Card(
            up=src.base_up,
            right=src.base_right,
            down=src.base_down,
            left=src.base_left,
            owner=owner,
            card_id=card_id,
            card_type=src.card_type,
            can_use=can_use
        )

    # ------------------------------------------------------------------
    # 对手手牌采样
    # ------------------------------------------------------------------

    def _sample_opponent_hand_cards(self) -> List[Card]:
        """
        为对手的未知手牌采样具体卡牌
        返回值：完整的对手手牌列表（包括已知 + 采样）
        """
        opp_player = self.original_state.players[self.opp_idx]

        # 采样未知卡牌ID
        pool = list(self.available_pool_ids)
        random.shuffle(pool)
        sampled_ids = pool[:self.opp_unknown_count]

        # 构建完整手牌
        result = []
        sampled_iter = iter(sampled_ids)
        for i, card in enumerate(opp_player.hand):
            if i in self.opp_known_indices:
                result.append(self._build_card_from_id(
                    card.card_id, owner=card.owner, can_use=card.can_use))
            else:
                try:
                    cid = next(sampled_iter)
                except StopIteration:
                    cid = random.choice(pool) if pool else 1
                new_card = self._build_card_from_id(
                    cid, owner=card.owner, can_use=card.can_use)
                if new_card:
                    result.append(new_card)
                else:
                    # fallback
                    result.append(card.copy())
        return result

    # ------------------------------------------------------------------
    # 构建模拟用的GameState
    # ------------------------------------------------------------------

    def _build_simulation_state(self) -> GameState:
        """
        构建一次模拟使用的GameState：
        - 复制棋盘
        - 己方手牌不变
        - 对手未知手牌重新采样
        """
        # 复制棋盘
        board_copy = copy.deepcopy(self.original_state.board)

        # 己方手牌（深拷贝）
        my_player = self.original_state.players[self.ai_player_idx]
        my_hand_copy = [card.copy() for card in my_player.hand]

        # 对手手牌（采样未知卡牌）
        opp_hand_copy = self._sample_opponent_hand_cards()

        # 正确的玩家顺序
        if self.ai_player_idx == 0:
            players = [
                Player(my_player.name, my_hand_copy),
                Player(self.original_state.players[1].name, opp_hand_copy)
            ]
        else:
            players = [
                Player(self.original_state.players[0].name, opp_hand_copy),
                Player(my_player.name, my_hand_copy)
            ]

        sim_state = GameState(
            board_copy, players,
            current_player_idx=self.original_state.current_player_idx,
            rules=list(self.original_state.rules)
        )

        # 应用同类规则
        if '同类强化' in sim_state.rules or '同类弱化' in sim_state.rules:
            sim_state.recalculate_type_modifiers()

        return sim_state

    # ------------------------------------------------------------------
    # 随机模拟到终局
    # ------------------------------------------------------------------

    def _random_playout(self, state: GameState) -> float:
        """
        从当前状态随机完成游戏到结束。

        返回值:
            1.0  = AI胜
            0.0  = 平局
           -1.0  = AI负
        """
        max_moves = 10  # 安全上限，防止死循环
        moves_done = 0

        while not state.is_game_over() and moves_done < max_moves:
            moves_done += 1
            player = state.players[state.current_player_idx]
            playable = player.get_playable_cards(state.rules)

            if not playable:
                break

            card = random.choice(playable)
            available = state.board.available_positions()
            if not available:
                break

            pos = random.choice(available)

            try:
                state.make_move(pos[0], pos[1], card)
            except Exception:
                break

        winner = state.get_winner()
        if winner is None:
            return 0.0

        ai_name = self.original_state.players[self.ai_player_idx].name
        return 1.0 if winner == ai_name else -1.0

    # ------------------------------------------------------------------
    # 评估单个走法
    # ------------------------------------------------------------------

    def evaluate_move(self, card: Card, row: int, col: int,
                      num_simulations: int) -> float:
        """
        通过蒙特卡洛模拟评估一个走法的期望胜率。

        Parameters
        ----------
        card : Card
            要打出的卡牌
        row, col : int
            目标位置
        num_simulations : int
            模拟次数

        Returns
        -------
        float
            期望胜率 [-1.0, 1.0]，正值表示AI有利
        """
        total = 0.0
        valid = 0

        for _ in range(num_simulations):
            sim_state = self._build_simulation_state()

            # 在当前模拟状态中执行目标走法
            # 注意：sim_state中的card引用可能不是同一个对象，需要找到对应的
            my_player_sim = sim_state.players[sim_state.current_player_idx]
            my_card = None
            for c in my_player_sim.hand:
                if c.card_id == card.card_id:
                    my_card = c
                    break

            if my_card is None:
                # fallback：使用手牌中的同名卡
                playable = my_player_sim.get_playable_cards(sim_state.rules)
                if playable:
                    my_card = playable[0]
                else:
                    continue

            try:
                sim_state.make_move(row, col, my_card)
            except Exception:
                continue

            score = self._random_playout(sim_state)
            total += score
            valid += 1

        return total / max(valid, 1)

    # ------------------------------------------------------------------
    # 主入口：寻找最佳走法
    # ------------------------------------------------------------------

    def find_best_move(self, base_simulations: int = 150
                       ) -> Tuple[Optional[Tuple], List[Tuple]]:
        """
        寻找当前局面下的最佳走法。

        Parameters
        ----------
        base_simulations : int
            每个候选走法的基准模拟次数

        Returns
        -------
        (best_move, move_scores)
            best_move  : (card, (row, col)) 或 None
            move_scores: [(move, score), ...] 所有评估过的走法及其分数
        """
        self.start_time = time.time()

        moves = self.original_state.get_available_moves()
        if not moves:
            return None, []

        # 如果对手没有未知卡牌，可以用更少的模拟
        if self.opp_unknown_count == 0:
            base_simulations = max(50, base_simulations // 3)

        move_scores: List[Tuple[Tuple, float]] = []

        # 随机打乱走法顺序（避免对先评估的走法不公平）
        random.shuffle(moves)

        for idx, (card, (row, col)) in enumerate(moves):
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                break

            # 动态调整剩余走法的模拟次数
            remaining = len(moves) - idx
            if remaining > 0:
                time_per_move = (self.time_limit - elapsed) / remaining
                # 粗略估计：每次模拟约0.002-0.01秒
                sims = min(base_simulations, max(30, int(time_per_move / 0.005)))
            else:
                sims = base_simulations

            score = self.evaluate_move(card, row, col, sims)
            move_scores.append(((card, (row, col)), score))

        if not move_scores:
            return None, []

        # 选择最高分的走法
        best_move, best_score = max(move_scores, key=lambda x: x[1])
        return best_move, move_scores


# ------------------------------------------------------------------
# 便捷函数 - 与现有 ai_server.py 接口兼容
# ------------------------------------------------------------------

def monte_carlo_best_move(
    game_state: GameState,
    all_cards: List[Card],
    my_owner: str,
    time_limit: float = 5.0,
    base_simulations: int = 150,
    verbose: bool = False
) -> Tuple[Optional[Tuple], List]:
    """
    使用蒙特卡洛方法找到最佳走法。
    接口与 find_best_move_parallel 兼容。

    Parameters
    ----------
    game_state : GameState
        当前游戏状态
    all_cards : List[Card]
        完整卡牌数据库
    my_owner : str
        己方标识 ('red' / 'blue')
    time_limit : float
        时间预算（秒）
    base_simulations : int
        每个走法的基准模拟次数
    verbose : bool
        是否输出详细信息

    Returns
    -------
    (best_move, best_path)
        best_move : (card, (row, col)) 或 None
        best_path : []  占位，与 minimax 接口兼容
    """
    ai_player_idx = 0 if game_state.players[0].name == 'me' else 1

    solver = MonteCarloSolver(
        game_state=game_state,
        all_cards=all_cards,
        ai_player_idx=ai_player_idx,
        time_limit=time_limit
    )

    if verbose:
        print(f"[MC] 对手未知手牌: {solver.opp_unknown_count} 张")
        print(f"[MC] 可用卡池: {len(solver.available_pool_ids)} 张")
        print(f"[MC] 基准模拟次数: {base_simulations}")

    best_move, move_scores = solver.find_best_move(
        base_simulations=base_simulations
    )

    if verbose and move_scores:
        print(f"[MC] 评估了 {len(move_scores)} 个走法:")
        for (card, pos), score in sorted(move_scores, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    卡牌ID={card.card_id} ({card.display()}) at ({pos[0]},{pos[1]})"
                  f" → 胜率期望={score:+.3f}")

    return best_move, []


# ------------------------------------------------------------------
# 卡组构建 - 参考 LinkRoss Sample/SolverA 的 get_deck 逻辑
# ------------------------------------------------------------------

def build_deck_monte_carlo(
    available_cards: List[Card],
    rules: List[str],
    card_event_cards: List[Card] = None,
    card_type_map: Dict[int, str] = None,
    card_star_map: Dict[int, int] = None
) -> List[int]:
    """
    使用蒙特卡洛采样构建最优卡组。
    参考 LinkRoss SolverA.get_deck() 的规则感知卡组选择策略。

    Parameters
    ----------
    available_cards : List[Card]
        可用的卡牌列表
    rules : List[str]
        当前规则列表
    card_event_cards : List[Card]
        对手可能的卡牌（如果已知NPC卡组）
    card_type_map : Dict[int, str]
        卡牌ID → 卡牌类型
    card_star_map : Dict[int, int]
        卡牌ID → 星级

    Returns
    -------
    List[int]
        5张卡牌的ID列表（最优卡组）
    """
    if card_star_map is None:
        card_star_map = {}

    # 按星级分组
    cards_by_star = {i: [] for i in range(1, 6)}
    cards_by_type = {}

    for card in available_cards:
        star = card_star_map.get(card.card_id, 3)
        cards_by_star[star].append(card.card_id)
        if card_type_map:
            ct = card_type_map.get(card.card_id)
            if ct:
                cards_by_type.setdefault(ct, {i: [] for i in range(1, 6)})[star].append(card.card_id)

    # 规则感知选择策略
    same = '同数' in rules
    plus = '加算' in rules
    rev = '逆转' in rules
    ace = '王牌杀手' in rules
    strengthen = '同类强化' in rules
    weaken = '同类弱化' in rules

    # 同类强化+非逆转 → 需要同类型
    # 同类弱化+逆转 → 需要同类型
    need_type = (strengthen and not rev) or (weaken and rev)
    # 同类弱化+非逆转 → 需要无类型
    # 同类强化+逆转 → 需要无类型
    need_no_type = (weaken and not rev) or (strengthen and rev)

    choose = []
    cnt5, cnt4 = 0, 0

    def sample_from_star_pool(pool, n):
        """从指定星级池中随机采样n张"""
        available = [cid for cid in pool if cid not in choose]
        return random.sample(available, min(n, len(available)))

    # Phase 1: 规则驱动的类型选择
    if need_no_type and cards_by_type and '' in cards_by_type:
        no_type_pool = cards_by_type['']
        if no_type_pool.get(5):
            cnt5 = 1
            choose.extend(sample_from_star_pool(no_type_pool[5], 1))
        if no_type_pool.get(4) and cnt4 + cnt5 < 2:
            n = min(2 - cnt4 - cnt5, len(no_type_pool[4]))
            cnt4 += n
            choose.extend(sample_from_star_pool(no_type_pool[4], n))
        for star in [3, 2, 1]:
            if len(choose) >= 3:
                break
            if no_type_pool.get(star):
                choose.extend(sample_from_star_pool(no_type_pool[star],
                                                    3 - len(choose)))
    elif need_type and card_event_cards and cards_by_type:
        # 分析对手类型，选择对手没有的类型
        enemy_types = set()
        for c in card_event_cards:
            if card_type_map and c.card_id in card_type_map:
                ct = card_type_map[c.card_id]
                if ct:
                    enemy_types.add(ct)
        # 优先使用对手没有的类型
        order = [t for t in cards_by_type if t and t not in enemy_types]
        order += list(enemy_types)

        best_choose = []
        for t in order:
            pool = cards_by_type[t]
            temp_choose = []
            if pool.get(5):
                temp_choose.extend(sample_from_star_pool(pool[5], 1))
            if pool.get(4):
                temp_choose.extend(sample_from_star_pool(pool[4],
                                                         min(2 - len(temp_choose), len(pool[4]))))
            for star in [3, 2, 1]:
                if len(temp_choose) >= 3:
                    break
                if pool.get(star):
                    temp_choose.extend(sample_from_star_pool(pool[star],
                                                            3 - len(temp_choose)))
            if len(temp_choose) > len(best_choose):
                best_choose = temp_choose
        choose = best_choose

    # Phase 2: 补全卡组到5张
    if len(choose) < 5:
        # 优先5星
        if cards_by_star[5] and not cnt5:
            cnt5 = 1
            choose.extend(sample_from_star_pool(cards_by_star[5], 1))
        # 再补4星
        if len(choose) < 5 and cards_by_star[4] and cnt4 + cnt5 < 2:
            n = min(2 - cnt4 - cnt5, len(cards_by_star[4]))
            cnt4 += n
            choose.extend(sample_from_star_pool(cards_by_star[4], n))
        # 剩余用3-1星补全
        for star in [3, 2, 1]:
            if len(choose) >= 5:
                break
            choose.extend(sample_from_star_pool(cards_by_star[star],
                                                5 - len(choose)))

    # 去重并补全到5张
    choose = list(set(choose))
    if len(choose) < 5:
        all_remaining = [cid for cid in
                        [c.card_id for c in available_cards]
                        if cid not in choose]
        random.shuffle(all_remaining)
        choose += all_remaining[:5 - len(choose)]

    random.shuffle(choose)
    return choose[:5]
