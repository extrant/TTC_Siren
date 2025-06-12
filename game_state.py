from typing import List, Optional
from board import Board
from player import Player
from card import Card
import copy

class MoveRecord:
    """移动记录，用于undo操作"""
    def __init__(self, row: int, col: int, card: Card, player_idx: int, 
                 flipped_cards: List[tuple] = None, type_modifiers: dict = None):
        self.row = row
        self.col = col
        self.card = card  # 放置的卡牌
        self.player_idx = player_idx  # 执行移动的玩家索引
        self.flipped_cards = flipped_cards or []  # [(row, col, original_owner), ...]
        self.type_modifiers = type_modifiers or {}  # {card_id: original_modifier, ...}

class GameState:
    """
    幻卡游戏状态类，包含牌桌、双方玩家、当前回合玩家、规则。
    严格按照官方规则进行胜负判定。
    现在支持make_move/undo_move机制以避免深拷贝。
    """
    def __init__(self, board: Board, players: List[Player], current_player_idx: int = 0, rules: Optional[List[str]] = None):
        self.board = board  # 当前牌桌
        self.players = players  # [红方玩家, 蓝方玩家]，约定0为红，1为蓝
        self.current_player_idx = current_player_idx  # 当前回合玩家索引（0或1）
        self.rules = rules if rules is not None else []  # 当前规则列表
        self.move_history = []  # 移动历史，用于undo

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def opponent_player(self) -> Player:
        return self.players[1 - self.current_player_idx]

    def copy(self) -> 'GameState':
        """深拷贝当前局面，便于AI搜索。"""
        return GameState(copy.deepcopy(self.board), [copy.deepcopy(p) for p in self.players], self.current_player_idx, list(self.rules))

    def is_game_over(self) -> bool:
        """判断游戏是否结束（牌桌满）。"""
        return all(self.board.grid[r][c] is not None for r in range(3) for c in range(3))

    def count_cards(self) -> (int, int):
        """统计红蓝双方拥有的卡牌总数（包括牌桌和手牌）。"""
        red_count = 0
        blue_count = 0
        # 统计牌桌
        for r in range(3):
            for c in range(3):
                card = self.board.get_card(r, c)
                if card:
                    if card.owner == 'red':
                        red_count += 1
                    elif card.owner == 'blue':
                        blue_count += 1
        # 统计手牌
        for p in self.players:
            for card in p.hand:
                if card.owner == 'red':
                    red_count += 1
                elif card.owner == 'blue':
                    blue_count += 1
        return red_count, blue_count

    def get_winner(self) -> Optional[str]:
        """
        判断胜负，返回胜者名称，平局返回None。
        规则：9格占满后，包括后手未打出的那张卡在内，拥有更多卡牌的人胜。
        """
        red_count, blue_count = self.count_cards()
        if not self.is_game_over():
            return None  # 未结束
        if red_count > blue_count:
            return self.players[0].name  # 红方胜
        elif blue_count > red_count:
            return self.players[1].name  # 蓝方胜
        else:
            return None  # 平局

    def get_available_moves(self) -> List[tuple]:
        """
        获取当前玩家的所有可用动作
        返回: [(card, (row, col)), ...]
        """
        moves = []
        playable_cards = self.current_player.get_playable_cards(self.rules)
        
        for card in playable_cards:
            for row in range(3):
                for col in range(3):
                    if self.board.is_empty(row, col):
                        moves.append((card, (row, col)))
        return moves

    def make_move(self, row: int, col: int, card: Card) -> MoveRecord:
        """
        执行移动并返回移动记录，用于后续undo
        这是新的高性能移动执行方法
        """
        # 检查移动是否有效
        hand_card = next((c for c in self.current_player.hand if c.card_id == card.card_id), None)
        if not self.board.is_empty(row, col) or not hand_card:
            return None
        
        # 记录原始类型修正值（用于undo）
        original_type_modifiers = {}
        if '同类强化' in self.rules or '同类弱化' in self.rules:
            # 记录所有卡牌的当前type_modifier
            for r in range(3):
                for c in range(3):
                    board_card = self.board.get_card(r, c)
                    if board_card:
                        original_type_modifiers[f"board_{r}_{c}"] = board_card.type_modifier
            for i, player in enumerate(self.players):
                for j, hand_card_item in enumerate(player.hand):
                    original_type_modifiers[f"hand_{i}_{j}"] = hand_card_item.type_modifier
        
        # 设置卡牌所有者并放置
        card.owner = 'red' if self.current_player_idx == 0 else 'blue'
        self.board.place_card(row, col, card)
        self.current_player.play_card(hand_card)
        
        # 记录翻转的卡牌（用于undo）
        flipped_cards = []
        self.resolve_flip_with_record(row, col, card, flipped_cards)
        
        # 同类强化/弱化处理
        if '同类强化' in self.rules or '同类弱化' in self.rules:
            self.apply_same_type_effect(card)
        
        # 创建移动记录
        move_record = MoveRecord(
            row=row,
            col=col,
            card=card,
            player_idx=self.current_player_idx,
            flipped_cards=flipped_cards,
            type_modifiers=original_type_modifiers
        )
        
        # 切换玩家
        self.current_player_idx = 1 - self.current_player_idx
        self.move_history.append(move_record)
        
        return move_record

    def undo_move(self, move_record: MoveRecord = None) -> bool:
        """
        撤销移动，恢复到移动前的状态
        如果不提供move_record，则撤销最后一次移动
        """
        if move_record is None:
            if not self.move_history:
                return False
            move_record = self.move_history.pop()
        else:
            # 从历史中移除指定记录
            if move_record in self.move_history:
                self.move_history.remove(move_record)
        
        # 恢复玩家索引
        self.current_player_idx = move_record.player_idx
        
        # 恢复翻转的卡牌
        for flipped_row, flipped_col, original_owner in move_record.flipped_cards:
            flipped_card = self.board.get_card(flipped_row, flipped_col)
            if flipped_card:
                flipped_card.owner = original_owner
        
        # 移除放置的卡牌，恢复到手牌
        self.board.remove_card(move_record.row, move_record.col)
        self.current_player.hand.append(move_record.card)
        move_record.card.owner = None  # 重置所有者
        
        # 恢复类型修正值
        if move_record.type_modifiers:
            # 恢复棋盘卡牌的type_modifier
            for key, original_modifier in move_record.type_modifiers.items():
                if key.startswith("board_"):
                    parts = key.split("_")
                    r, c = int(parts[1]), int(parts[2])
                    board_card = self.board.get_card(r, c)
                    if board_card:
                        board_card.type_modifier = original_modifier
                elif key.startswith("hand_"):
                    parts = key.split("_")
                    player_idx, hand_idx = int(parts[1]), int(parts[2])
                    if (player_idx < len(self.players) and 
                        hand_idx < len(self.players[player_idx].hand)):
                        self.players[player_idx].hand[hand_idx].type_modifier = original_modifier
        
        return True

    def resolve_flip_with_record(self, row: int, col: int, card: Card, flipped_cards: List[tuple], 
                                flipped_set=None, chain_only=False):
        """
        翻面判定，支持基础规则、加算、同数及连锁。
        现在也支持逆转和王牌杀手规则。
        同时记录翻转的卡牌用于undo操作。
        """
        if flipped_set is None:
            flipped_set = set()
        directions = [(-1, 0, 'up', 'down'), (1, 0, 'down', 'up'), (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
        owner = card.owner
        board = self.board
        to_flip = []  # [(nr, nc, reason)]
        
        # --- 基础规则（包含逆转和王牌杀手）---
        for dr, dc, my_dir, opp_dir in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                opp_card = board.get_card(nr, nc)
                if opp_card and opp_card.owner != owner:
                    # 使用新的比较方法
                    result = card.compare_values(my_dir, opp_card, opp_dir, self.rules)
                    if result == 1:  # 我方获胜
                        to_flip.append((nr, nc, 'base'))
        
        if not chain_only:
            # --- 加算规则（修正版） ---
            if '加算' in self.rules:
                plus_list = []  # [(nr, nc, opp_card, 和)]
                sum_map = {}
                for dr, dc, my_dir, opp_dir in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 3 and 0 <= nc < 3:
                        opp_card = board.get_card(nr, nc)
                        if opp_card:
                            # 对于加算规则，使用原始数值（不受同类强化/弱化、逆转和王牌杀手影响）
                            my_value = card.get_base_value(my_dir)
                            opp_value = opp_card.get_base_value(opp_dir)
                            s = my_value + opp_value
                            plus_list.append((nr, nc, opp_card, s))
                            sum_map.setdefault(s, []).append((nr, nc, opp_card))
                # 找出出现次数>=2的和
                valid_sums = [s for s, lst in sum_map.items() if len(lst) >= 2]
                # 至少有一个是敌方卡
                for s in valid_sums:
                    if any(opp_card.owner != owner for _, _, opp_card in sum_map[s]):
                        for nr, nc, opp_card in sum_map[s]:
                            if opp_card.owner != owner:
                                to_flip.append((nr, nc, 'plus'))
            
            # --- 同数规则 ---
            if '同数' in self.rules:
                same_list = []  # [(nr, nc, opp_card)]
                for dr, dc, my_dir, opp_dir in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 3 and 0 <= nc < 3:
                        opp_card = board.get_card(nr, nc)
                        if opp_card:
                            # 对于同数规则，使用原始数值（不受同类强化/弱化、逆转和王牌杀手影响）
                            my_value = card.get_base_value(my_dir)
                            opp_value = opp_card.get_base_value(opp_dir)
                            if my_value == opp_value:
                                same_list.append((nr, nc, opp_card))
                # 至少两次且至少有一次是对方卡
                if len(same_list) >= 2 and any(opp_card.owner != owner for _, _, opp_card in same_list):
                    for nr, nc, opp_card in same_list:
                        if opp_card.owner != owner:
                            to_flip.append((nr, nc, 'same'))
        
        # --- 执行翻转并递归连锁 ---
        for nr, nc, reason in to_flip:
            if (nr, nc) in flipped_set:
                continue  # 已翻转过，避免死循环
            opp_card = board.get_card(nr, nc)
            if opp_card and opp_card.owner != owner:
                # 记录原始所有者（用于undo）
                original_owner = opp_card.owner
                flipped_cards.append((nr, nc, original_owner))
                
                # 执行翻转
                opp_card.owner = owner
                flipped_set.add((nr, nc))
                
                # 连锁：被翻转卡牌如果与其他对方卡牌满足规则则继续翻转
                if reason in ('same', 'plus'):
                    self.resolve_flip_with_record(nr, nc, opp_card, flipped_cards, flipped_set, chain_only=True)

    # 保留原有的play_move方法以兼容现有代码
    def play_move(self, row: int, col: int, card: Card) -> bool:
        """兼容性方法，使用make_move实现"""
        move_record = self.make_move(row, col, card)
        return move_record is not None

    def apply_same_type_effect(self, played_card: Card):
        """
        应用同类强化/弱化效果
        智能计算每张卡牌应该受到的修正值
        """
        if not played_card.card_type:
            return  # 无类型的卡牌不触发同类效果
        
        # 重新计算所有卡牌的同类修正值
        self.recalculate_type_modifiers()
    
    def recalculate_type_modifiers(self):
        """
        重新计算所有卡牌的同类强化/弱化修正值
        根据幻卡规则：影响"双方同类型的卡片，无论在牌桌上还是在牌组内"
        """
        if '同类强化' not in self.rules and '同类弱化' not in self.rules:
            return
        
        # 统计每种类型的总数量（棋盘+所有手牌）
        type_counts = {}
        
        # 1. 统计棋盘上的卡牌类型
        for r in range(3):
            for c in range(3):
                board_card = self.board.get_card(r, c)
                if board_card and board_card.card_type:
                    type_counts[board_card.card_type] = type_counts.get(board_card.card_type, 0) + 1
        
        # 2. 统计所有手牌中的卡牌类型（包括已知和未知）
        for player in self.players:
            for hand_card in player.hand:
                if hand_card.card_type:
                    type_counts[hand_card.card_type] = type_counts.get(hand_card.card_type, 0) + 1
        
        # 3. 对于未知对手手牌，使用智能推测
        type_counts = self._estimate_unknown_hand_types(type_counts)
        
        # 计算修正值并应用
        for card_type, count in type_counts.items():
            if '同类强化' in self.rules:
                # 同类强化：每多一张同类型卡牌，该类型所有卡牌+1
                modifier = count - 1  # 第一张不加成，从第二张开始每张+1
            elif '同类弱化' in self.rules:
                # 同类弱化：每多一张同类型卡牌，该类型所有卡牌-1
                modifier = -(count - 1)  # 第一张不减成，从第二张开始每张-1
            else:
                modifier = 0
            
            # 应用到棋盘上的同类型卡牌
            for r in range(3):
                for c in range(3):
                    board_card = self.board.get_card(r, c)
                    if board_card and board_card.card_type == card_type:
                        board_card.apply_type_modifier(modifier)
            
            # 应用到手牌中的同类型卡牌
            for player in self.players:
                for hand_card in player.hand:
                    if hand_card.card_type == card_type:
                        hand_card.apply_type_modifier(modifier)
    
    def _estimate_unknown_hand_types(self, type_counts: dict) -> dict:
        """
        智能推测未知手牌可能包含的卡牌类型
        基于已知信息、游戏规则和统计学原理
        """
        # 统计未知卡牌数量
        unknown_count = 0
        for player in self.players:
            for hand_card in player.hand:
                # 检查是否为未知卡牌（全0或AI生成的预测卡牌）
                if (hand_card.up == 0 and hand_card.right == 0 and 
                    hand_card.down == 0 and hand_card.left == 0) or \
                   getattr(hand_card, '_is_prediction', False):
                    unknown_count += 1
        
        if unknown_count == 0:
            return type_counts
        
        # 收集已知的类型信息
        known_types = set(type_counts.keys())
        
        # 智能推测策略
        estimated_counts = type_counts.copy()
        
        if known_types:
            # 策略1：保守估计 - 对已有类型假设可能有额外卡牌
            for card_type in known_types:
                current_count = type_counts.get(card_type, 0)
                
                # 根据当前数量调整估计概率
                if current_count == 1:
                    # 如果某类型只有1张，很可能对手手牌有同类型（触发强化）
                    probability = 0.4  # 40%概率
                elif current_count == 2:
                    # 已有2张，再有的概率较低
                    probability = 0.2  # 20%概率
                else:
                    # 已有3+张，再有的概率很低
                    probability = 0.1  # 10%概率
                
                # 计算估计的额外数量
                estimated_additional = max(0, int(unknown_count * probability))
                if estimated_additional > 0:
                    estimated_counts[card_type] = current_count + estimated_additional
        
        # 策略2：根据同类强化/弱化规则调整估计
        if '同类强化' in self.rules:
            # 在同类强化规则下，玩家更倾向于使用同类型卡牌
            # 如果棋盘上出现了某种类型，很可能对手也有相同类型
            for card_type in known_types:
                if estimated_counts.get(card_type, 0) == 1:
                    # 强化概率：如果棋盘上只有1张该类型，对手很可能有同类型
                    estimated_counts[card_type] = 2  # 至少假设有2张（触发+1强化）
        
        elif '同类弱化' in self.rules:
            # 在同类弱化规则下，玩家会尽量避免同类型聚集
            # 估计会更保守
            for card_type in list(estimated_counts.keys()):
                if estimated_counts[card_type] > 1:
                    # 减少估计数量
                    estimated_counts[card_type] = max(1, estimated_counts[card_type] - 1)
        
        return estimated_counts 