from typing import List, Optional
from board import Board
from player import Player
from card import Card
import copy

class GameState:
    """
    幻卡游戏状态类，包含牌桌、双方玩家、当前回合玩家、规则。
    严格按照官方规则进行胜负判定。
    """
    def __init__(self, board: Board, players: List[Player], current_player_idx: int = 0, rules: Optional[List[str]] = None):
        self.board = board  # 当前牌桌
        self.players = players  # [红方玩家, 蓝方玩家]，约定0为红，1为蓝
        self.current_player_idx = current_player_idx  # 当前回合玩家索引（0或1）
        self.rules = rules if rules is not None else []  # 当前规则列表

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

    def play_move(self, row: int, col: int, card: Card) -> bool:
        #print(f"[GameState] play_move: player={self.current_player.name}, row={row}, col={col}, card_id={card.card_id}")
        #print(f"[GameState] current_player.hand={[c.card_id for c in self.current_player.hand]}")
        # 用card_id判断是否在手牌
        hand_card = next((c for c in self.current_player.hand if c.card_id == card.card_id), None)
        if self.board.is_empty(row, col) and hand_card:
            card.owner = 'red' if self.current_player_idx == 0 else 'blue'
            self.board.place_card(row, col, card)
            self.current_player.play_card(hand_card)
            #print(f"[GameState] play_move success: placed card_id={card.card_id}")
            self.resolve_flip(row, col, card)
            
            # 同类强化/弱化处理
            if '同类强化' in self.rules or '同类弱化' in self.rules:
                self.apply_same_type_effect(card)
            
            self.current_player_idx = 1 - self.current_player_idx
            return True
        #print("[GameState] play_move failed: not empty or card not in hand")
        return False

    def apply_same_type_effect(self, played_card: Card):
        """
        应用同类强化/弱化效果
        """
        if not played_card.card_type:
            return  # 无类型的卡牌不触发同类效果
        
        delta = 0
        if '同类强化' in self.rules:
            delta = 1
        elif '同类弱化' in self.rules:
            delta = -1
        
        # 影响牌桌上的同类型卡牌
        for r in range(3):
            for c in range(3):
                board_card = self.board.get_card(r, c)
                if board_card and board_card.card_type == played_card.card_type:
                    board_card.modify_stats(delta)
        
        # 影响双方手牌中的同类型卡牌
        for player in self.players:
            for hand_card in player.hand:
                if hand_card.card_type == played_card.card_type:
                    hand_card.modify_stats(delta)

    def resolve_flip(self, row: int, col: int, card: Card, flipped_set=None, chain_only=False):
        """
        翻面判定，支持基础规则、加算、同数及连锁。
        现在也支持逆转和王牌杀手规则。
        flipped_set: 递归时传递，记录已翻转的格子，避免死循环。
        chain_only: 连携递归时为True，仅用基础规则。
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
                            # 对于加算规则，使用原始数值（不受逆转和王牌杀手影响）
                            my_value = getattr(card, my_dir)
                            opp_value = getattr(opp_card, opp_dir)
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
                            # 对于同数规则，使用原始数值（不受逆转和王牌杀手影响）
                            my_value = getattr(card, my_dir)
                            opp_value = getattr(opp_card, opp_dir)
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
                opp_card.owner = owner
                flipped_set.add((nr, nc))
                # 打印翻转消息
                #if reason == 'same':
                #    print(f"[同数] ({nr},{nc}) 卡牌ID={opp_card.card_id} 被翻转为 {owner}")
                #elif reason == 'plus':
                #    print(f"[加算] ({nr},{nc}) 卡牌ID={opp_card.card_id} 被翻转为 {owner}")
                #elif reason == 'base':
                #    pass  # 基础规则不打印
                # 连锁：被翻转卡牌如果与其他对方卡牌满足规则则继续翻转
                if reason in ('same', 'plus'):
                    # if reason != 'base':
                        # print(f"[连锁] 递归检查 ({nr},{nc}) 卡牌ID={opp_card.card_id}")
                    self.resolve_flip(nr, nc, opp_card, flipped_set, chain_only=True) 