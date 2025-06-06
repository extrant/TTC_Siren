from typing import List
from card import Card

class Player:
    """
    幻卡玩家类，包含手牌（剩余卡牌）、已用卡牌。
    """
    def __init__(self, name: str, hand: List[Card]):
        self.name = name  # 玩家名称
        self.hand = hand  # 剩余手牌（List[Card]）
        self.used_cards: List[Card] = []  # 已用卡牌

    def play_card(self, card: Card):
        """打出一张卡牌，移出手牌，加入已用卡牌。"""
        if card in self.hand:
            self.hand.remove(card)
            self.used_cards.append(card)

    def __repr__(self):
        return f"Player({self.name}, hand={self.hand}, used={self.used_cards})" 