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

    def get_playable_cards(self, rules: List[str] = None) -> List[Card]:
        """
        获取当前可以使用的卡牌列表
        对于秩序/混乱规则，只返回can_use为True的卡牌
        """
        if rules and ('秩序' in rules or '混乱' in rules):
            return [card for card in self.hand if card.can_use]
        else:
            return list(self.hand)  # 正常情况下所有手牌都可用

    def __repr__(self):
        return f"Player({self.name}, hand={self.hand}, used={self.used_cards})" 