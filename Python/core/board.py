from typing import Optional, List
from .card import Card

class Board:
    """
    幻卡牌桌类，3x3格子，支持放置卡牌、获取卡牌、判断格子是否为空。
    """
    def __init__(self):
        # 3x3二维数组，每格为Card或None
        self.grid: List[List[Optional[Card]]] = [[None for _ in range(3)] for _ in range(3)]

    def place_card(self, row: int, col: int, card: Card) -> bool:
        """在指定位置放置卡牌，若成功返回True，若已被占用返回False。"""
        if self.grid[row][col] is None:
            self.grid[row][col] = card
            return True
        return False

    def get_card(self, row: int, col: int) -> Optional[Card]:
        """获取指定格子的卡牌。"""
        return self.grid[row][col]

    def is_empty(self, row: int, col: int) -> bool:
        """判断指定格子是否为空。"""
        return self.grid[row][col] is None

    def remove_card(self, row: int, col: int) -> Optional[Card]:
        """移除指定位置的卡牌，返回被移除的卡牌，如果位置为空则返回None。"""
        card = self.grid[row][col]
        self.grid[row][col] = None
        return card

    def available_positions(self) -> List[tuple]:
        """返回所有可用（空）格子的坐标列表。"""
        return [(r, c) for r in range(3) for c in range(3) if self.grid[r][c] is None]

    def has_card(self, card_id):
        for row in self.grid:
            for c in row:
                if c and c.card_id == card_id:
                    return True
        return False

    def __repr__(self):
        card_lines = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                card = self.grid[r][c]
                if card is None:
                    card_lines[r][c] = ["       ", "       ", "       ", "       "]
                else:
                    lines = card.display_multiline()
                    # 只在这里加颜色
                    if card.owner == 'red':
                        color_start = '\033[31m'
                        color_end = '\033[0m'
                    elif card.owner == 'blue':
                        color_start = '\033[34m'
                        color_end = '\033[0m'
                    else:
                        color_start = ''
                        color_end = ''
                    lines = [f"{color_start}{line}{color_end}" for line in lines]
                    card_lines[r][c] = lines
        # 拼接每一行
        lines = []
        sep = "+-------+-------+-------+"
        for row in range(3):
            lines.append(sep)
            for subline in range(4):
                lines.append("|" + "|".join(card_lines[row][col][subline] for col in range(3)) + "|")
        lines.append(sep)
        return '\n'.join(lines) 