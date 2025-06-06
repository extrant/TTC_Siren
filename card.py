from typing import Optional

class Card:
    """
    幻卡卡牌类，包含四个方向的数字、归属方、卡牌ID。
    """
    def __init__(self, up: int, right: int, down: int, left: int, owner: Optional[str] = None, card_id: Optional[int] = None):
        self.up = up      # 上侧数字
        self.right = right  # 右侧数字
        self.down = down    # 下侧数字
        self.left = left    # 左侧数字
        self.owner = owner  # 归属方（'red'/'blue'/None）
        self.card_id = card_id  # 卡牌ID或类型编号

    def __repr__(self):
        return f"Card(ID={self.card_id}, U={self.up}, R={self.right}, D={self.down}, L={self.left}, owner={self.owner})"

    def display(self) -> str:
        """
        返回卡牌四面数值的字符串表示，格式如：U5 L2 D3 R1
        """
        return f"U{self.up} L{self.left} D{self.down} R{self.right}"

    def display_multiline(self) -> list:
        """
        返回4行字符串列表，不加颜色，每行宽度7。
        """
        owner = self.owner[0].upper() if self.owner else ' '
        cid = self.card_id if self.card_id is not None else ' '
        lines = [
            f"[{owner}:{cid}]".center(7),
            f"U{self.up}".center(7),
            f"L{self.left}" + "   " + f"R{self.right}",
            f"D{self.down}".center(7)
        ]
        return lines 