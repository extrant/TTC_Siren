from typing import Optional

class Card:
    """
    幻卡卡牌类，包含四个方向的数字、归属方、卡牌ID、卡牌类型。
    """
    def __init__(self, up: int, right: int, down: int, left: int, owner: Optional[str] = None, 
                 card_id: Optional[int] = None, card_type: Optional[str] = None, can_use: bool = True):
        self.up = up      # 上侧数字
        self.right = right  # 右侧数字
        self.down = down    # 下侧数字
        self.left = left    # 左侧数字
        self.owner = owner  # 归属方（'red'/'blue'/None）
        self.card_id = card_id  # 卡牌ID或类型编号
        self.card_type = card_type  # 卡牌类型（兽人、拂晓、帝国、蛮神）
        self.can_use = can_use  # 是否可以使用（用于秩序/混乱规则）

    def __repr__(self):
        return f"Card(ID={self.card_id}, U={self.up}, R={self.right}, D={self.down}, L={self.left}, owner={self.owner}, type={self.card_type})"

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

    def get_effective_value(self, direction: str, rules: list) -> int:
        """
        根据规则获取指定方向的有效数值
        """
        base_value = getattr(self, direction)
        
        # 王牌杀手规则：1与A(10)同时大于对方
        if '王牌杀手' in rules:
            if base_value == 1:
                return 11  # 给1一个特殊的高数值用于比较
            elif base_value == 10:
                return 11  # A也给同样的高数值
        
        return base_value

    def compare_values(self, my_direction: str, other_card: 'Card', other_direction: str, rules: list) -> int:
        """
        比较两张卡牌的数值
        返回: 1表示我方胜, -1表示对方胜, 0表示平局
        """
        my_value = self.get_effective_value(my_direction, rules)
        other_value = other_card.get_effective_value(other_direction, rules)
        
        # 逆转规则：数字小的一方获胜
        if '逆转' in rules:
            if my_value < other_value:
                return 1
            elif my_value > other_value:
                return -1
            else:
                return 0
        else:
            # 正常规则：数字大的一方获胜
            if my_value > other_value:
                return 1
            elif my_value < other_value:
                return -1
            else:
                return 0

    def modify_stats(self, delta: int):
        """
        修改卡牌的四个方向数值（用于同类强化/弱化）
        """
        self.up = max(1, min(10, self.up + delta))
        self.right = max(1, min(10, self.right + delta))
        self.down = max(1, min(10, self.down + delta))
        self.left = max(1, min(10, self.left + delta))

    def copy(self) -> 'Card':
        """
        创建卡牌的深拷贝
        """
        return Card(self.up, self.right, self.down, self.left, self.owner, self.card_id, self.card_type, self.can_use) 