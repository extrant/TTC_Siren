from typing import Optional

class Card:
    """
    幻卡卡牌类，包含四个方向的数字、归属方、卡牌ID、卡牌类型。
    """
    def __init__(self, up: int, right: int, down: int, left: int, owner: Optional[str] = None, 
                 card_id: Optional[int] = None, card_type: Optional[str] = None, can_use: bool = True):
        # 原始数值（永远不变）
        self.base_up = up      
        self.base_right = right  
        self.base_down = down    
        self.base_left = left    
        
        # 当前有效数值（用于显示和兼容性）
        self.up = up      
        self.right = right  
        self.down = down    
        self.left = left    
        
        self.owner = owner  # 归属方（'red'/'blue'/None）
        self.card_id = card_id  # 卡牌ID或类型编号
        self.card_type = card_type  # 卡牌类型（兽人、拂晓、帝国、蛮神）
        self.can_use = can_use  # 是否可以使用（用于秩序/混乱规则）
        
        # 同类强化/弱化修正值
        self.type_modifier = 0

    def __repr__(self):
        return f"Card(ID={self.card_id}, U={self.up}, R={self.right}, D={self.down}, L={self.left}, owner={self.owner}, type={self.card_type})"

    def display(self) -> str:
        """
        返回卡牌四面数值的字符串表示，格式如：U5 L2 D3 R1
        如果有同类修正，显示为：U5+1 L2+1 D3+1 R1+1
        """
        if self.type_modifier != 0:
            modifier_str = f"{self.type_modifier:+d}"
            return f"U{self.base_up}{modifier_str} L{self.base_left}{modifier_str} D{self.base_down}{modifier_str} R{self.base_right}{modifier_str}"
        else:
            return f"U{self.base_up} L{self.base_left} D{self.base_down} R{self.base_right}"

    def display_multiline(self) -> list:
        """
        返回4行字符串列表，不加颜色，每行宽度7。
        如果有同类修正，显示修正后的数值。
        """
        owner = self.owner[0].upper() if self.owner else ' '
        cid = self.card_id if self.card_id is not None else ' '
        
        # 获取显示用的数值
        up_display = self.get_display_value('up')
        right_display = self.get_display_value('right')
        down_display = self.get_display_value('down')
        left_display = self.get_display_value('left')
        
        lines = [
            f"[{owner}:{cid}]".center(7),
            f"U{up_display}".center(7),
            f"L{left_display}" + "   " + f"R{right_display}",
            f"D{down_display}".center(7)
        ]
        return lines 

    def get_base_value(self, direction: str) -> int:
        """获取指定方向的原始数值"""
        return getattr(self, f'base_{direction}')
    
    def get_modified_value(self, direction: str) -> int:
        """获取应用同类修正后的数值"""
        base_value = self.get_base_value(direction)
        return max(1, min(10, base_value + self.type_modifier))
    
    def get_display_value(self, direction: str) -> str:
        """获取用于显示的数值字符串"""
        if self.type_modifier != 0:
            base_val = self.get_base_value(direction)
            modifier_str = f"{self.type_modifier:+d}"
            return f"{base_val}{modifier_str}"
        else:
            return str(self.get_base_value(direction))
    
    def get_effective_value(self, direction: str, rules: list) -> int:
        """
        根据规则获取指定方向的最终有效数值
        考虑同类强化/弱化，但不包括王牌杀手规则（王牌杀手需要特殊处理）
        """
        # 应用同类修正
        return self.get_modified_value(direction)

    def compare_values(self, my_direction: str, other_card: 'Card', other_direction: str, rules: list) -> int:
        """
        比较两张卡牌的数值
        返回: 1表示我方胜, -1表示对方胜, 0表示平局
        """
        my_value = self.get_effective_value(my_direction, rules)
        other_value = other_card.get_effective_value(other_direction, rules)
        
        # 王牌杀手规则：1与A同时大于对方，其他数字大小不变
        if '王牌杀手' in rules:
            my_is_ace_killer = (my_value == 1 or my_value == 10)
            other_is_ace_killer = (other_value == 1 or other_value == 10)
            
            if my_is_ace_killer and other_is_ace_killer:
                # 1 vs A 或 A vs 1：按原始数值比较
                # 1 < 10，所以A胜过1
                pass  # 继续使用下面的正常比较逻辑
            elif my_is_ace_killer and not other_is_ace_killer:
                # 我方是1或A，对方是2-9：我方胜
                return 1 if not '逆转' in rules else -1
            elif not my_is_ace_killer and other_is_ace_killer:
                # 我方是2-9，对方是1或A：对方胜
                return -1 if not '逆转' in rules else 1
            # 如果都不是1或A，继续正常比较
        
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

    def apply_type_modifier(self, modifier: int):
        """
        应用同类强化/弱化修正值
        """
        self.type_modifier = modifier
        # 更新显示用的当前数值（为了兼容性）
        self.up = self.get_modified_value('up')
        self.right = self.get_modified_value('right')
        self.down = self.get_modified_value('down')
        self.left = self.get_modified_value('left')
    
    def modify_stats(self, delta: int):
        """
        修改卡牌的四个方向数值（用于同类强化/弱化）
        这是为了向后兼容而保留的方法
        """
        self.apply_type_modifier(delta)

    def copy(self) -> 'Card':
        """
        创建卡牌的深拷贝
        """
        new_card = Card(self.base_up, self.base_right, self.base_down, self.base_left, 
                       self.owner, self.card_id, self.card_type, self.can_use)
        new_card.type_modifier = self.type_modifier
        # 更新显示数值
        new_card.up = new_card.get_modified_value('up')
        new_card.right = new_card.get_modified_value('right')
        new_card.down = new_card.get_modified_value('down')
        new_card.left = new_card.get_modified_value('left')
        return new_card 