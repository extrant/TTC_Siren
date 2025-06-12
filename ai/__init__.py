"""
AI智能模块

包含AI算法、搜索引擎和未知卡牌智能处理功能。
"""

from .ai import find_best_move_parallel, SEARCH_STATS
from .unknown_card_handler import (
    initialize_unknown_card_handler, 
    get_unknown_card_handler
)

__all__ = [
    'find_best_move_parallel', 
    'SEARCH_STATS',
    'initialize_unknown_card_handler',
    'get_unknown_card_handler'
] 