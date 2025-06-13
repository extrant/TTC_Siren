#!/usr/bin/env python3
"""
智能未知牌处理配置文件
"""

# 采样策略配置
SAMPLING_CONFIG = {
    # 基础配置 - 大幅减少采样数量
    'max_unknown_cards_per_hand': 50,  # 每个未知手牌最多生成的卡牌数量
    'fallback_sample_multiplier': 3,   # 回退策略的采样倍数
    
    # 性能优化配置
    'performance_mode': False,           # 启用性能模式
    'aggressive_sampling': False,        # 激进采样模式
    'min_samples_per_unknown': 5,       # 每个未知卡牌的最小采样数
    'max_samples_per_unknown': 10,       # 每个未知卡牌的最大采样数
    
    # 规则特定配置
    'rule_specific': {
        '同类强化': {
            'priority_ratio': 0.4,     # 优先类型的比例
            'balanced_ratio': 0.4,     # 平衡分布的比例
            'random_ratio': 0.2,       # 随机采样的比例
        },
        '同类弱化': {
            'priority_ratio': 0.3,     # 同类弱化时降低优先类型比例
            'balanced_ratio': 0.5,
            'random_ratio': 0.2,
        },
        '逆转': {
            'low_value_ratio': 0.7,    # 低数值卡牌的比例
            'random_ratio': 0.3,       # 随机卡牌的比例
        },
        '王牌杀手': {
            'special_ratio': 0.6,      # 含1或A卡牌的比例
            'normal_ratio': 0.4,       # 普通卡牌的比例
        },
        '同数': {
            'combo_setup_ratio': 0.5,  # 设置连携陷阱的卡牌比例
            'defensive_ratio': 0.3,    # 防御性卡牌比例
            'random_ratio': 0.2,       # 随机卡牌比例
        },
        '加算': {
            'sum_combo_ratio': 0.5,    # 加算连携卡牌比例
            'defensive_ratio': 0.3,    # 防御性卡牌比例
            'random_ratio': 0.2,       # 随机卡牌比例
        },
        '选拔': {
            'exact_star_matching': True,    # 精确星级匹配
            'fallback_to_closest': True,    # 回退到最接近的星级
            'strategic_star_selection': True, # 战略性星级选择
            'confidence_boost': 0.3,        # 选拔规则的置信度提升
            'max_deviation': 0,             # 允许的最大星级偏差
        }
    },
    
    # 星级分布配置（默认策略）
    'star_distribution': {
        1: 0.35,  # 1星卡牌35%
        2: 0.30,  # 2星卡牌30%
        3: 0.20,  # 3星卡牌20%
        4: 0.10,  # 4星卡牌10%
        5: 0.05,  # 5星卡牌5%
    },
    
    # 类型分布权重
    'type_weights': {
        '兽人': 1.0,
        '拂晓': 1.0,
        '帝国': 1.0,
        '蛮神': 1.0,
        'no_type': 0.8,  # 无类型卡牌权重稍低
    },
    
    # 高级配置
    'advanced': {
        'consider_board_synergy': True,    # 是否考虑与棋盘的协同
        'consider_hand_synergy': True,     # 是否考虑与已知手牌的协同
        'adaptive_sampling': True,         # 是否使用自适应采样
        'position_aware': True,            # 是否考虑位置相关性
        'opponent_behavior_modeling': True, # 是否启用对手行为建模
    },
    
    # 对手行为建模配置
    'opponent_behavior': {
        # 同数规则下的对手倾向
        '同数': {
            'preferred_values': [2, 3, 4, 5, 6],  # 容易形成同数的中等数值
            'avoid_extreme_values': True,          # 避免极端数值(1,10)
            'setup_traps': True,                   # 倾向于设置连携陷阱
            'corner_preference': 1.2,              # 角落位置偏好系数
        },
        # 加算规则下的对手倾向  
        '加算': {
            'preferred_sum_ranges': [(5, 9), (10, 14)],  # 偏好的加算范围
            'complementary_values': True,                  # 倾向于互补数值
            'setup_traps': True,                          # 倾向于设置连携陷阱
            'edge_preference': 1.3,                       # 边缘位置偏好系数
        },
        # 逆转规则下的对手倾向
        '逆转': {
            'low_value_preference': 2.0,  # 强烈偏好低数值
            'max_preferred_value': 5,     # 最大偏好数值
            'high_value_avoidance': 0.3,  # 避免高数值的程度
        },
        # 王牌杀手规则下的对手倾向
        '王牌杀手': {
            'ace_killer_preference': 1.8,  # 对1和A的偏好系数
            'mid_value_preference': 0.7,   # 对中等数值的偏好
            'strategic_positioning': True,  # 战略性定位
        },
        # 同类强化规则下的对手倾向
        '同类强化': {
            'type_synergy_preference': 2.0,  # 类型协同偏好
            'snowball_strategy': True,       # 雪球战略
            'defensive_typing': False,       # 防御性类型选择
        },
        # 同类弱化规则下的对手倾向
        '同类弱化': {
            'type_diversity_preference': 1.5,  # 类型多样性偏好
            'disruption_strategy': True,       # 破坏战略
            'min_value_avoidance': True,       # 避免最小值
        },
        # 选拔规则下的对手倾向
        '选拔': {
            'star_conservation_strategy': True,    # 星级保守策略
            'early_game_stars': [1, 2, 3],        # 早期偏好使用的星级
            'late_game_stars': [4, 5],            # 后期保留的星级
            'adaptive_threshold': 5,               # 棋盘卡牌数适应阈值
            'high_star_preservation': 0.7,        # 高星级保留倾向
            'strategic_timing': True,              # 战略性时机选择
        }
    },
    
    # 选拔规则专用配置
    'draft_mode': {
        'total_star_limits': {
            1: 2,  # 1星卡牌2张
            2: 2,  # 2星卡牌2张
            3: 2,  # 3星卡牌2张
            4: 2,  # 4星卡牌2张
            5: 2,  # 5星卡牌2张
        },
        'strict_enforcement': True,         # 严格执行星级限制
        'intelligent_distribution': True,   # 智能分配剩余星级
        'opponent_modeling': True,          # 对手选拔建模
        'priority_stars': [3, 4, 5],       # 优先考虑的星级（高价值卡牌）
        'early_game_preference': [1, 2],   # 前期偏好的星级
        'late_game_preference': [4, 5],    # 后期偏好的星级
    },
}

# 调试和性能配置
DEBUG_CONFIG = {
    'verbose_sampling': False,      # 是否输出详细采样信息
    'log_statistics': False,        # 是否记录统计信息
    'performance_monitoring': True, # 是否监控性能
}

# 快速访问函数
def get_sampling_config():
    """获取采样配置"""
    return SAMPLING_CONFIG

def get_debug_config():
    """获取调试配置"""
    return DEBUG_CONFIG

def get_rule_config(rule_name):
    """获取特定规则的配置"""
    return SAMPLING_CONFIG['rule_specific'].get(rule_name, {})

def get_star_distribution():
    """获取星级分布"""
    return SAMPLING_CONFIG['star_distribution']

def get_max_cards_per_unknown():
    """获取每个未知手牌的最大生成数量"""
    return SAMPLING_CONFIG['max_unknown_cards_per_hand'] 