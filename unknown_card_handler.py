#!/usr/bin/env python3
"""
智能未知牌处理模块
根据游戏规则、已知信息和统计分析来合理处理未知手牌
"""

import random
import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from card import Card
from collections import defaultdict, Counter
from unknown_card_config import get_sampling_config, get_debug_config, get_max_cards_per_unknown


class UnknownCardHandler:
    """
    智能未知牌处理器
    """
    
    def __init__(self, all_cards: List[Card], card_type_map: Dict[int, str], card_star_map: Dict[int, int]):
        self.all_cards = all_cards
        self.card_type_map = card_type_map
        self.card_star_map = card_star_map
        
        # 预计算统计信息
        self._compute_card_statistics()
    
    def _compute_card_statistics(self):
        """预计算卡牌统计信息"""
        self.cards_by_star = defaultdict(list)
        self.cards_by_type = defaultdict(list)
        self.value_distribution = defaultdict(int)
        
        for card in self.all_cards:
            star = self.card_star_map.get(card.card_id, 1)
            card_type = self.card_type_map.get(card.card_id)
            
            self.cards_by_star[star].append(card)
            if card_type:
                self.cards_by_type[card_type].append(card)
            
            # 统计数值分布
            for value in [card.up, card.right, card.down, card.left]:
                self.value_distribution[value] += 1
    
    def generate_unknown_cards(self, 
                             count: int,
                             rules: List[str], 
                             used_cards: Set[int],
                             board_state: Optional[object] = None,
                             known_hand: List[Card] = None,
                             owner: str = None,
                             can_use: bool = True) -> List[Card]:
        """
        根据规则和游戏状态生成未知卡牌的合理估计
        现在支持动态采样数量调整
        
        Args:
            count: 需要生成的卡牌数量
            rules: 当前游戏规则
            used_cards: 已使用的卡牌ID集合
            board_state: 当前棋盘状态
            known_hand: 已知的手牌
            owner: 卡牌所有者
            can_use: 是否可用（秩序/混乱规则）
        """
        
        # 动态调整采样数量
        config = get_sampling_config()
        max_cards = get_max_cards_per_unknown()
        
        # 性能模式下更激进地减少采样
        if config.get('performance_mode', False):
            # 根据未知卡牌数量动态调整
            if count == 1:
                # 单张未知卡牌，采样3-5张
                max_samples = min(5, max_cards)
            elif count <= 2:
                # 2张未知卡牌，每张采样4-6张
                max_samples = min(6, max_cards)
            elif count <= 3:
                # 3张未知卡牌，每张采样5-7张
                max_samples = min(7, max_cards)
            else:
                # 更多未知卡牌，每张最多采样8张
                max_samples = min(8, max_cards)
                
            # 进一步根据游戏阶段调整
            board_occupancy = self._get_board_occupancy(board_state)
            if board_occupancy > 0.6:  # 后期游戏，减少采样
                max_samples = max(3, max_samples - 2)
                
            actual_samples = min(max_samples, max_cards)
        else:
            actual_samples = min(count * config.get('fallback_sample_multiplier', 2), max_cards)
        
        print(f"Dynamic sampling: {count} unknown cards → {actual_samples} samples (performance_mode: {config.get('performance_mode', False)})")
        
        # 1. 基础过滤：排除已使用的卡牌（可选）
        if config.get('aggressive_sampling', False):
            # 激进模式：允许重复使用（对手可能有相同卡牌）
            available_cards = self.all_cards.copy()
        else:
            # 保守模式：排除已使用的卡牌
            available_cards = [card for card in self.all_cards if card.card_id not in used_cards]
            if not available_cards:
                available_cards = self.all_cards.copy()
        
        # 2. 根据规则进行智能采样
        return self._smart_sampling_by_rules(available_cards, actual_samples, rules, 
                                           board_state, known_hand, owner, can_use)
    
    def _get_board_occupancy(self, board_state) -> float:
        """计算棋盘占用率"""
        if not board_state:
            return 0.0
        
        occupied = 0
        for r in range(3):
            for c in range(3):
                if board_state.get_card(r, c) is not None:
                    occupied += 1
        return occupied / 9.0
    
    def _smart_sampling_by_rules(self, available_cards: List[Card], max_samples: int,
                                rules: List[str], board_state, known_hand: List[Card],
                                owner: str, can_use: bool) -> List[Card]:
        """根据规则进行智能采样"""
        # 优先处理连携类规则（同数、加算）
        if '同数' in rules:
            return self._sample_for_same_number_rule(available_cards, max_samples, board_state, 
                                                    known_hand, owner, can_use)
        elif '加算' in rules:
            return self._sample_for_addition_rule(available_cards, max_samples, board_state,
                                                 known_hand, owner, can_use)
        elif '同类强化' in rules or '同类弱化' in rules:
            return self._sample_for_same_type_rules(available_cards, max_samples, board_state, 
                                                   known_hand, owner, can_use)
        elif '逆转' in rules:
            return self._sample_for_reverse_rule(available_cards, max_samples, owner, can_use)
        elif '王牌杀手' in rules:
            return self._sample_for_ace_killer_rule(available_cards, max_samples, owner, can_use)
        else:
            return self._sample_balanced_cards(available_cards, max_samples, owner, can_use)
    
    def generate_opponent_cards(self, 
                              count: int,
                              rules: List[str], 
                              used_cards: Set[int],
                              board_state: Optional[object] = None,
                              known_hand: List[Card] = None,
                              owner: str = None,
                              can_use: bool = True) -> List[Card]:
        """
        专门为对手生成未知卡牌，考虑真实玩家的策略行为
        现在支持动态采样数量调整
        
        Args:
            count: 需要生成的卡牌数量
            rules: 当前游戏规则
            used_cards: 已使用的卡牌ID集合
            board_state: 当前棋盘状态
            known_hand: 已知的手牌
            owner: 卡牌所有者
            can_use: 是否可用（秩序/混乱规则）
        """
        
        # 动态调整对手卡牌采样数量（通常比己方卡牌采样更少）
        config = get_sampling_config()
        max_cards = get_max_cards_per_unknown()
        
        if config.get('performance_mode', False):
            # 对手卡牌采样更加保守
            if count == 1:
                max_samples = min(4, max_cards)  # 单张对手卡牌采样4张
            elif count <= 2:
                max_samples = min(5, max_cards)  # 2张对手卡牌每张采样5张
            else:
                max_samples = min(6, max_cards)  # 更多对手卡牌每张最多采样6张
                
            # 对手卡牌根据游戏阶段更激进地减少采样
            board_occupancy = self._get_board_occupancy(board_state)
            if board_occupancy > 0.5:  # 中期开始就减少对手卡牌采样
                max_samples = max(3, max_samples - 1)
                
            actual_samples = min(max_samples, max_cards)
        else:
            actual_samples = min(count * config.get('fallback_sample_multiplier', 2), max_cards)
        
        print(f"Opponent sampling: {count} unknown cards → {actual_samples} samples")
        
        # 全部卡牌可用（对手可能有重复卡牌）
        available_cards = self.all_cards.copy()
        
        # 基于规则的对手行为建模
        if not config.get('advanced', {}).get('opponent_behavior_modeling', True):
            # 如果未启用行为建模，使用简化的智能采样
            return self._smart_sampling_by_rules(available_cards, actual_samples, rules, 
                                               board_state, known_hand, owner, can_use)
        
        # 分析当前游戏局势，选择最符合对手策略的采样方法
        primary_rule = self._determine_primary_rule(rules)
        
        if primary_rule == '同数':
            return self._sample_strategic_same_number_cards(available_cards, actual_samples, board_state, 
                                                          known_hand, owner, can_use, rules)
        elif primary_rule == '加算':
            return self._sample_strategic_addition_cards(available_cards, actual_samples, board_state,
                                                        known_hand, owner, can_use, rules)
        elif primary_rule in ['同类强化', '同类弱化']:
            return self._sample_strategic_same_type_cards(available_cards, actual_samples, board_state,
                                                        known_hand, owner, can_use, rules)
        elif primary_rule == '逆转':
            return self._sample_strategic_reverse_cards(available_cards, actual_samples, owner, can_use)
        elif primary_rule == '王牌杀手':
            return self._sample_strategic_ace_killer_cards(available_cards, actual_samples, owner, can_use)
        else:
            return self._sample_strategic_balanced_cards(available_cards, actual_samples, owner, can_use)
    
    def _determine_primary_rule(self, rules: List[str]) -> str:
        """确定主要规则，用于决定对手策略"""
        # 规则优先级（连携类规则优先级最高）
        rule_priority = {
            '同数': 10,      # 最高优先级（连携）
            '加算': 9,       # 次高优先级（连携）
            '同类强化': 8,
            '同类弱化': 7,
            '逆转': 6,
            '王牌杀手': 5,
            '秩序': 3,
            '混乱': 2
        }
        
        applicable_rules = [(rule, rule_priority.get(rule, 1)) for rule in rules if rule in rule_priority]
        
        if applicable_rules:
            # 返回优先级最高的规则
            return max(applicable_rules, key=lambda x: x[1])[0]
        else:
            return 'balanced'  # 默认平衡策略
    
    def _sample_strategic_same_number_cards(self, available_cards: List[Card], count: int,
                                          board_state, known_hand: List[Card],
                                          owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """战略性同数卡牌采样（高级对手行为）"""
        config = get_sampling_config()
        behavior_config = config.get('opponent_behavior', {}).get('同数', {})
        
        # 更智能的同数策略分析
        trap_setup_cards = []
        counter_cards = []
        defensive_cards = []
        
        for card in available_cards:
            card_values = [card.up, card.right, card.down, card.left]
            
            # 检查是否适合设置连携陷阱
            if self._is_excellent_trap_card(card, board_state, behavior_config):
                trap_setup_cards.append(card)
            elif self._is_counter_play_card(card, board_state, known_hand):
                counter_cards.append(card)
            elif self._is_defensive_card(card, board_state):
                defensive_cards.append(card)
        
        result = []
        
        # 60% 陷阱设置卡牌（对手更倾向于主动设置连携）
        trap_count = max(1, int(count * 0.6))
        if trap_setup_cards:
            result.extend(self._sample_cards_from_pool(trap_setup_cards, 
                                                      min(trap_count, len(trap_setup_cards)), 
                                                      owner, can_use, is_prediction=True))
        
        # 25% 反制卡牌
        remaining = count - len(result)
        counter_count = max(1, int(remaining * 0.4)) if remaining > 0 else 0
        if counter_cards and counter_count > 0:
            result.extend(self._sample_cards_from_pool(counter_cards,
                                                      min(counter_count, len(counter_cards)),
                                                      owner, can_use, is_prediction=True))
        
        # 剩余用防御/随机卡牌补足
        remaining = count - len(result)
        if remaining > 0:
            all_remaining = [card for card in available_cards 
                           if not any(card.card_id == r.card_id for r in result)]
            if all_remaining:
                result.extend(self._sample_cards_from_pool(all_remaining, remaining, owner, can_use, is_prediction=True))
        
        return result[:count]
    
    def _sample_strategic_addition_cards(self, available_cards: List[Card], count: int,
                                       board_state, known_hand: List[Card],
                                       owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """战略性加算卡牌采样（高级对手行为）"""
        # 类似同数，但重点关注加算组合
        return self._sample_for_addition_rule(available_cards, count, board_state, 
                                            known_hand, owner, can_use, is_opponent=True)
    
    def _sample_strategic_same_type_cards(self, available_cards: List[Card], count: int,
                                        board_state, known_hand: List[Card],
                                        owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """战略性同类卡牌采样（考虑雪球/破坏策略）"""
        return self._sample_for_same_type_rules(available_cards, count, board_state,
                                              known_hand, owner, can_use, is_opponent=True)
    
    def _sample_strategic_reverse_cards(self, available_cards: List[Card], count: int,
                                      owner: str, can_use: bool) -> List[Card]:
        """战略性逆转卡牌采样"""
        return self._enhanced_reverse_sampling(available_cards, count, owner, can_use)
    
    def _sample_strategic_ace_killer_cards(self, available_cards: List[Card], count: int,
                                         owner: str, can_use: bool) -> List[Card]:
        """战略性王牌杀手卡牌采样"""
        config = get_sampling_config()
        behavior_config = config.get('opponent_behavior', {}).get('王牌杀手', {})
        
        ace_killer_cards = []
        mid_value_cards = []
        other_cards = []
        
        ace_preference = behavior_config.get('ace_killer_preference', 1.8)
        mid_preference = behavior_config.get('mid_value_preference', 0.7)
        
        for card in available_cards:
            values = [card.up, card.right, card.down, card.left]
            has_ace_killer = 1 in values or 10 in values
            
            if has_ace_killer:
                # 根据偏好系数重复添加
                ace_killer_cards.extend([card] * int(ace_preference * 10))
            elif any(4 <= val <= 7 for val in values):
                mid_value_cards.extend([card] * int(mid_preference * 10))
            else:
                other_cards.append(card)
        
        # 混合采样
        all_weighted = ace_killer_cards + mid_value_cards + other_cards
        if len(all_weighted) >= count:
            sampled = random.sample(all_weighted, count)
            # 去重
            unique_cards = []
            seen_ids = set()
            for card in sampled:
                if card.card_id not in seen_ids:
                    unique_cards.append(card)
                    seen_ids.add(card.card_id)
                if len(unique_cards) >= count:
                    break
            
            return self._sample_cards_from_pool(unique_cards[:count], count, owner, can_use, is_prediction=True)
        else:
            return self._sample_cards_from_pool(available_cards[:count], count, owner, can_use, is_prediction=True)
    
    def _sample_strategic_balanced_cards(self, available_cards: List[Card], count: int,
                                       owner: str, can_use: bool) -> List[Card]:
        """战略性平衡卡牌采样（默认策略）"""
        return self._sample_balanced_cards(available_cards, count, owner, can_use, is_opponent=True)
    
    def _is_excellent_trap_card(self, card: Card, board_state, behavior_config: Dict) -> bool:
        """判断卡牌是否为优秀的陷阱设置卡牌"""
        # 除了基础的同数判断，还考虑位置策略
        is_basic_good = self._is_good_for_same_number_combo(card, board_state, behavior_config)
        
        if not is_basic_good:
            return False
        
        # 进一步检查：是否有多个相同数值（更容易触发连携）
        card_values = [card.up, card.right, card.down, card.left]
        value_counts = {}
        for val in card_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        
        # 有2个或以上相同数值的卡牌更适合设置陷阱
        max_count = max(value_counts.values())
        return max_count >= 2
    
    def _is_counter_play_card(self, card: Card, board_state, known_hand: List[Card]) -> bool:
        """判断卡牌是否适合反制对手策略"""
        # 基于已知信息判断是否能有效反制
        # 这里可以分析对手可能的下一步，选择相应的反制卡牌
        card_values = [card.up, card.right, card.down, card.left]
        avg_value = sum(card_values) / 4
        
        # 中等偏高数值适合反制
        return 5.5 <= avg_value <= 7.5
    
    def _sample_for_same_type_rules(self, available_cards: List[Card], count: int,
                                   board_state, known_hand: List[Card], 
                                   owner: str, can_use: bool, is_opponent: bool = False) -> List[Card]:
        """为同类强化/弱化规则采样卡牌"""
        result = []
        
        # 分析棋盘和已知手牌的类型分布
        type_priority = self._analyze_type_priority(board_state, known_hand)
        
        # 30%概率选择优先类型，40%概率选择平衡分布，30%概率随机
        priority_count = max(1, int(count * 0.3))
        balanced_count = max(1, int(count * 0.4))
        random_count = count - priority_count - balanced_count
        
        # 优先类型采样
        if type_priority and priority_count > 0:
            priority_type = type_priority[0]
            type_cards = [card for card in available_cards 
                         if self.card_type_map.get(card.card_id) == priority_type]
            if type_cards:
                result.extend(self._sample_cards_from_pool(type_cards, priority_count, owner, can_use, is_prediction=is_opponent))
        
        # 平衡分布采样
        if balanced_count > 0:
            balanced_cards = self._get_balanced_type_sample(available_cards, balanced_count)
            result.extend(self._sample_cards_from_pool(balanced_cards, balanced_count, owner, can_use, is_prediction=is_opponent))
        
        # 随机采样补足
        remaining = count - len(result)
        if remaining > 0:
            remaining_cards = [card for card in available_cards if not any(
                card.card_id == r.card_id for r in result)]
            if remaining_cards:
                result.extend(self._sample_cards_from_pool(remaining_cards, remaining, owner, can_use, is_prediction=is_opponent))
        
        return result[:count]
    
    def _sample_for_same_number_rule(self, available_cards: List[Card], count: int,
                                    board_state, known_hand: List[Card],
                                    owner: str, can_use: bool, is_opponent: bool = False) -> List[Card]:
        """为同数规则采样卡牌（模拟玩家倾向于设置连携陷阱）"""
        config = get_sampling_config()
        behavior_config = config.get('opponent_behavior', {}).get('同数', {})
        
        # 分析棋盘状态，寻找可能的同数机会
        combo_cards = []
        defensive_cards = []
        
        for card in available_cards:
            card_values = [card.up, card.right, card.down, card.left]
            
            # 检查是否适合设置同数陷阱
            if self._is_good_for_same_number_combo(card, board_state, behavior_config):
                combo_cards.append(card)
            elif self._is_defensive_card(card, board_state):
                defensive_cards.append(card)
        
        result = []
        combo_count = max(1, int(count * behavior_config.get('combo_setup_ratio', 0.5)))
        defensive_count = max(1, int(count * behavior_config.get('defensive_ratio', 0.3)))
        random_count = count - combo_count - defensive_count
        
        # 连携设置卡牌
        if combo_cards and combo_count > 0:
            result.extend(self._sample_cards_from_pool(combo_cards, combo_count, owner, can_use, is_prediction=is_opponent))
        
        # 防御性卡牌
        remaining = count - len(result)
        if defensive_cards and defensive_count > 0 and remaining > 0:
            actual_defensive = min(defensive_count, remaining)
            result.extend(self._sample_cards_from_pool(defensive_cards, actual_defensive, owner, can_use, is_prediction=is_opponent))
        
        # 随机补足
        remaining = count - len(result)
        if remaining > 0:
            remaining_cards = [card for card in available_cards 
                             if not any(card.card_id == r.card_id for r in result)]
            if remaining_cards:
                result.extend(self._sample_cards_from_pool(remaining_cards, remaining, owner, can_use, is_prediction=is_opponent))
        
        return result[:count]
    
    def _sample_for_addition_rule(self, available_cards: List[Card], count: int,
                                 board_state, known_hand: List[Card],
                                 owner: str, can_use: bool, is_opponent: bool = False) -> List[Card]:
        """为加算规则采样卡牌（模拟玩家倾向于设置加算连携）"""
        config = get_sampling_config()
        behavior_config = config.get('opponent_behavior', {}).get('加算', {})
        
        # 分析可能的加算组合
        sum_combo_cards = []
        defensive_cards = []
        
        for card in available_cards:
            if self._is_good_for_addition_combo(card, board_state, behavior_config):
                sum_combo_cards.append(card)
            elif self._is_defensive_card(card, board_state):
                defensive_cards.append(card)
        
        result = []
        combo_count = max(1, int(count * behavior_config.get('sum_combo_ratio', 0.5)))
        defensive_count = max(1, int(count * behavior_config.get('defensive_ratio', 0.3)))
        
        # 加算连携卡牌
        if sum_combo_cards and combo_count > 0:
            result.extend(self._sample_cards_from_pool(sum_combo_cards, combo_count, owner, can_use, is_prediction=is_opponent))
        
        # 防御性卡牌
        remaining = count - len(result)
        if defensive_cards and defensive_count > 0 and remaining > 0:
            actual_defensive = min(defensive_count, remaining)
            result.extend(self._sample_cards_from_pool(defensive_cards, actual_defensive, owner, can_use, is_prediction=is_opponent))
        
        # 随机补足
        remaining = count - len(result)
        if remaining > 0:
            remaining_cards = [card for card in available_cards 
                             if not any(card.card_id == r.card_id for r in result)]
            if remaining_cards:
                result.extend(self._sample_cards_from_pool(remaining_cards, remaining, owner, can_use, is_prediction=is_opponent))
        
        return result[:count]
    
    def _sample_for_reverse_rule(self, available_cards: List[Card], count: int,
                                owner: str, can_use: bool) -> List[Card]:
        """为逆转规则采样卡牌（偏好低数值，考虑对手行为）"""
        # 使用增强的逆转采样，考虑对手的真实行为模式
        return self._enhanced_reverse_sampling(available_cards, count, owner, can_use)
    
    def _sample_for_ace_killer_rule(self, available_cards: List[Card], count: int,
                                   owner: str, can_use: bool) -> List[Card]:
        """为王牌杀手规则采样卡牌（偏好1和A，考虑对手策略）"""
        # 直接使用战略性王牌杀手采样
        return self._sample_strategic_ace_killer_cards(available_cards, count, owner, can_use)
    
    def _sample_balanced_cards(self, available_cards: List[Card], count: int,
                              owner: str, can_use: bool, is_opponent: bool = False) -> List[Card]:
        """平衡采样（默认策略）"""
        # 按星级分层采样
        star_distribution = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.08, 5: 0.02}
        result = []
        
        for star, ratio in star_distribution.items():
            star_count = max(1, int(count * ratio))
            star_cards = [card for card in available_cards 
                         if self.card_star_map.get(card.card_id, 1) == star]
            
            if star_cards and star_count > 0:
                sampled = self._sample_cards_from_pool(star_cards, star_count, owner, can_use, is_prediction=is_opponent)
                result.extend(sampled)
                
                # 从available_cards中移除已采样的卡牌
                sampled_ids = {card.card_id for card in sampled}
                available_cards = [card for card in available_cards 
                                 if card.card_id not in sampled_ids]
        
        # 补足到目标数量
        remaining = count - len(result)
        if remaining > 0 and available_cards:
            result.extend(self._sample_cards_from_pool(available_cards, remaining, owner, can_use, is_prediction=is_opponent))
        
        return result[:count]
    
    def _sample_cards_from_pool(self, card_pool: List[Card], count: int,
                               owner: str, can_use: bool, is_prediction: bool = False, 
                               is_opponent: bool = False) -> List[Card]:
        """从卡牌池中采样指定数量的卡牌"""
        if not card_pool:
            return []
        
        # 确保不超过池子大小
        count = min(count, len(card_pool))
        
        # 随机采样（不重复）
        sampled_cards = random.sample(card_pool, count)
        
        # 创建新的Card实例，设置正确的owner和can_use
        result = []
        for i, card in enumerate(sampled_cards):
            # 对于对手预测卡牌，使用特殊ID标记（>= 1000）
            card_id = card.card_id
            if is_opponent or is_prediction:
                card_id = 1000 + (card.card_id % 1000)  # 确保ID >= 1000
            
            new_card = Card(
                up=card.up,
                right=card.right,
                down=card.down,
                left=card.left,
                owner=owner,
                card_id=card_id,
                card_type=card.card_type,
                can_use=can_use
            )
            # 标记为生成的卡牌（用于区分真实已知和AI预测）
            if is_opponent or is_prediction:
                new_card._is_generated = True
                new_card._is_prediction = True
            result.append(new_card)
        
        return result
    
    def _analyze_type_priority(self, board_state, known_hand: List[Card]) -> List[str]:
        """分析类型优先级"""
        type_counts = defaultdict(int)
        
        # 分析棋盘上的类型
        if board_state:
            for r in range(3):
                for c in range(3):
                    card = board_state.get_card(r, c)
                    if card and card.card_type:
                        type_counts[card.card_type] += 2  # 棋盘上的卡牌权重更高
        
        # 分析已知手牌的类型
        if known_hand:
            for card in known_hand:
                if card.card_type:
                    type_counts[card.card_type] += 1
        
        # 按出现频率排序
        return sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)
    
    def _get_balanced_type_sample(self, available_cards: List[Card], count: int) -> List[Card]:
        """获取类型平衡的卡牌样本"""
        type_groups = defaultdict(list)
        
        for card in available_cards:
            card_type = self.card_type_map.get(card.card_id, 'no_type')
            type_groups[card_type].append(card)
        
        result = []
        types = list(type_groups.keys())
        
        # 轮流从每个类型中选择卡牌
        for i in range(count):
            type_name = types[i % len(types)]
            if type_groups[type_name]:
                card = random.choice(type_groups[type_name])
                result.append(card)
                type_groups[type_name].remove(card)
        
        return result
    
    def _is_good_for_same_number_combo(self, card: Card, board_state, behavior_config: Dict) -> bool:
        """判断卡牌是否适合同数连携"""
        card_values = [card.up, card.right, card.down, card.left]
        
        # 偏好中等数值，容易形成同数
        preferred_values = behavior_config.get('preferred_values', [2, 3, 4, 5, 6])
        
        # 检查是否有偏好数值
        has_preferred = any(val in preferred_values for val in card_values)
        
        # 检查是否有重复数值（利于同数）
        has_duplicates = len(set(card_values)) < 4
        
        # 避免极端数值
        avoid_extreme = behavior_config.get('avoid_extreme_values', True)
        if avoid_extreme:
            has_extreme = 1 in card_values or 10 in card_values
            if has_extreme:
                return False
        
        return has_preferred or has_duplicates
    
    def _is_good_for_addition_combo(self, card: Card, board_state, behavior_config: Dict) -> bool:
        """判断卡牌是否适合加算连携"""
        card_values = [card.up, card.right, card.down, card.left]
        
        # 检查是否有利于形成特定和数的组合
        preferred_ranges = behavior_config.get('preferred_sum_ranges', [(5, 9), (10, 14)])
        
        # 计算可能的和数
        possible_sums = []
        for i, val1 in enumerate(card_values):
            for j, val2 in enumerate(card_values):
                if i != j:
                    possible_sums.append(val1 + val2)
        
        # 检查是否在偏好范围内
        for sum_val in possible_sums:
            for min_range, max_range in preferred_ranges:
                if min_range <= sum_val <= max_range:
                    return True
        
        # 检查是否有互补数值（如3+7=10, 4+6=10等）
        complementary = behavior_config.get('complementary_values', True)
        if complementary:
            common_sums = [8, 10, 12]  # 常见的目标和数
            for target_sum in common_sums:
                for val in card_values:
                    complement = target_sum - val
                    if complement in card_values and complement != val:
                        return True
        
        return False
    
    def _is_defensive_card(self, card: Card, board_state) -> bool:
        """判断卡牌是否适合防御策略"""
        # 高数值卡牌通常更适合防御
        card_values = [card.up, card.right, card.down, card.left]
        avg_value = sum(card_values) / 4
        
        # 平均值较高的卡牌更适合防御
        return avg_value >= 6.0
    
    def _enhanced_reverse_sampling(self, available_cards: List[Card], count: int, 
                                  owner: str, can_use: bool) -> List[Card]:
        """增强的逆转规则采样，考虑对手行为"""
        config = get_sampling_config()
        behavior_config = config.get('opponent_behavior', {}).get('逆转', {})
        
        # 按照对手偏好进行权重计算
        low_preference = behavior_config.get('low_value_preference', 2.0)
        max_preferred = behavior_config.get('max_preferred_value', 5)
        high_avoidance = behavior_config.get('high_value_avoidance', 0.3)
        
        weighted_cards = []
        for card in available_cards:
            card_values = [card.up, card.right, card.down, card.left]
            avg_value = sum(card_values) / 4
            
            # 计算权重
            if avg_value <= max_preferred:
                weight = low_preference
            elif avg_value >= 8:
                weight = high_avoidance
            else:
                weight = 1.0
            
            weighted_cards.extend([card] * int(weight * 10))  # 权重转换为重复次数
        
        # 从加权列表中采样
        if len(weighted_cards) >= count:
            sampled = random.sample(weighted_cards, count)
            # 去重
            unique_cards = []
            seen_ids = set()
            for card in sampled:
                if card.card_id not in seen_ids:
                    unique_cards.append(card)
                    seen_ids.add(card.card_id)
                if len(unique_cards) >= count:
                    break
            
            return self._sample_cards_from_pool(unique_cards[:count], count, owner, can_use, is_prediction=True)
        else:
            return self._sample_cards_from_pool(available_cards[:count], count, owner, can_use, is_prediction=True)


# 全局处理器实例
_unknown_card_handler = None

def get_unknown_card_handler():
    """获取全局未知卡牌处理器"""
    global _unknown_card_handler
    if _unknown_card_handler is None:
        # 这里需要导入相关模块，但为了避免循环导入，我们在使用时初始化
        pass
    return _unknown_card_handler

def initialize_unknown_card_handler(all_cards, card_type_map, card_star_map):
    """初始化全局未知卡牌处理器"""
    global _unknown_card_handler
    _unknown_card_handler = UnknownCardHandler(all_cards, card_type_map, card_star_map)
    return _unknown_card_handler 