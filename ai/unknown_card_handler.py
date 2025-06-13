#!/usr/bin/env python3
"""
智能未知牌处理模块
根据游戏规则、已知信息和统计分析来合理处理未知手牌
"""

import random
import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from core.card import Card
from collections import defaultdict, Counter
from config.unknown_card_config import get_sampling_config, get_debug_config, get_max_cards_per_unknown


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
        
        # 动态调整采样数量 - 选拔模式下使用精确采样
        config = get_sampling_config()
        max_cards = get_max_cards_per_unknown()
        
        # 选拔模式下使用精确采样，不生成额外卡牌
        if '选拔' in rules:
            actual_samples = count
            print(f"选拔模式使用精确采样: {count} unknown cards → {actual_samples} samples")
        elif config.get('performance_mode', False):
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
            print(f"Dynamic sampling: {count} unknown cards → {actual_samples} samples (performance_mode: {config.get('performance_mode', False)})")
        else:
            actual_samples = min(count * config.get('fallback_sample_multiplier', 2), max_cards)
            print(f"Standard sampling: {count} unknown cards → {actual_samples} samples")
        
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
        # 选拔规则优先级最高（影响所有采样）
        if '选拔' in rules:
            return self._sample_for_draft_rule(available_cards, max_samples, board_state,
                                             known_hand, owner, can_use, rules)
        # 优先处理连携类规则（同数、加算）
        elif '同数' in rules:
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
        
        if primary_rule == '选拔':
            return self._sample_strategic_draft_cards(available_cards, actual_samples, board_state,
                                                     known_hand, owner, can_use, rules)
        elif primary_rule == '同数':
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
        # 规则优先级（选拔规则优先级最高，连携类规则次之）
        rule_priority = {
            '选拔': 15,      # 最高优先级（影响所有采样）
            '同数': 10,      # 连携类规则优先级高
            '加算': 9,       # 连携类规则优先级高
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
    
    def _sample_strategic_draft_cards(self, available_cards: List[Card], count: int,
                                     board_state, known_hand: List[Card],
                                     owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """战略性选拔卡牌采样（对手行为建模）"""
        # 对手在选拔模式下的策略思考：
        # 1. 优先保留高星级卡牌到关键时刻
        # 2. 早期使用中低星级卡牌
        # 3. 根据剩余星级配额调整策略
        return self._sample_for_draft_rule(available_cards, count, board_state,
                                         known_hand, owner, can_use, rules)
    
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
    
    def _sample_for_draft_rule(self, available_cards: List[Card], count: int,
                              board_state, known_hand: List[Card],
                              owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """
        为选拔规则进行智能采样 - 不限制采样广度，保持不对称博弈的预测能力
        
        Args:
            available_cards: 可用卡牌池
            count: 需要采样的数量
            board_state: 棋盘状态
            known_hand: 已知手牌
            owner: 卡牌所有者
            can_use: 是否可用
            rules: 所有规则列表
        """
        config = get_sampling_config()
        draft_config = config.get('draft_mode', {})
        
        # 分析当前星级使用情况（用于评估，但不强制限制）
        star_usage = self._analyze_star_usage(board_state, known_hand)
        available_stars = self._calculate_available_stars(star_usage, draft_config)
        
        print(f"选拔模式星级分析: 已使用={star_usage}, 可用={available_stars}")
        
        # 在选拔模式下，不限制采样，而是提供更广泛的预测
        # 但是会根据星级约束给予不同的权重
        weighted_candidates = []
        
        for card in available_cards:
            star_level = self.card_star_map.get(card.card_id, 1)
            base_weight = 1.0
            
            # 根据星级约束调整权重，但不完全排除
            if available_stars.get(star_level, 0) > 0:
                # 还有配额的星级，给予更高权重
                base_weight = 2.0
            elif available_stars.get(star_level, 0) == 0:
                # 配额已满的星级，给予较低权重但不排除
                base_weight = 0.3
            
            # 根据当前规则调整权重
            rule_score = self._calculate_card_rule_score(card, rules, board_state)
            
            # 新增：边角战略评分
            corner_score = self._calculate_corner_strategy_score(card, board_state)
            
            final_weight = base_weight * (1.0 + rule_score * 0.3 + corner_score * 0.4)
            
            # 添加到加权候选池中
            weighted_candidates.extend([card] * max(1, int(final_weight * 10)))
        
        # 进行加权随机采样
        if len(weighted_candidates) < count:
            # 如果加权候选不足，使用所有候选
            return self._sample_cards_from_pool(available_cards, min(count, len(available_cards)), 
                                               owner, can_use, is_prediction=True)
        else:
            # 从加权池中采样
            sampled = random.sample(weighted_candidates, count)
            # 去重
            unique_cards = []
            seen_ids = set()
            for card in sampled:
                if card.card_id not in seen_ids:
                    unique_cards.append(card)
                    seen_ids.add(card.card_id)
                if len(unique_cards) >= count:
                    break
            
            # 如果去重后不足，补充更多卡牌
            if len(unique_cards) < count:
                remaining_cards = [card for card in available_cards 
                                 if card.card_id not in seen_ids]
                additional = min(count - len(unique_cards), len(remaining_cards))
                if additional > 0:
                    unique_cards.extend(random.sample(remaining_cards, additional))
            
            return self._sample_cards_from_pool(unique_cards[:count], count, 
                                               owner, can_use, is_prediction=True)
    
    def _analyze_star_usage(self, board_state, known_hand: List[Card]) -> Dict[int, int]:
        """分析当前星级使用情况"""
        star_usage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # 统计棋盘上的卡牌星级
        if board_state:
            for r in range(3):
                for c in range(3):
                    card = board_state.get_card(r, c)
                    if card and card.card_id and card.card_id in self.card_star_map:
                        star = self.card_star_map.get(card.card_id, 1)
                        star_usage[star] += 1
        
        # 统计已知手牌的星级（包括己方和对手的已知卡牌）
        if known_hand:
            for card in known_hand:
                if (card.card_id and card.card_id > 0 and card.card_id in self.card_star_map and
                    not (card.up == 0 and card.right == 0 and card.down == 0 and card.left == 0)):  # 排除未知卡牌
                    star = self.card_star_map.get(card.card_id, 1)
                    star_usage[star] += 1
        
        return star_usage
    
    def _calculate_available_stars(self, star_usage: Dict[int, int], 
                                 draft_config: Dict) -> Dict[int, int]:
        """计算每个星级还能使用的数量"""
        total_limits = draft_config.get('total_star_limits', {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        available_stars = {}
        
        for star, limit in total_limits.items():
            used = star_usage.get(star, 0)
            available = max(0, limit - used)
            available_stars[star] = available
            
        return available_stars
    
    def _intelligent_star_distribution(self, available_cards: List[Card], count: int,
                                     available_stars: Dict[int, int], owner: str, 
                                     can_use: bool, rules: List[str], board_state) -> List[Card]:
        """智能星级分配策略"""
        config = get_sampling_config()
        draft_config = config.get('draft_mode', {})
        
        # 按星级分组可用卡牌
        cards_by_star = {}
        for card in available_cards:
            star = self.card_star_map.get(card.card_id, 1)
            if available_stars.get(star, 0) > 0:  # 只考虑还有配额的星级
                if star not in cards_by_star:
                    cards_by_star[star] = []
                cards_by_star[star].append(card)
        
        if not cards_by_star:
            print("没有符合星级限制的卡牌，使用回退策略")
            return self._draft_fallback_sampling(available_cards, count, owner, can_use, rules)
        
        result = []
        remaining_count = count
        
        # 第一步：优先分配高价值星级（4星、5星）
        priority_stars = draft_config.get('priority_stars', [3, 4, 5])
        for star in sorted(priority_stars, reverse=True):  # 从高到低
            if star in cards_by_star and available_stars.get(star, 0) > 0 and remaining_count > 0:
                # 计算这个星级应该分配多少张
                star_allocation = min(
                    remaining_count,
                    available_stars[star],
                    len(cards_by_star[star]),
                    max(1, remaining_count // 3)  # 至少分配1张，最多分配1/3
                )
                
                # 从该星级卡牌中智能选择
                star_cards = self._select_best_cards_for_rules(cards_by_star[star], 
                                                             star_allocation, rules, board_state)
                selected = self._sample_cards_from_pool(star_cards, star_allocation, 
                                                       owner, can_use, is_prediction=True)
                result.extend(selected)
                remaining_count -= len(selected)
                available_stars[star] -= len(selected)
                
                # 从cards_by_star中移除已选择的卡牌
                selected_ids = {card.card_id for card in selected}
                cards_by_star[star] = [card for card in cards_by_star[star] 
                                     if card.card_id not in selected_ids]
        
        # 第二步：分配中等星级（2星、3星）
        mid_stars = [2, 3]
        for star in mid_stars:
            if star in cards_by_star and available_stars.get(star, 0) > 0 and remaining_count > 0:
                star_allocation = min(remaining_count, available_stars[star], len(cards_by_star[star]))
                star_cards = self._select_best_cards_for_rules(cards_by_star[star], 
                                                             star_allocation, rules, board_state)
                selected = self._sample_cards_from_pool(star_cards, star_allocation, 
                                                       owner, can_use, is_prediction=True)
                result.extend(selected)
                remaining_count -= len(selected)
                available_stars[star] -= len(selected)
                
                # 从cards_by_star中移除已选择的卡牌
                selected_ids = {card.card_id for card in selected}
                cards_by_star[star] = [card for card in cards_by_star[star] 
                                     if card.card_id not in selected_ids]
        
        # 第三步：用低星级补足
        if remaining_count > 0 and 1 in cards_by_star and available_stars.get(1, 0) > 0:
            star_allocation = min(remaining_count, available_stars[1], len(cards_by_star[1]))
            star_cards = cards_by_star[1]
            selected = self._sample_cards_from_pool(star_cards, star_allocation, 
                                                   owner, can_use, is_prediction=True)
            result.extend(selected)
            remaining_count -= len(selected)
        
        # 如果还是不够，随机补足（理论上不应该发生）
        if remaining_count > 0:
            print(f"选拔模式警告: 仍需补足{remaining_count}张卡牌")
            fallback_cards = [card for card in available_cards 
                            if not any(card.card_id == r.card_id for r in result)]
            if fallback_cards:
                selected = self._sample_cards_from_pool(fallback_cards, remaining_count, 
                                                       owner, can_use, is_prediction=True)
                result.extend(selected)
        
        print(f"选拔模式完成: 生成{len(result)}张卡牌 (目标{count}张)")
        return result[:count]
    
    def _select_best_cards_for_rules(self, star_cards: List[Card], count: int, 
                                   rules: List[str], board_state) -> List[Card]:
        """根据规则选择最适合的卡牌"""
        if not star_cards:
            return []
        
        # 如果需要的数量大于等于可用数量，直接返回所有卡牌
        if count >= len(star_cards):
            return star_cards
        
        # 根据其他规则进行评分
        scored_cards = []
        for card in star_cards:
            score = self._calculate_card_rule_score(card, rules, board_state)
            scored_cards.append((card, score))
        
        # 按分数排序，选择最佳的卡牌
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        return [card for card, _ in scored_cards[:count]]
    
    def _calculate_card_rule_score(self, card: Card, rules: List[str], board_state) -> float:
        """计算卡牌在当前规则下的评分"""
        score = 0.0
        
        # 基础数值评分
        avg_value = (card.up + card.right + card.down + card.left) / 4
        score += avg_value * 0.1
        
        # 根据规则调整评分
        if '同数' in rules:
            # 有重复数值的卡牌加分
            values = [card.up, card.right, card.down, card.left]
            if len(set(values)) < 4:
                score += 2.0
        
        if '加算' in rules:
            # 有常见和数的卡牌加分
            values = [card.up, card.right, card.down, card.left]
            for i, v1 in enumerate(values):
                for j, v2 in enumerate(values):
                    if i != j and v1 + v2 in [8, 10, 12]:
                        score += 1.5
        
        if '逆转' in rules:
            # 低数值卡牌加分
            if avg_value <= 5:
                score += 3.0
            elif avg_value >= 8:
                score -= 2.0
        
        if '王牌杀手' in rules:
            # 含1或A的卡牌加分
            values = [card.up, card.right, card.down, card.left]
            ace_count = sum(1 for v in values if v in [1, 10])
            score += ace_count * 2.0
        
        if '同类强化' in rules and card.card_type:
            # 有类型的卡牌加分
            score += 1.0
        
        if '同类弱化' in rules and card.card_type:
            # 多样化类型更有价值
            score += 0.5
        
        return score
    
    def _calculate_corner_strategy_score(self, card: Card, board_state) -> float:
        """
        计算卡牌的边角放置战略评分
        高数值边应该优先占据角落位置，避免弱势边暴露
        """
        values = [card.up, card.right, card.down, card.left]  # U, R, D, L
        score = 0.0
        
        # 识别高数值边（8, 9, A/10）
        high_values = [v for v in values if v >= 8]
        medium_values = [v for v in values if 5 <= v <= 7]
        low_values = [v for v in values if v <= 4]
        
        # 基础战略评分
        if len(high_values) >= 3:
            # 三边及以上高数值 - 非常适合角落放置
            score += 5.0
            
            # 检查是否有弱势边需要保护
            if len(low_values) >= 1:
                score += 2.0  # 有弱势边需要隐藏，更适合角落
                
        elif len(high_values) >= 2:
            # 双边高数值 - 适合边角放置
            score += 3.0
            
            # 检查高数值边的位置组合
            high_positions = [i for i, v in enumerate(values) if v >= 8]
            
            # 相邻高数值边特别适合角落（如右下角：右边+下边）
            if self._are_adjacent_sides(high_positions):
                score += 2.0
                
        elif len(high_values) == 1:
            # 单边高数值
            score += 1.0
        
        # AA组合特殊处理
        ace_positions = [i for i, v in enumerate(values) if v == 10]  # A = 10
        if len(ace_positions) >= 2:
            score += 4.0
            
            # AA在相邻位置特别适合对应角落
            if self._are_adjacent_sides(ace_positions):
                score += 3.0
                
                # 检查具体的AA组合并评估最优位置
                if self._is_optimal_aa_combination(ace_positions, values):
                    score += 5.0
        
        # 分析棋盘状态，评估最优角落是否可用
        if board_state:
            corner_bonus = self._evaluate_corner_availability(card, board_state)
            score += corner_bonus
        
        # 弱势边暴露惩罚
        weakness_penalty = self._calculate_weakness_exposure_penalty(values)
        score -= weakness_penalty
        
        return score

    def _are_adjacent_sides(self, positions: List[int]) -> bool:
        """检查边的位置是否相邻"""
        if len(positions) < 2:
            return False
            
        # 棋盘边的相邻关系: 0(上)-1(右), 1(右)-2(下), 2(下)-3(左), 3(左)-0(上)
        adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                if pos2 in adjacency[pos1]:
                    return True
        return False

    def _is_optimal_aa_combination(self, ace_positions: List[int], values: List[int]) -> bool:
        """检查是否是最优的AA组合，适合直接占据最佳角落"""
        if len(ace_positions) < 2:
            return False
            
        # 右下角最优：右(1) + 下(2) = AA
        if 1 in ace_positions and 2 in ace_positions:
            return True
            
        # 左下角次优：左(3) + 下(2) = AA  
        if 3 in ace_positions and 2 in ace_positions:
            return True
            
        # 右上角：右(1) + 上(0) = AA
        if 1 in ace_positions and 0 in ace_positions:
            return True
            
        # 左上角：左(3) + 上(0) = AA
        if 3 in ace_positions and 0 in ace_positions:
            return True
            
        return False

    def _evaluate_corner_availability(self, card: Card, board_state) -> float:
        """评估角落位置的可用性和适配度"""
        if not board_state:
            return 0.0
            
        corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]  # 四个角落
        available_corners = []
        
        for pos in corner_positions:
            if not board_state.get_card(pos[0], pos[1]):  # 位置空闲
                available_corners.append(pos)
        
        if not available_corners:
            return 0.0  # 没有可用角落
            
        bonus = 0.0
        values = [card.up, card.right, card.down, card.left]
        
        # 评估每个可用角落的适配度
        for corner in available_corners:
            corner_score = self._calculate_corner_fit_score(values, corner, board_state)
            bonus += corner_score * 0.5  # 每个可用角落提供适配奖励
            
        return min(bonus, 3.0)  # 限制最大奖励

    def _calculate_corner_fit_score(self, values: List[int], corner_pos: tuple, board_state) -> float:
        """计算卡牌对特定角落位置的适配度"""
        row, col = corner_pos
        fit_score = 0.0
        
        # 检查相邻位置的威胁
        adjacent_positions = []
        if row == 0 and col == 0:  # 左上角
            adjacent_positions = [(0, 1), (1, 0)]  # 右邻、下邻
            # 卡牌的右边(1)和下边(2)数值重要
            if values[1] >= 8: fit_score += 2.0
            if values[2] >= 8: fit_score += 2.0
        elif row == 0 and col == 2:  # 右上角  
            adjacent_positions = [(0, 1), (1, 2)]  # 左邻、下邻
            # 卡牌的左边(3)和下边(2)数值重要
            if values[3] >= 8: fit_score += 2.0
            if values[2] >= 8: fit_score += 2.0
        elif row == 2 and col == 0:  # 左下角
            adjacent_positions = [(1, 0), (2, 1)]  # 上邻、右邻
            # 卡牌的上边(0)和右边(1)数值重要
            if values[0] >= 8: fit_score += 2.0
            if values[1] >= 8: fit_score += 2.0
        elif row == 2 and col == 2:  # 右下角
            adjacent_positions = [(1, 2), (2, 1)]  # 上邻、左邻
            # 卡牌的上边(0)和左边(3)数值重要
            if values[0] >= 8: fit_score += 2.0
            if values[3] >= 8: fit_score += 2.0
            
        return fit_score

    def _calculate_weakness_exposure_penalty(self, values: List[int]) -> float:
        """计算弱势边暴露的惩罚"""
        penalty = 0.0
        
        # 识别特别弱的边（1-3）
        very_weak = [v for v in values if v <= 3]
        weak = [v for v in values if 4 <= v <= 5]
        
        # 弱势边越多，越需要小心放置
        penalty += len(very_weak) * 1.5
        penalty += len(weak) * 0.8
        
        # 特殊情况：一张卡有极端差异（如1,9,9,9）
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val >= 7:  # 数值差异很大
            penalty += 2.0
            
        return penalty
    
    def _simple_star_matching(self, available_cards: List[Card], count: int,
                            available_stars: Dict[int, int], owner: str, can_use: bool) -> List[Card]:
        """简单星级匹配策略"""
        result = []
        remaining_count = count
        
        # 按星级从高到低分配
        for star in sorted(available_stars.keys(), reverse=True):
            if available_stars[star] > 0 and remaining_count > 0:
                star_cards = [card for card in available_cards 
                            if self.card_star_map.get(card.card_id, 1) == star]
                if star_cards:
                    allocation = min(remaining_count, available_stars[star], len(star_cards))
                    selected = self._sample_cards_from_pool(star_cards, allocation, 
                                                           owner, can_use, is_prediction=True)
                    result.extend(selected)
                    remaining_count -= len(selected)
        
        return result[:count]
    
    def _draft_fallback_sampling(self, available_cards: List[Card], count: int,
                                owner: str, can_use: bool, rules: List[str]) -> List[Card]:
        """选拔模式的回退采样策略"""
        print("使用选拔模式回退策略")
        
        # 移除选拔规则，使用其他规则进行采样
        fallback_rules = [rule for rule in rules if rule != '选拔']
        
        if fallback_rules:
            # 递归调用智能采样，但排除选拔规则
            return self._smart_sampling_by_rules(available_cards, count, fallback_rules, 
                                               None, [], owner, can_use)
        else:
            # 如果没有其他规则，使用平衡采样
            return self._sample_balanced_cards(available_cards, count, owner, can_use)
    
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
                card_id = 1000 + card.card_id  # 保持原始ID的映射关系
            
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