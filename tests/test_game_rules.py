import unittest
from unittest.mock import patch

from ai.ai import evaluate_move
from ai_server import _select_unknown_cards_for_slots, parse_hand, select_endgame_robust_move
from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player


class FakeHandler:
    def generate_opponent_cards(self, count, rules, used_cards, board_state=None,
                                known_hand=None, owner=None, can_use=True):
        cards = []
        for idx in range(10):
            cards.append(Card(
                up=idx + 1,
                right=idx + 2,
                down=idx + 3,
                left=idx + 4,
                owner=owner,
                card_id=2001 + idx,
                card_type="test",
                can_use=can_use,
            ))
        return cards

    def generate_unknown_cards(self, count, rules, used_cards, board_state=None,
                               known_hand=None, owner=None, can_use=True):
        return self.generate_opponent_cards(count, rules, used_cards, board_state,
                                            known_hand, owner, can_use)


class GameRuleTests(unittest.TestCase):
    def test_parse_hand_keeps_real_slot_count(self):
        hand_json = [
            {'numU': 1, 'numR': 4, 'numD': 3, 'numL': 2, 'canUse': True},
            {'numU': 0, 'numR': 0, 'numD': 0, 'numL': 0, 'canUse': False},
            {'numU': 2, 'numR': 5, 'numD': 4, 'numL': 3, 'canUse': True},
            {'numU': 0, 'numR': 0, 'numD': 0, 'numL': 0, 'canUse': True},
            {'numU': 3, 'numR': 6, 'numD': 5, 'numL': 4, 'canUse': True},
        ]

        with patch('ai_server.find_card_id_by_stats', side_effect=[101, 102, 103]), \
             patch('ai_server.get_card_type_map', return_value={101: 'type_a', 102: 'type_b', 103: 'type_c'}), \
             patch('ai_server.ensure_handler_initialized', return_value=None), \
             patch('ai_server.get_unknown_card_handler', return_value=FakeHandler()), \
             patch('ai_server.random.sample', side_effect=lambda seq, n: list(seq)[:n]):
            used_cards = set()
            hand = parse_hand(hand_json, 'blue', used_cards, rules=['同数'], board_state=Board(), is_opponent=True)

        self.assertEqual(len(hand), 5)
        self.assertEqual([card.card_id for card in hand], [101, 2001, 102, 2002, 103])
        self.assertEqual([card.can_use for card in hand], [True, False, True, True, True])

    def test_endgame_unknown_selection_prefers_immediate_addition_threat(self):
        board = Board()
        board.place_card(0, 1, Card(1, 1, 5, 1, owner='blue', card_id=1))
        board.place_card(1, 0, Card(1, 4, 1, 1, owner='blue', card_id=2))
        board.place_card(2, 2, Card(1, 1, 1, 1, owner='red', card_id=3))
        board.place_card(0, 2, Card(1, 1, 1, 1, owner='red', card_id=4))

        threat = Card(3, 1, 1, 4, owner='red', card_id=10)
        filler = Card(1, 1, 1, 1, owner='red', card_id=11)

        selected = _select_unknown_cards_for_slots(
            [filler, threat],
            1,
            board,
            ['加算'],
            'red',
            is_opponent=True,
        )

        self.assertEqual(selected[0].card_id, 10)

    def test_endgame_robust_selection_prefers_safe_corner(self):
        board = Board()
        board.place_card(0, 1, Card(7, 7, 7, 7, owner='blue', card_id=1))
        board.place_card(1, 0, Card(7, 7, 7, 7, owner='blue', card_id=2))

        my_card = Card(4, 4, 4, 4, owner='blue', card_id=10)
        weak_opp = Card(1, 1, 1, 1, owner='red', card_id=20)
        strong_opp = Card(1, 8, 8, 1, owner='red', card_id=21)

        base_state = GameState(
            board,
            [
                Player('red', [weak_opp]),
                Player('blue', [my_card]),
            ],
            current_player_idx=1,
            rules=[]
        )

        strong_state = base_state.copy()
        strong_state.players[0].hand[0] = strong_opp

        best_move, scored_moves = select_endgame_robust_move(
            base_state,
            [base_state, strong_state],
            ai_player_idx=1
        )

        self.assertIsNotNone(best_move)
        self.assertEqual(best_move[1], (0, 0))
        risky = next(item for item in scored_moves if item['move'][1] == (2, 2))
        safe = next(item for item in scored_moves if item['move'][1] == (0, 0))
        self.assertGreater(safe['safety_ratio'], risky['safety_ratio'])

    def test_endgame_exposure_penalty_reduces_corner_score(self):
        card = Card(7, 8, 1, 6, owner='blue', card_id=10)

        early_board = Board()
        early_state = GameState(
            early_board,
            [Player('red', []), Player('blue', [card])],
            current_player_idx=1,
            rules=[]
        )

        late_board = Board()
        late_board.place_card(1, 0, Card(9, 9, 9, 9, owner='red', card_id=1))
        late_board.place_card(1, 1, Card(9, 9, 9, 9, owner='red', card_id=2))
        late_board.place_card(1, 2, Card(9, 9, 9, 9, owner='red', card_id=3))
        late_board.place_card(2, 0, Card(9, 9, 9, 9, owner='red', card_id=4))
        late_board.place_card(2, 1, Card(9, 9, 9, 9, owner='red', card_id=5))
        late_board.place_card(2, 2, Card(9, 9, 9, 9, owner='red', card_id=6))
        late_state = GameState(
            late_board,
            [Player('red', []), Player('blue', [card])],
            current_player_idx=1,
            rules=[]
        )

        early_score = evaluate_move((card, (0, 2)), early_state, {})
        late_score = evaluate_move((card, (0, 2)), late_state, {})

        self.assertLess(late_score, early_score)

    def test_endgame_corner_safety_gate_filters_risky_corner(self):
        board = Board()
        board.place_card(1, 0, Card(9, 9, 9, 9, owner='red', card_id=1))
        board.place_card(1, 1, Card(9, 9, 9, 9, owner='red', card_id=2))
        board.place_card(1, 2, Card(9, 9, 9, 9, owner='red', card_id=3))
        board.place_card(2, 0, Card(9, 9, 9, 9, owner='red', card_id=4))
        board.place_card(2, 1, Card(9, 9, 9, 9, owner='red', card_id=5))
        board.place_card(2, 2, Card(9, 9, 9, 9, owner='red', card_id=6))

        my_card = Card(7, 8, 1, 6, owner='blue', card_id=10)
        base_state = GameState(
            board,
            [Player('red', []), Player('blue', [my_card])],
            current_player_idx=1,
            rules=[]
        )

        def fake_evaluate(base_state, move, scenario_states, ai_player_idx):
            if move[1] == (0, 2):
                return (8.0, 0.2, 8.0, 6.5)
            return (7.9, 0.9, 7.9, 0.5)

        with patch('ai_server._evaluate_endgame_move_robustly', side_effect=fake_evaluate):
            best_move, scored_moves = select_endgame_robust_move(
                base_state,
                [base_state, base_state.copy()],
                ai_player_idx=1
            )

        self.assertIsNotNone(best_move)
        self.assertNotEqual(best_move[1], (0, 2))
        risky = next(item for item in scored_moves if item['move'][1] == (0, 2))
        self.assertEqual(risky['corner_risk'], 6.5)

    def test_ace_killer_only_special_cases_one_and_a(self):
        one = Card(1, 2, 3, 4)
        ten = Card(10, 2, 3, 4)
        two = Card(2, 2, 3, 4)

        self.assertEqual(one.compare_values('up', ten, 'up', ['王牌杀手']), 1)
        self.assertEqual(ten.compare_values('up', one, 'up', ['王牌杀手']), -1)
        self.assertEqual(one.compare_values('up', two, 'up', ['逆转', '王牌杀手']), 1)

    def test_same_number_flips_only_when_two_sides_match(self):
        board = Board()
        board.place_card(0, 1, Card(2, 2, 5, 2, owner='blue', card_id=1))
        board.place_card(1, 0, Card(2, 3, 2, 3, owner='red', card_id=2))

        player_card = Card(5, 4, 2, 3, owner='red', card_id=10)
        players = [
            Player('red', [player_card]),
            Player('blue', []),
        ]
        state = GameState(board, players, current_player_idx=0, rules=['同数'])

        move_record = state.make_move(1, 1, player_card)

        self.assertIsNotNone(move_record)
        self.assertEqual(state.board.get_card(0, 1).owner, 'red')
        self.assertEqual(state.board.get_card(1, 0).owner, 'red')

    def test_same_number_does_not_flip_with_only_one_match(self):
        board = Board()
        board.place_card(0, 1, Card(2, 2, 5, 5, owner='blue', card_id=1))

        player_card = Card(2, 5, 2, 3, owner='red', card_id=10)
        players = [
            Player('red', [player_card]),
            Player('blue', []),
        ]
        state = GameState(board, players, current_player_idx=0, rules=['同数'])

        move_record = state.make_move(0, 0, player_card)

        self.assertIsNotNone(move_record)
        self.assertEqual(state.board.get_card(0, 1).owner, 'blue')

    def test_base_capture_does_not_trigger_combo_chain(self):
        board = Board()
        board.place_card(1, 0, Card(1, 9, 1, 1, owner='red', card_id=1))
        board.place_card(1, 1, Card(1, 1, 1, 1, owner='red', card_id=2))

        player_card = Card(1, 1, 5, 1, owner='blue', card_id=10)
        players = [
            Player('red', []),
            Player('blue', [player_card]),
        ]
        state = GameState(board, players, current_player_idx=1, rules=[])

        move_record = state.make_move(0, 0, player_card)

        self.assertIsNotNone(move_record)
        self.assertEqual(state.board.get_card(1, 0).owner, 'blue')
        self.assertEqual(state.board.get_card(1, 1).owner, 'red')

    def test_same_number_capture_still_triggers_combo_chain(self):
        board = Board()
        board.place_card(0, 1, Card(1, 9, 5, 1, owner='blue', card_id=1))
        board.place_card(1, 0, Card(1, 3, 1, 1, owner='blue', card_id=2))
        board.place_card(0, 2, Card(1, 1, 1, 1, owner='blue', card_id=3))

        player_card = Card(5, 1, 1, 3, owner='red', card_id=10)
        players = [
            Player('red', [player_card]),
            Player('blue', []),
        ]
        state = GameState(board, players, current_player_idx=0, rules=['同数'])

        move_record = state.make_move(1, 1, player_card)

        self.assertIsNotNone(move_record)
        self.assertEqual(state.board.get_card(0, 1).owner, 'red')
        self.assertEqual(state.board.get_card(1, 0).owner, 'red')
        self.assertEqual(state.board.get_card(0, 2).owner, 'red')

    def test_plus_uses_current_type_modified_values(self):
        board = Board()
        top_card = Card(1, 1, 4, 1, owner='blue', card_id=1)
        top_card.apply_type_modifier(1)
        board.place_card(0, 1, top_card)
        board.place_card(1, 0, Card(1, 4, 1, 1, owner='blue', card_id=2))

        player_card = Card(2, 1, 1, 3, owner='red', card_id=10)
        players = [
            Player('red', [player_card]),
            Player('blue', []),
        ]
        state = GameState(board, players, current_player_idx=0, rules=['加算'])

        move_record = state.make_move(1, 1, player_card)

        self.assertIsNotNone(move_record)
        self.assertEqual(state.board.get_card(0, 1).owner, 'red')
        self.assertEqual(state.board.get_card(1, 0).owner, 'red')

    def test_same_type_strengthen_counts_each_placed_type_card(self):
        board = Board()
        first_card = Card(1, 1, 1, 1, owner='red', card_id=1, card_type='拂晓')
        second_card = Card(2, 2, 2, 2, owner='blue', card_id=2, card_type='拂晓')
        hand_card = Card(3, 3, 3, 3, owner='blue', card_id=3, card_type='拂晓')
        players = [
            Player('red', []),
            Player('blue', [hand_card]),
        ]
        state = GameState(board, players, current_player_idx=1, rules=['同类强化'])

        board.place_card(0, 0, first_card)
        state.recalculate_type_modifiers()
        self.assertEqual(first_card.type_modifier, 1)
        self.assertEqual(hand_card.type_modifier, 1)

        board.place_card(0, 1, second_card)
        state.recalculate_type_modifiers()
        self.assertEqual(first_card.type_modifier, 2)
        self.assertEqual(second_card.type_modifier, 2)
        self.assertEqual(hand_card.type_modifier, 2)

    def test_same_type_weaken_counts_each_placed_type_card(self):
        board = Board()
        first_card = Card(5, 5, 5, 5, owner='red', card_id=1, card_type='拂晓')
        second_card = Card(6, 6, 6, 6, owner='blue', card_id=2, card_type='拂晓')
        players = [
            Player('red', []),
            Player('blue', []),
        ]
        state = GameState(board, players, current_player_idx=1, rules=['同类弱化'])

        board.place_card(0, 0, first_card)
        board.place_card(0, 1, second_card)
        state.recalculate_type_modifiers()

        self.assertEqual(first_card.type_modifier, -2)
        self.assertEqual(second_card.type_modifier, -2)
        self.assertEqual(first_card.get_modified_value('up'), 3)
        self.assertEqual(second_card.get_modified_value('up'), 4)

    def test_undo_move_restores_hand_card_owner(self):
        board = Board()
        player_card = Card(5, 4, 2, 3, owner='blue', card_id=10)
        players = [
            Player('red', []),
            Player('blue', [player_card]),
        ]
        state = GameState(board, players, current_player_idx=1, rules=[])

        move_record = state.make_move(1, 1, player_card)
        self.assertIsNotNone(move_record)
        self.assertEqual(player_card.owner, 'blue')

        state.undo_move(move_record)

        self.assertEqual(player_card.owner, 'blue')
        self.assertEqual(state.players[1].hand[0].owner, 'blue')


if __name__ == '__main__':
    unittest.main()
