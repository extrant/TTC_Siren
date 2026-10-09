"""Regression tests for D-004: make_move must remove the hand card that is played, even when several cards in a
hand share a card_id (two copies of one real card, or two placeholders whose id is None).

Before the fix, make_move located the hand card by card_id only: it removed the *first* card with that id but put the
card that was passed on the board, so after undo_move the hand held a duplicate reference and the original card was
lost (see CSHARP_MIGRATION.md, D-004).
"""
import random
import unittest

from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player

RULE_SETS = [[], ['同数'], ['加算', '同数'], ['逆转'], ['同类强化'], ['同类弱化', '加算']]
TYPES = ['兽人', '拂晓', '帝国', '蛮神', None]


def board_cards(state):
    return [state.board.grid[r][c] for r in range(3) for c in range(3) if state.board.grid[r][c] is not None]


def identities(cards):
    return sorted(id(card) for card in cards)


class DuplicateHandIdTests(unittest.TestCase):
    def test_second_card_with_the_same_id_is_the_one_placed_and_removed(self):
        first = Card(1, 2, 3, 4, owner='red', card_id=7)
        second = Card(5, 6, 7, 8, owner='red', card_id=7)
        other = Card(9, 9, 9, 9, owner='red', card_id=8)
        state = GameState(Board(), [Player('red', [first, second, other]), Player('blue', [])], 0, [])

        record = state.make_move(1, 1, second)

        self.assertIsNotNone(record)
        self.assertIs(state.board.get_card(1, 1), second)
        self.assertEqual([id(c) for c in state.players[0].hand], [id(first), id(other)])

        state.undo_move(record)
        self.assertEqual(len(state.players[0].hand), 3)
        self.assertEqual(identities(state.players[0].hand), identities([first, second, other]))
        self.assertIsNone(state.board.get_card(1, 1))
        self.assertEqual([id(c) for c in state.players[0].hand], [id(first), id(other), id(second)])  # D-001 unchanged

    def test_two_cards_without_an_id_are_distinguished_by_identity(self):
        first = Card(1, 1, 1, 1, owner='blue', card_id=None)
        second = Card(2, 2, 2, 2, owner='blue', card_id=None)
        state = GameState(Board(), [Player('red', []), Player('blue', [first, second])], 1, [])

        record = state.make_move(0, 2, second)

        self.assertIs(state.board.get_card(0, 2), second)
        self.assertEqual([id(c) for c in state.players[1].hand], [id(first)])
        state.undo_move(record)
        self.assertEqual(identities(state.players[1].hand), identities([first, second]))

    def test_a_copy_that_is_not_in_the_hand_still_matches_by_card_id(self):
        # count_captured_cards in ai_server passes card.copy(); the fallback must keep that working.
        first = Card(1, 2, 3, 4, owner='red', card_id=7)
        second = Card(5, 6, 7, 8, owner='red', card_id=9)
        state = GameState(Board(), [Player('red', [first, second]), Player('blue', [])], 0, [])
        probe = Card(5, 6, 7, 8, owner='red', card_id=9)

        record = state.make_move(2, 2, probe)

        self.assertIsNotNone(record)
        self.assertIs(state.board.get_card(2, 2), probe)
        self.assertEqual([id(c) for c in state.players[0].hand], [id(first)])  # the hand card with that id was removed

    def test_card_that_is_not_in_the_hand_and_has_no_matching_id_is_rejected(self):
        state = GameState(Board(), [Player('red', [Card(1, 1, 1, 1, owner='red', card_id=1)]), Player('blue', [])], 0, [])
        self.assertIsNone(state.make_move(0, 0, Card(1, 1, 1, 1, owner='red', card_id=2)))

    def test_seeded_random_make_undo_never_loses_or_duplicates_a_card(self):
        for seed in range(300):
            rng = random.Random(20261009 + seed)
            rules = rng.choice(RULE_SETS)

            def hand(color, n):
                cards = []
                for _ in range(n):
                    kind = rng.random()
                    if kind < 0.4 and cards:
                        src = rng.choice(cards)  # a same-id copy with identical edges
                        cards.append(Card(src.base_up, src.base_right, src.base_down, src.base_left, owner=color,
                                          card_id=src.card_id, card_type=src.card_type))
                    elif kind < 0.6:
                        cards.append(Card(rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10),
                                          rng.randint(1, 10), owner=color, card_id=None, card_type=rng.choice(TYPES)))
                    else:
                        cards.append(Card(rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10),
                                          rng.randint(1, 10), owner=color, card_id=rng.randint(1, 4),
                                          card_type=rng.choice(TYPES)))
                return cards

            red, blue = hand('red', rng.randint(2, 5)), hand('blue', rng.randint(2, 5))
            state = GameState(Board(), [Player('red', red), Player('blue', blue)], rng.randint(0, 1), rules)
            if '同类强化' in rules or '同类弱化' in rules:
                state.recalculate_type_modifiers()
            everything = identities(red + blue)
            stack = []

            for _ in range(60):
                mover = state.current_player
                empty = [(r, c) for r in range(3) for c in range(3) if state.board.is_empty(r, c)]
                can_make = bool(empty) and bool(mover.hand) and (not stack or rng.random() < 0.6)
                if can_make:
                    card = rng.choice(mover.hand)
                    r, c = rng.choice(empty)
                    before_hand = list(mover.hand)
                    record = state.make_move(r, c, card)
                    self.assertIsNotNone(record, f'seed {seed}')
                    self.assertIs(state.board.get_card(r, c), card, f'seed {seed}')
                    self.assertEqual(identities(mover.hand), identities([x for x in before_hand if x is not card]),
                                     f'seed {seed}')
                    stack.append(record)
                elif stack:
                    state.undo_move(stack.pop())
                else:
                    break
                present = board_cards(state) + state.players[0].hand + state.players[1].hand
                self.assertEqual(identities(present), everything, f'seed {seed}: a card was lost or duplicated')

            while stack:
                state.undo_move(stack.pop())
            self.assertEqual(board_cards(state), [], f'seed {seed}')
            self.assertEqual(identities(state.players[0].hand), identities(red), f'seed {seed}')
            self.assertEqual(identities(state.players[1].hand), identities(blue), f'seed {seed}')


if __name__ == '__main__':
    unittest.main()
