"""Regression tests for D-002: undo_move must restore type_modifier by card identity.

Before the fix, the modifiers were saved by hand *index* but undo_move appends the played card to the end of
the hand, so values were written onto the wrong cards (see CSHARP_MIGRATION.md, D-002).
"""
import random
import unittest

from core.board import Board
from core.card import Card
from core.game_state import GameState
from core.player import Player

TYPES = ['兽人', '拂晓', '帝国', '蛮神', None]


def all_cards(state):
    cards = [state.board.grid[r][c] for r in range(3) for c in range(3) if state.board.grid[r][c] is not None]
    for p in state.players:
        cards.extend(p.hand)
    return cards


class UndoTypeModifierTests(unittest.TestCase):
    def test_modifiers_follow_the_card_not_the_hand_index(self):
        board = Board()
        board.place_card(0, 0, Card(5, 5, 5, 5, owner='blue', card_id=40, card_type='拂晓'))
        a = Card(5, 5, 5, 5, owner='red', card_id=1)
        b = Card(5, 5, 5, 5, owner='red', card_id=2)
        c = Card(5, 5, 5, 5, owner='red', card_id=3)
        a.apply_type_modifier(1)
        c.apply_type_modifier(2)
        state = GameState(board, [Player('red', [a, b, c]), Player('blue', [])], 0, ['同类强化'])

        record = state.make_move(2, 2, b)
        self.assertIsNotNone(record)
        state.undo_move(record)

        self.assertEqual([card.card_id for card in state.players[0].hand], [1, 3, 2])  # D-001 order change is unchanged
        self.assertEqual((a.type_modifier, b.type_modifier, c.type_modifier), (1, 0, 2))

    def test_typed_move_restores_every_modifier(self):
        board = Board()
        board.place_card(0, 0, Card(5, 5, 5, 5, owner='blue', card_id=40, card_type='拂晓'))
        played = Card(4, 4, 4, 4, owner='red', card_id=46, card_type='拂晓')
        other_same_type = Card(4, 4, 4, 4, owner='red', card_id=47, card_type='拂晓')
        other_type = Card(4, 4, 4, 4, owner='red', card_id=31, card_type='帝国')
        hand_blue = [Card(3, 3, 3, 3, owner='blue', card_id=53, card_type='拂晓')]
        state = GameState(board, [Player('red', [played, other_same_type, other_type]), Player('blue', hand_blue)],
                          0, ['同类强化'])
        state.recalculate_type_modifiers()
        before = {id(card): card.type_modifier for card in all_cards(state)}
        self.assertEqual(before[id(played)], 1)

        record = state.make_move(1, 1, played)
        self.assertEqual(other_same_type.type_modifier, 2)
        self.assertEqual(other_type.type_modifier, 0)
        state.undo_move(record)

        self.assertEqual({id(card): card.type_modifier for card in all_cards(state)}, before)

    def test_random_make_undo_roundtrip_restores_all_modifiers(self):
        for seed in range(300):
            rng = random.Random(seed)
            rules = [rng.choice(['同类强化', '同类弱化'])]
            board = Board()
            next_id = [0]

            def new_card(owner):
                next_id[0] += 1
                return Card(rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10),
                            owner=owner, card_id=next_id[0], card_type=rng.choice(TYPES))

            for cell in rng.sample(range(9), rng.randint(0, 4)):
                board.place_card(cell // 3, cell % 3, new_card(rng.choice(['red', 'blue'])))
            players = [Player('red', [new_card('red') for _ in range(rng.randint(2, 5))]),
                       Player('blue', [new_card('blue') for _ in range(rng.randint(2, 5))])]
            state = GameState(board, players, rng.randint(0, 1), rules)
            state.recalculate_type_modifiers()
            before = {id(card): card.type_modifier for card in all_cards(state)}

            stack = []
            for _ in range(rng.randint(1, 5)):
                moves = state.get_available_moves()
                if not moves:
                    break
                card, (r, c) = rng.choice(moves)
                stack.append(state.make_move(r, c, card))
            while stack:
                state.undo_move(stack.pop())

            after = {id(card): card.type_modifier for card in all_cards(state)}
            self.assertEqual(after, before, f'seed={seed} rules={rules}')


if __name__ == '__main__':
    unittest.main()
