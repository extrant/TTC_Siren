import unittest

from core.card import Card
from npc_deck_recommender_demo import (
    generate_npc_decks,
    generate_player_candidates,
    player_deck_is_legal,
    recommend_decks,
)


def card(card_id, up, right, down, left, card_type=None):
    return Card(up, right, down, left, card_id=card_id, card_type=card_type)


class NpcDeckRecommenderDemoTests(unittest.TestCase):
    def setUp(self):
        self.card_index = {
            1: card(1, 4, 4, 4, 4),
            2: card(2, 5, 5, 4, 3),
            3: card(3, 6, 4, 5, 3),
            4: card(4, 3, 6, 4, 5),
            5: card(5, 8, 8, 7, 7),
            6: card(6, 9, 7, 8, 6),
            7: card(7, 10, 9, 8, 7),
            8: card(8, 9, 9, 9, 9),
            101: card(101, 10, 10, 8, 8),
            102: card(102, 9, 9, 9, 8),
            103: card(103, 8, 10, 8, 10),
            104: card(104, 10, 8, 10, 8),
            105: card(105, 9, 10, 9, 10),
            106: card(106, 8, 9, 10, 9),
        }
        self.star_map = {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 4,
            6: 4,
            7: 5,
            8: 5,
            101: 5,
            102: 5,
            103: 5,
            104: 5,
            105: 5,
            106: 5,
        }

    def test_player_candidates_respect_official_deck_limits(self):
        owned = [self.card_index[idx] for idx in range(1, 9)]
        npc = [self.card_index[idx] for idx in range(101, 106)]

        candidates = generate_player_candidates(owned, npc, [], self.star_map)

        self.assertTrue(candidates)
        for _, _, deck in candidates:
            self.assertTrue(player_deck_is_legal(deck, self.star_map))

    def test_npc_decks_ignore_player_star_limits(self):
        npc = [self.card_index[idx] for idx in range(101, 107)]

        decks = generate_npc_decks(npc, [], self.star_map)

        self.assertTrue(decks)
        self.assertEqual(sum(1 for card_item in decks[0] if self.star_map[card_item.card_id] == 5), 5)

    def test_recommend_decks_returns_legal_player_deck_against_npc_pool(self):
        result = recommend_decks(
            [1, 2, 3, 4, 5],
            [101, 102, 103, 104, 105, 106],
            ["加算"],
            top_n=1,
            card_index=self.card_index,
            star_map=self.star_map,
        )

        self.assertIn("best", result)
        self.assertEqual(len(result["best"]["deck"]), 5)
        self.assertEqual(result["best"]["warnings"], [])
        self.assertEqual(len(result["npc_decks"][0]), 5)


if __name__ == "__main__":
    unittest.main()
