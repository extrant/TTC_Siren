import unittest
from unittest.mock import patch

import app as server_app
import draft_analyzer_app
from draft_analyzer_app import card_sides_from_payload, expand_selection_candidates


def selection_payload():
    return [
        [
            {
                "left": {"id": "node11L", "up": 7, "right": 8, "down": 1, "left": 6},
                "right": {"id": "node11R", "up": 3, "right": 7, "down": 8, "left": 10},
            },
            {
                "left": {"id": "node12L", "up": 7, "right": 4, "down": 4, "left": 8},
                "right": {"id": "node12R", "up": 7, "right": 9, "down": 1, "left": 10},
            },
            {
                "left": {"id": "node13L", "up": 8, "right": 6, "down": 7, "left": 1},
                "right": {"id": "node13R", "up": 4, "right": 8, "down": 10, "left": 6},
            },
        ],
        [
            {
                "left": {"id": "node25L", "up": 4, "right": 3, "down": 7, "left": 3},
                "right": {"id": "node25R", "up": 7, "right": 7, "down": 9, "left": 4},
            },
            {
                "left": {"id": "node26L", "up": 7, "right": 5, "down": 5, "left": 3},
                "right": {"id": "node26R", "up": 6, "right": 8, "down": 4, "left": 7},
            },
            {
                "left": {"id": "node27L", "up": 4, "right": 1, "down": 6, "left": 7},
                "right": {"id": "node27R", "up": 7, "right": 9, "down": 8, "left": 1},
            },
        ],
        [
            {"id": "node39", "up": 4, "right": 3, "down": 2, "left": 3},
            {"id": "node40", "up": 4, "right": 4, "down": 3, "left": 4},
            {"id": "node41", "up": 2, "right": 5, "down": 2, "left": 5},
        ],
    ]


class DraftSelectionAnalyzerTest(unittest.TestCase):
    def test_original_payload_rotates_to_ui_sides(self):
        self.assertEqual(card_sides_from_payload({"original": "7,A,8,3"}), (3, 7, 8, 10))

    def test_selection_payload_expands_to_27_complete_decks(self):
        expanded = expand_selection_candidates(selection_payload())

        self.assertEqual(len(expanded), 27)
        first_deck, first_labels = expanded[0]
        self.assertEqual(len(first_deck), 5)
        self.assertEqual(first_labels, ["备选1-候选1", "备选2-候选1", "备选3-候选1"])
        self.assertEqual(
            [(card.base_up, card.base_right, card.base_down, card.base_left) for card in first_deck],
            [(7, 8, 1, 6), (3, 7, 8, 10), (4, 3, 7, 3), (7, 7, 9, 4), (4, 3, 2, 3)],
        )

    def test_api_draft_analyze_accepts_selection_mode(self):
        verification = (
            50.0,
            {"rounds": 1, "threat_rounds": 0, "simulations": 2, "average": 50.0, "floor": 50.0, "min": 50.0},
        )
        with patch.object(server_app, "_draft_valid_database_cards", return_value=[]), patch.object(
            server_app, "_draft_verification_score", return_value=verification
        ):
            response = server_app.app.test_client().post(
                "/api/draft/analyze",
                json={"mode": "selection", "draftChoices": selection_payload(), "timeBudget": 5.0},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["results"]), 27)
        self.assertIn("selection", data["best"])
        self.assertTrue(any("展开完整卡组：27套" == item for item in data["process"]))

    def test_api_draft_analyze_returns_recommendation_groups(self):
        verification = (
            50.0,
            {"rounds": 1, "threat_rounds": 0, "simulations": 2, "average": 50.0, "floor": 50.0, "min": 50.0},
        )
        flat_payload = [
            [
                card
                for choice in selection_payload()[0]
                for card in (choice["left"], choice["right"])
            ],
            [
                card
                for choice in selection_payload()[1]
                for card in (choice["left"], choice["right"])
            ],
            selection_payload()[2],
        ]
        with patch.object(server_app, "_draft_valid_database_cards", return_value=[]), patch.object(
            server_app, "_draft_verification_score", return_value=verification
        ):
            response = server_app.app.test_client().post(
                "/api/draft/analyze",
                json={"rules": ["选拔", "加算"], "groups": flat_payload, "timeBudget": 5.0},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["recommendation"]), 3)
        self.assertEqual(data["recommendation"], data["best"]["groups"])
        self.assertEqual([group["stage"] for group in data["recommendation"]], [1, 2, 3])
        self.assertEqual(len(data["recommendation"][0]["cards"]), 2)
        self.assertEqual(len(data["recommendation"][2]["cards"]), 1)
        self.assertIn("id", data["recommendation"][0]["cards"][0])
        self.assertIn("up", data["recommendation"][0]["cards"][0])
        self.assertIn("right", data["recommendation"][0]["cards"][0])
        self.assertIn("down", data["recommendation"][0]["cards"][0])
        self.assertIn("left", data["recommendation"][0]["cards"][0])

    def test_random_selection_endpoint_returns_frontend_payload(self):
        response = draft_analyzer_app.app.test_client().post("/api/random", json={})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["mode"], "selection")
        self.assertEqual(data["rules"], ["选拔"])
        self.assertEqual(len(data["candidates"]), 3)
        self.assertEqual(len(expand_selection_candidates(data["candidates"])), 27)

    def test_standalone_analyze_accepts_random_selection_payload(self):
        def fake_collect(candidates, *_args):
            samples = [
                {"scores": [40.0 + index], "rounds": 1, "threat_rounds": 0}
                for index, _candidate in enumerate(candidates)
            ]
            return samples, 2, 1.25

        with patch.object(draft_analyzer_app, "valid_database_cards", return_value=[]), patch.object(
            draft_analyzer_app, "collect_samples_parallel", side_effect=fake_collect
        ) as collect_mock:
            response = draft_analyzer_app.app.test_client().post(
                "/api/analyze",
                json={"rules": ["选拔"], "candidates": selection_payload(), "timeBudget": 5.0},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["results"]), 27)
        self.assertEqual(len(collect_mock.call_args_list[0].args[0]), 27)
        self.assertEqual(len(collect_mock.call_args_list[1].args[0]), 8)
        self.assertNotIn("未进入TopK深算，使用启发式评分", data["best"]["reasons"])
        self.assertTrue(any(item.startswith("步进深算：第一段全量") for item in data["process"]))
        self.assertTrue(any("第一段全量深算：2个进程，27套，每套约1.25秒，共27局" == item for item in data["process"]))
        self.assertTrue(any("第二段追加深算：第一段前8套进入追加验证" == item for item in data["process"]))
        self.assertTrue(any("排序策略：27套均已深算，TopK候选使用两段合并样本" == item for item in data["process"]))


if __name__ == "__main__":
    unittest.main()
