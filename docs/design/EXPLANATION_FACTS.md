# Explanation facts v2 and the text templates

Status: implemented in Phase 8A (`csharp/src/TTC.Siren.Explain`, decision A-24). The response of `/ai_move` gets an additive field

```json
"explanation": {"version": 2, "level": "brief|full", "language": "zh-CN", "facts": [ ... ], "text": {"brief": "...", "full": "..."}}
```

Request parameter `explain`: `"brief"` (default) or `"full"` (also scores the alternatives). If the explanation cannot be produced the response carries `explanation_error` instead and the move is unaffected.

## 1. Principles

- **Facts are computed by the engine and verified by recomputation** (tests `ExplanationFactsTests`, `FlipAnalyzerTests`): flips against the engine's own `MakeMove`, threats and capture probabilities by brute force with `MakeMove`, exact outcomes by a brute-force minimax, alternative scores against fixed-depth searches, the robust figures against the engine's candidate table (bit for bit).
- **The text says only what the facts contain.** Every number in a text is found in the facts of the same response (tested); a win is claimed only when the exact value says win.
- All searches the explainer runs use node budgets, never the clock, so the facts of a decision are reproducible (replay).
- `side to move` is "self", the other side "opponent" (normally the plugin user and the opponent; the engine also answers for the opponent when it is the one to move).

## 2. Fact families (order in `facts`)

| `type` | content |
|---|---|
| `immediate_effect` | `move` (real card id, cell and cell name, effective edges, `assumed`), `flips` (cell, card, `cause`, the compared sides and values; for `combo`: the triggering cell and `combo_kind`), `flip_count`, `cause_counts`, `cards_before` / `cards_after` (own/opponent, hands included), `modifier_changes` (同类强化/弱化: type, from, to, how many cards), `active_rules`. Causes: `basic`, `reverse` (逆转 decided), `fallen_ace` (王牌杀手), `same`, `plus`, `combo` (连携, one level, D-013). |
| `exposure` | `sides` (each side of the placed card: `wall`/`empty`/`own`/`enemy` neighbour and its value), `exposed_sides`, `threats` (each opponent card, known or assumed, that can take the placed card next turn: where, by which cause, how many cards that move flips), `capturable_now`, `unknown_capture_probability` and `_percent` (share of the legal candidates for an unknown opponent card — database cards not seen, inside the deck limits and the NPC pool — that could take it), `position_kind`, `position_weight`, `placement_edge_score` (the terms the evaluation uses). |
| `opponent_reply` | the opponent's strongest reply from the search line (second ply): card (and whether assumed), cell, its flips, `captures_placed_card`, `source` (`search_table`: read from the transposition table of the very search; `line_search`: a bounded fresh search when the table has no usable line, e.g. after a robust override). |
| `outcome` | `pv` (the line, up to six plies, alternating `self`/`opponent`), `kind`: `exact` (at most six empty cells: the exact card difference, `result` win/draw/loss, `basis` known_hands or assumed_hands) or `heuristic` (the search score at `depth`, `result` ahead/level/behind with ±0.25 as the level band). |
| `alternatives` (full) | up to four other moves scored at the SAME depth as the chosen move (`eval_depth`; the largest depth that fits a node budget): `score`, `score_delta`, `tied` (exactly equal score, Q-5 (b)), `flips`, the reply that follows, and `why_worse`: `tied`, `heuristic_better` (a robust override chose the safer move), `exposed_to_reply` (their reply takes the placed card: side and values), `fewer_captures`, `bigger_counter`, or `score_gap`. |
| `robust_endgame` | `scenarios`, `safety_ratio`/`safety_percent` (share of scenarios where the move does not lose), `average_difference`, `worst_difference` (exact card differences over the scenarios), `final_score`, `overridden`, `override_rule` (`safety_floor`, `safer_and_close`, `clearly_better`, `aggressive_margin`: the clause of the decision that fired), the standard move and its figures. |
| `unknown_cards` | counts of unknown cards (opponent, own), `strategy` (`npc_pool`, `rule_model` with `primary_rule`, `random_sample`, `threat_ranking`, `none`), NPC name and pool size, the assumed cards of this decision (slot, real id, star, type). |
| `confidence` | `solved_to_end` (the whole game tree was searched), `depth_reached` / `depth_requested`, `empty_cells`, `nodes`, `search_ms`, `stopped_early`, `robust_used`. |

## 3. Templates

`csharp/src/TTC.Siren.Explain/templates/explain.zh-CN.json` (embedded in the assembly; `Server:TemplatesDirectory` can point to a directory holding another `explain.zh-CN.json`). It maps a key to several phrasings; the phrasing is chosen deterministically from the numbers of the fact being told (FNV-1a hash), so the same facts always give the same text. A template may only use slots that the renderer fills; an unknown slot or key fails loudly. The slots per key are visible in `ExplanationRenderer.cs`. A different voice (Phase 8D, the mascot) is a different file with the same keys.

Brief = two sentences: what the move does now, and the outlook (exact result or heuristic score with its depth). Full = paragraphs in the order of the families: the move and its flips with causes, exposure and threats, the opponent's reply and the line, the outcome, the alternatives, the robust endgame, the unknown cards, confidence.

Golden snapshots of the text: `tests/data/explanation_snapshots_v2.json` (30 decisions). A deliberate change of wording or facts is accepted by running the tests once with `TTC_UPDATE_SNAPSHOTS=1` and reviewing the diff.
