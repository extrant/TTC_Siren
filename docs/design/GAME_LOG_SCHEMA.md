# Game log schema — version 2

Status: implemented in Phase 8A (version 1) and changed in Phase 8B (version 2; `csharp/src/TTC.Siren.Server/GameLog`). The schema version is the field `v` of every record; a change of meaning of any field bumps it.

**Version 2 (Phase 8B, review L8-2 and L8-4):** (1) integers are written as integers (`"max_depth": 4`, `"card_id": 357`; version 1 wrote some of them as `4.0`); (2) `request_raw` is the request as it was received — its members in the sender's order, every number written the way the sender wrote it (`4`, `4.0`, `1E1`), only the white space between tokens is removed (a compact body comes back byte for byte); (3) `opponent_move` has `ambiguous` and `candidate_ids` (see there). Files of version 1 are still read: a game written by version 1 may continue in version 2, so one file can hold records of both (every record carries its own `v`; the games API reports the highest as `schema`).

## 1. Storage

- One file per game: `<GameLog.Directory>/<key id>/<game id>.jsonl`; UTF-8, one JSON object per line, append-only.
- `<key id>` is the id of the API key that made the requests (never the secret); `<game id>` matches `^[A-Za-z0-9_-]{1,64}$`. Both are validated before they touch a path.
- A key sees only its own games (list, read, download, delete). The files hold users' requests: they are never committed, and the directory must be backed up and protected like a database.
- Retention: a file not written for `GameLog.RetentionDays` days (default 30) is deleted by the cleanup job (at start-up and once a day). Switch the whole feature off with `GameLog.Enabled = false`.
- Floats are written the way Python's `repr` writes them (shortest round-trip), so a score read back is the same double. 64-bit integers that exceed JSON-safe range (the seed) are strings.

## 2. Common fields

Every record: `v` (2; 1 in files written before Phase 8B), `seq` (0, 1, 2 … without gaps), `type`, `t` (UTC, `yyyy-MM-ddTHH:mm:ss.fffZ`), `elapsed_ms` (since the game started).

## 3. Records

### `game_start` (first record)
`game_id`; `id_source` (`client` = the request carried `gameId`, `new` = a game the server started, `inferred` is recorded on the request, see below); `key_id`; `rules` (list), `rules_text` (as sent), `open_mode`; `my_owner` (`red`/`blue`: the plugin user's colour); `npc_name`, `npc_ids`; `known_decks` (`mine`, `opponent`: card ids visible in the hands of the first request); `server`; `schema` (2).

### `request`
- `request_no` (1-based within the game); `seed` (string; the 64-bit seed of the unknown-card sampling, fresh and cryptographic in production); `settings` (`max_depth`, `max_time` as sent, `aggressive`, `solver`, `explain`, `show_search_progress`, `tt_entries`).
- `request_raw`: the plugin's request as received (version 2: tokens exactly as written, white space between tokens removed). With `seed` this is all a decision depends on.
- `rules`; `state`: the position before the search in the engine form `{cur, board[9], hands[2], generated[2]}` (a card is `[id|null, type|null, "r"|"b"|null, modifier, base u,r,d,l, display u,r,d,l, canUse]`; `generated` marks the hand cards that are assumed cards for unknown slots). The state is what the server searched: hands in request order, unknown slots already filled with assumed cards.
- `id_inference`: why this request was attached to the game (`consistent successor`, `same position (repeated request)`) or null.
- `http_status`, `error` (null on success).
- `result` (when the engine answered): `move` / `standard_move` (`slot`, `row`, `col`, `card_id`; the standard move is the search's, the move may differ when the robust endgame overrode it), `standard_score`, `depth_reached`, `nodes`, `tt_hits`, `cutoffs`, `root_moves`, `search_ms`, `depths` (per completed depth: `depth`, `score`, `score_bits`, `move`, `nodes_total`, `elapsed_ms`), `robust` (`scenarios`, `overridden`, `elapsed_ms`) or null, `timings`.
- `explanation`: `{version: 2, level, facts, text: {brief, full}}` (see `EXPLANATION_FACTS.md`).

### `opponent_move` (inferred)
Written when a request follows an earlier request of the same game: the difference of the two positions, recomputed by the rules.
`between` (`[previous request_no, request_no]`); `consistent` (true when the board of the new request equals the board the rules predict after the user's move and the opponent's card; false: the flips below are only the owner changes, without causes, and `note` says why); `card_id`, `card_edges`, `row`, `col`, `owner` (colour of the opponent); `ambiguous` and `candidate_ids` (version 2, see below); `flips` (`row`, `col`, `card_id`, `cause`: `basic`, `reverse`, `fallen_ace`, `same`, `plus`, `combo`); `my_move` (`row`, `col`, `as_recommended`) or null; `new_cells`; `note`.

**Ambiguous cards (L8-4).** A request names a card by its four edges (the plugin sends no card id), so a card inferred between two requests is only known up to its edges. When more than one database card has these four edges — the review's example is ids 12 and 369 — the record has `"ambiguous": true` and `candidate_ids` lists all of them in database order; `card_id` is the first one, as Python's `find_card_id_by_stats` would choose. `ambiguous` is `false` and `candidate_ids` holds the single id for a unique card, and `[]` when no card could be identified. The type of the cards can differ (the request carries none), so the type-dependent rules (同类强化 / 同类弱化) may differ between the candidates; a UI should show `card_edges`, with the ids only as candidates. The same holds for the cards inside `state` and `flips`, whose ids come from the same lookup.

### `game_end`
`source` (`observed`: a request arrived with a full board; `own_last_move`: the move just recommended fills the board); `final` (`mine`, `theirs`: cards on the board plus cards left in hands, as the rules count); `winner` (`mine`/`theirs`/`draw`); `board`.
If the opponent places the ninth card and the plugin never asks again, the server never sees that card: no `opponent_move` and no `game_end` can be written for it.

## 4. Game identity

1. A request with `"gameId": "<id>"` (valid pattern) belongs to that game; the game is created on first use. Recommended for plugins.
2. Otherwise the server infers it: the request continues the most recent open game of the same key if it is a consistent successor of that game's last request — same rules and colour, the same side to move, within `GameLog.GapMinutes` (default 15) of the previous request, every card of the old board still on its cell with the same edges, at most four new cards, and no card in the mover's known hand that was not in it before. Otherwise a new game starts (`id_source: new`).
3. A finished game (`game_end` written) is never continued by inference.

## 5. Replay

A `request` record is replayed with `GameReplay.ReplayAsync`: the same `request_raw` and `seed`, search depth set to what the original completed (when it was cut short by the clock) and no time limit. The move, the standard move, the score and move of every depth, the robust result and all explanation facts and texts are reproduced; only measured times differ (and, for a search the clock interrupted, the node count and the requested depth). Replay needs a table of the same size (`settings.tt_entries`) for the principal variation to be the same.

## 6. API

`GET /v1/games?offset=&limit=`, `GET /v1/games/{id}` (JSON, all records), `GET /v1/games/{id}/download` (JSONL), `DELETE /v1/games/{id}`. A game of another key answers 404 (its existence is not revealed).
