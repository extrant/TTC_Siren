# CSHARP_MIGRATION.md — TTC_Siren Python → C# 迁移进度（唯一事实来源）

> 每个 Claude 会话：先读 `CLAUDE.md`（长期规则），再读本文件。
> 本文件记录：Python 架构、性能热点、baseline、C# 目标架构、Phase 定义与状态、设计决定、语义差异、benchmark 历史、BLOCKED、下一步。
> 任何 Phase 完成后必须更新本文件。

---

## 0. 状态总览

| Phase | 名称 | 状态 |
|---|---|---|
| 1 | Python 架构分析 / profiling / baseline | **DONE**（2026-10-07） |
| 2 | Card / Move / RuleSet / GameState / MakeMove / UndoMove + 规则 parity 工具 | **DONE**（2026-10-07，验收见 §7 Phase 2） |
| 3 | Evaluator / MoveOrdering / History / Minimax / AlphaBeta / ID（ReferenceCompat，节点级 parity） | **DONE**（2026-10-07，验收见 §7 Phase 3） |
| 4 | Zobrist / 带 bound 的 TranspositionTable / Fast 搜索模式 | NOT STARTED — 审查门禁 **`PASS_TO_PHASE_4`**（技术上可以开始；仍需用户明确授权，建议先决定 Q-5） |
| 5 | 精确残局 + 信息集 robust 残局 + Python-vs-C# 全量 benchmark | NOT STARTED |
| 6 | Profiler 驱动优化（单线程）→ 可选 root parallelism | NOT STARTED |
| 7 | 集成策略：Python 服务调用 C# 引擎（桥接 + 回退） | NOT STARTED |
| 8 | 全量服务移植（请求解析、未知牌采样、MC、分析文案、app.py 子系统、HTTP） | NOT STARTED |
| 9 | 可选：CP-SAT 隐藏牌组合 / deck optimization / scenario generation | NOT STARTED |

**审查门禁（Phase 2+3 独立审查，2026-10-07）**：结果 **`BLOCK_PHASE_4`**。报告：`docs/reviews/OPUS_PHASE3_REVIEW.md`。阻塞项为 3 个 HIGH：H-1 搜索异常时 GameState 未恢复；H-2 ReferenceCompat TT 在服务端等价配置下溢出；H-3 `Move` 以手牌槽位为身份。规则、评估、搜索与 Python 的 parity 已被独立复核（新种子重新生成全部 golden，外加对抗性用例，0 不一致）。修复任务：`.claude/sonnet_review_fix_goal.md`。修复完成并经复审改为 PASS 之前，**不得开始 Phase 4**（即使 `.claude/phase_goal.md` 按 §0 选中了 Phase 4）。Phase 4 开始前仍需用户决定 Q-5。
**整改进展（2026-10-07）**：H-1、H-2、H-3 的整改已完成，**等待复审**（记录见 §7 Phase 3 的 "Review remediation (H-1/H-2/H-3)"；测量见 §10）。门禁状态 **仍为 `BLOCK_PHASE_4`**，由复审者而不是整改者改为 PASS；Phase 4 仍为 NOT STARTED。
**复审结果（2026-10-07，`docs/reviews/OPUS_PHASE3_REVIEW.md` §15）**：门禁改为 **`PASS_TO_PHASE_4`**。H-1、H-2、H-3 均经独立验证为 **RESOLVED**，没有引入回归：43/43 测试通过；在原 golden 与新种子 golden 上 parity 计数完全不变；对抗性用例 0 不一致；"过期走法" fuzz 115,955 次 0 错误；中途抛异常（含调用方已有走法、ID 第 2 层以后抛出）时状态都能恢复；服务端等价调用 4 个局面全部完成，TT 峰值 512 MiB。M-1 到 M-3、L-1 到 L-12、O-1 到 O-6 仍未处理，均不阻塞（建议在 Phase 4 依赖新 golden 之前处理 M-1 到 M-3）。Phase 4 仍需用户明确授权；Q-5 的建议见审查报告 §15.6（选项 b：只允许在分数完全相等的走法之间不同，并逐条验证、报告）。

分析基线：分支 `csharp-solver-migration`，commit `eb60a02`。Phase 1 未修改任何 Python 代码，只新增了 `CLAUDE.md` 和本文件。
Phase 2 同样没有修改任何已跟踪文件（`git status` 只有新增的未跟踪文件：`csharp/`、`tools/`、`.claude/`、两份文档），尚未 commit。

---

## 1. Python 当前架构

### 1.1 模块与入口

| 文件 | 行数 | 作用 | 是否搜索热路径 |
|---|---|---|---|
| `core/card.py` | 159 | `Card`：base 四边、显示四边 `.up/.right/.down/.left`、`type_modifier`、owner、card_id、card_type、can_use；`compare_values` | 是 |
| `core/board.py` | 72 | `Board`：3×3 `List[List[Optional[Card]]]`；`__repr__` 带 ANSI 颜色（被用作 TT key！） | 是 |
| `core/player.py` | 29 | `Player`：name、hand（有序 list）、used_cards；`get_playable_cards`（秩序/混乱过滤 `can_use`） | 是 |
| `core/game_state.py` | 451 | `GameState`、`MoveRecord`；`make_move/undo_move`、`resolve_flip_with_record`、`recalculate_type_modifiers`、`copy()`(deepcopy) | 是 |
| `ai/ai.py` | 774 | `evaluate_state`、`evaluate_move`、`order_moves`、`get_state_hash`、`minimax`、`iterative_deepening_search`、`find_best_move_parallel`（名字含 parallel 但实际单线程） | 是 |
| `ai_server.py` | 2397 | Flask `/ai_move`；请求解析；未知牌处理；**信息集残局**（`_build_endgame_scenarios`、`_solve_endgame_exact`、`select_endgame_robust_move`）；胜率/推荐文案 | 部分 |
| `ai/unknown_card_handler.py` | 1265 | 规则驱动的未知牌随机采样（`random.sample`），预测牌 id = 1000 + 真实 id | 否（请求预处理） |
| `config/unknown_card_config.py` | 178 | 采样配置 | 否 |
| `ai/monte_carlo.py` | 587 | 可选 `solver=monte_carlo`：随机 playout，每次模拟 deepcopy | 否（备用求解器） |
| `app.py` | 1714 | 打包入口（PyInstaller/PyPy 都启动它）：复用 `ai_server.app`，再加 `/api/recommend`、`/api/draft/analyze`、`/api/winrate` | 否 |
| `draft_analyzer_app.py`、`npc_deck_recommender_demo.py` | — | 选拔/NPC 卡组分析的独立实现（被测试引用） | 否 |
| `console_ui.py` | 330 | rich 控制台面板、`SearchReporter` | 否 |
| `csv_compat.py` | 42 | 用 stdlib csv 替代 pandas（`read_csv/iterrows/notna`） | 否 |

数据：`data/幻卡数据库.csv`（UTF-8 BOM）共 475 行，id 1..475 连续。列：`序号,名称,Top,Bottom,Left,Right,星级,TripleTriadCardType,…`。
- 星级分布：1★53、2★98、3★170、4★83、5★69，另有 **id 474、475 两行四边全 0、星级 0、无类型**（空占位行，会进入候选池）。
- 类型：无类型 360；兽人 35、拂晓 22、帝国 28、蛮神 30。
- **10 组四边数值重复**（例如 12 与 369 都是 2,5,5,2）；`get_card_lookup` 取首个匹配。

### 1.2 真实调用链（`POST /ai_move`，solver=minimax）

```
app.py __main__ → ai_server.app.run(threaded=True)
ai_server.ai_move()                                   ai_server.py:1999
 ├─ parse_board()            ai_server.py:156   ⚠ numL→right、numR→left 左右互换（:161, :809 同样）
 ├─ parse_rules_and_open_mode()  :943           子串匹配：同数/加算/逆转/王牌杀手/同类强化/同类弱化/秩序/混乱/选拔；全明牌/三明牌→open_mode
 ├─ parse_hand(my) / parse_hand(opp, is_opponent=True)  :778
 │    未知牌(四边全0) → UnknownCardHandler 随机采样 / NPC 卡池 / 兜底 random.sample
 │    残局风险选牌 _select_unknown_cards_for_slots :246（按 _score_unknown_candidate 排序）
 ├─ 构造 Player/GameState（players[0]=红, [1]=蓝；currentPlayer 1=蓝 2=红，0=按棋盘张数推断）
 ├─ 同类规则 → game_state.recalculate_type_modifiers()
 ├─ find_best_move_parallel()                  ai/ai.py:756   清空全局 TRANSPOSITION_TABLE
 │    └─ iterative_deepening_search()          ai/ai.py:607   depth=1..max_depth(默认10)，max_time 默认10s
 │         └─ minimax(state, depth, -inf, +inf, True, ai_idx, history_table)   ai/ai.py:443
 │              ├─ is_time_up() → evaluate_state                         :473
 │              ├─ depth==0 或 棋盘满 → evaluate_state                    :477
 │              ├─ TT 查找 get_state_hash → depth>=depth 直接返回          :481
 │              ├─ moves = playable_cards × available_positions          :489
 │              ├─ order_moves → sorted(evaluate_move, reverse=True)     :498 / :427 / :346
 │              ├─ 对每个 move: make_move → minimax → undo_move          core/game_state.py:101/:156
 │              │      make_move → resolve_flip_with_record (:203) → apply_same_type_effect → recalculate_type_modifiers (:313)
 │              ├─ cutoff 时 history[(card_id,row,col)] += 2**depth (上限 1e6)
 │              └─ TT 写入 {depth, score, move}（无 bound 类型）
 └─ 若 occupancy>=5 且 对手未知牌<=2（_should_use_endgame_robust_mode :721）：
      ├─ _build_endgame_scenarios()            :418   构造 ≤16/24 个"可能世界"（GameState 深拷贝）
      ├─ select_endgame_robust_move()          :667
      │    └─ 对每个我方走法 _evaluate_endgame_move_robustly()  :613
      │         ├─ 每个场景 scenario.copy() + make_move
      │         ├─ occupancy>=5 → _solve_endgame_exact()   :541（完整 minimax + memo，无 α-β）
      │         └─ 否则 _best_immediate_reply_score()      :592（每个回应再 copy 一次）
      └─ 覆盖规则（:2171–2196）：按 safety_ratio / final_score 比较后可能替换 minimax 的走法
 → analyze_opponent_hand / calculate_win_probability / generate_move_recommendation（文案，非搜索）
```

`solver=monte_carlo` 走 `ai/monte_carlo.py`，不经过 minimax。

### 1.3 状态组成（真实字段）

**Card**（`core/card.py`）
- `base_up/right/down/left`：原始值（不变）。
- `up/right/down/left`：**显示值**。构造时等于 base（不 clamp，全 0 卡就是 0）；只在 `apply_type_modifier` 中更新为 `clamp(base+mod,1,10)`。**undo 时不恢复**（见 D-003）。
- `type_modifier`：同类修正层数（int）。
- 有效值 `get_effective_value(dir)` = `get_modified_value` = `max(1, min(10, base + type_modifier))`（所有规则比较都用这个，不用显示值）。
- `owner`：`'red' | 'blue' | None`；`card_id`：`int | None`（预测牌为 1000+真实id）；`card_type`：`str | None`；`can_use`：bool。
- 附加动态属性：`_is_generated`、`_is_prediction`（未知牌标记，仅请求层使用）。

**Board**：`grid[3][3]`，元素为 Card 引用或 None。

**Player**：`name`、`hand`（**有序 list**，顺序影响走法生成与排序并列时的选择）、`used_cards`（make/undo 不维护它，搜索中无意义）。

**GameState**：`board`、`players[2]`、`current_player_idx`、`rules: List[str]`、`move_history`。

**MoveRecord**：`row, col, card, player_idx, flipped_cards=[(r,c,原owner)], type_modifiers={"board_r_c"|"hand_i_j": 原modifier}, card_original_owner`。

### 1.4 Move 的生成、应用、回滚

- **生成**（minimax 内联，`ai.py:489`）：`for card in current_player.get_playable_cards(rules)`（秩序/混乱时过滤 `can_use`，否则全部手牌，按手牌当前顺序）`for pos in board.available_positions()`（行优先空格）。Move = `(card对象, (row, col))`。最多 5×9=45。
- **应用** `make_move(row, col, card)`：
  1. `hand_card = 手牌中第一张 card_id 相同的卡`（不是按对象身份！见 D-004）；格子非空或找不到则返回 None。
  2. 若规则含同类强化/弱化：快照所有棋盘卡（按格子）和 **双方手牌（按当前下标）** 的 `type_modifier`。
  3. `card.owner = 当前玩家颜色`；放到棋盘；`hand.remove(hand_card)`（按身份移除）。
  4. `resolve_flip_with_record`（见 §1.5）。
  5. 同类规则且打出的卡 **有类型** → `recalculate_type_modifiers()`（无类型卡不触发重算）。
  6. 切换当前玩家；记录入 `move_history`。
- **回滚** `undo_move(record)`：
  1. 从 `move_history` 线性查找并移除记录；`current_player_idx = record.player_idx`。
  2. 被翻的卡恢复原 owner。
  3. 移除棋盘上的卡，**`hand.append(card)` 追加到末尾**（手牌顺序改变，D-001）。
  4. `card.owner = 原 owner`。
  5. 按卡牌身份恢复 `type_modifier`（棋盘和双方手牌里每一张卡）。这是 **D-002 修复后**的行为；修复前按快照时的手牌下标写回已重排的手牌（见 §4 D-002）。显示值 `.up` 等不恢复（D-003）。

### 1.5 规则结算会修改哪些状态

`resolve_flip_with_record(row, col, card, flipped, flipped_set, chain_only)`，方向顺序固定为 **上、下、左、右**：

| 规则 | 判定 | 修改的状态 |
|---|---|---|
| 基础 | 相邻且 `owner != 我` 的卡，`compare_values == 1` → 记 reason='base' | 被翻卡 `owner` |
| 逆转 | `compare_values` 内：小者胜 | 同上 |
| 王牌杀手 | `compare_values` 内：双方有效值都 ∈{1,10} 且不等时 1 胜 10；同时有逆转则反过来；否则按普通/逆转比较 | 同上 |
| 加算 | 所有相邻卡（含己方）计算 `我有效值+对方有效值`；某个和出现 ≥2 次且组内至少一张敌方 → 组内敌方卡 reason='plus'。**不使用逆转/王牌杀手** | 同上 |
| 同数 | 所有相邻卡中有效值相等的集合；≥2 张且至少一张敌方 → 其中敌方卡 reason='same'。不使用逆转/王牌杀手 | 同上 |
| reason 优先级 | 已是 same/plus 的格子不会被改写；base 可被 plus/same 覆盖（同时满足 base 与 plus 的卡按 plus 处理 → 会触发连携） | — |
| 连携 Combo | 只有 reason ∈{same,plus} 的翻转卡，以 `chain_only=True` 递归：**只做基础比较**（含逆转/王牌杀手），翻得的卡 reason='base' **不再继续连携**（只有一层） | 被翻卡 `owner`；`flipped_set` 防重复 |
| 同类强化/弱化 | 落子结算后，若打出的卡有类型：统计棋盘上各类型张数 n；对棋盘 **和双方手牌** 中该类型的卡 `apply_type_modifier(+n)`（强化）或 `(-n)`（弱化）；两条规则同时存在时 **强化优先**；棋盘上不存在的类型、无类型卡 **不被触碰** | `type_modifier`、显示值 `.up/.right/.down/.left` |
| 秩序/混乱 | 只影响走法生成：`can_use` 为 False 的手牌不可出；make/undo 不改 `can_use` | 无 |
| 选拔/全明牌/三明牌 | 不影响结算，仅影响未知牌采样/分析 | 无 |

最终状态与连携遍历顺序无关（网格图无三角形，连携卡之间互不相邻）；但 `flipped_cards` 记录顺序依遍历顺序而定，undo 结果不受影响。

### 1.6 评估函数

**`evaluate_state(state, ai_idx, aggressive)`**（`ai.py:235`，叶子与终局都用它）：
- `base = 我方总牌数 - 对方总牌数`（`count_cards` 统计棋盘 + 双方手牌，按 **card.owner** 计数）。
- `position`：我方卡 +w、对方卡 -w；w=角 1.5、中心 1.2、边 1.0。
- `edge`：每张棋盘卡 `±_calculate_placement_edge_score`（暴露边强/隐藏边弱加分），所有项都是 0.5 的倍数 → 求和顺序（set 迭代）不影响结果。
- 返回 `base*bw + position*pw + edge*ew`；普通 (0.7, 0.3, 0.25)；激进模式按占用格数分 balanced(≤2)/hard(≤5)/ultra 三档。
- **终局不给胜负大分**，与非终局同一启发式。

**`evaluate_move(move, state, history, aggressive)`**（`ai.py:346`，只用于排序）：
`min(h*0.1,1000)` + （激进）卡力 + 星级特例（3★双8角 +40、4★双9 +30、5★三9 +50）+ 角15/中10 + 吃子（每个可翻敌卡 30+|差|*5 + 星级差*10，乘 capture_weight）+ 相邻己方 +10 + `placement_edge_score*5.0` − `endgame_exposure_penalty*exposure_weight`，最后 `min(score,1000.0)`。
- 星级来自 `ai.get_card_star_map()[card_id]`（预测牌 id≥1000 → 0；None → 0）。
- **吃子差值与星级特例读取的是显示值 `.up` 等**（可能陈旧，D-003），吃子判定本身用有效值。
- 成本：每次调用约 4 次 compare + 2×placement_edge_score（各 8 次 getattr 取值）+ 1 次 `_count_occupied_cells`（激进时 2 次）+ 若干 dict 查找。**每个内部节点对所有走法都调用一次**（含 depth==1 的节点）。

### 1.7 搜索：minimax / ID / TT / history

- **minimax**：fail-hard 风格的 α-β；maximizing 由调用方交替传入；根节点 maximizing=True、`ai_player_idx = 根当前玩家`。
  每个节点：计时检查 → 终止检查 → TT 查找 → 生成走法 → 排序 → 循环（每个子节点前再检查一次时间，超时就 `break`）→ cutoff 时更新 history → **无论是否 cutoff/超时都写 TT**。
- **TT**（`ai.py:15, :433, :481, :545`）：全局 `dict`；key = `str(board)`（带 ANSI 颜色的多行字符串：owner、card_id、base 值 + 修正）+ 双方手牌 card_id **无分隔符拼接**（D-005）+ 当前玩家。value = `{'depth','score','move'}`，**没有 exact/lower/upper bound 标记**（D-006）；命中条件仅为 `entry.depth >= depth`。每次请求清空。
- **history**：`dict[(card_id,row,col)] → int`，在 cutoff 时 `+= 2**depth`（上限 1e6）；按 ID 整个过程共享，传入 `evaluate_move`。
- **ID**：depth 从 1 递增到 max_depth；depth>1 时若超时则丢弃该层结果；"预测下一层耗时" 的代码只打印、**从不 break**；**深度超过剩余空格后仍然继续重复搜索**（浪费，见 baseline `mid3_ascension`）。
- **全局可变状态**：`TRANSPOSITION_TABLE`、`SEARCH_STATS`、`START_TIME`、`TIME_LIMIT` 都是模块级全局，Flask `threaded=True` 下并发请求会互相干扰（D-008）。
- `ai.get_card_star_map()` 用 **相对当前工作目录** 的路径读 CSV（D-009）。

### 1.8 `_solve_endgame_exact` 与普通 Minimax 的关系

- 完全独立的函数，**普通 minimax 不会调用它**。只在 robust 残局路径中、`_board_occupancy(base_state) >= 5` 时用于 "我方落子后" 的局面。
- 纯 minimax（**无 α-β、无深度限制、无启发式、无走法排序**），叶子值 = AI 视角牌数差（int）；memo `cache`（每个候选走法一个 dict，在该走法的所有场景之间共享），key = `_endgame_state_key`：当前玩家 + 棋盘 9 格 (card_id, owner, mod, base4) + 双方手牌 (card_id, owner, can_use, mod, base4) **按手牌顺序** + rules 元组。
- 因为普通 minimax 终局用启发式分数，而 exact solver 用真实牌数差，**两者结论可能不同**，robust 层据此可以覆盖 minimax 的走法。

### 1.9 隐藏牌 / scenario solver 如何构造可能世界

1. 请求解析阶段：对手的未知槽位先填入 "预测牌"（handler 随机采样，id=1000+真实id；或从 NPC 卡池）。`_is_generated_unknown_card` 以 `_is_generated` / `_is_prediction` / id 为 None / id≥1000 判定未知。
2. minimax 在 **单一** 采样世界上搜索（即把预测牌当作真实牌）。
3. 若 occupancy≥5 且对手未知槽 ≤2：`_build_endgame_scenarios`
   - `_build_legal_unknown_candidates`：遍历全卡库（可选 NPC 卡池过滤），排除棋盘与双方已知手牌中出现过的真实 id，克隆为候选（**base id，非 1000+**，标记 generated），按对手已知牌检查卡组星级限制（★4+ ≤2、★5 ≤1）。为空时退回 handler 随机生成。
   - `_unique_unknown_candidates` 去重 → 按 `(_score_unknown_candidate 威胁分, 显示值之和, -card_id)` 降序排序。
   - `sample_count = min(24, max(16, unknown*12))`；`candidate_limit = min(len, max(sample_count*unknown, unknown))`；对前 `candidate_limit` 个候选做 `itertools.permutations(·, unknown)`，按 id 元组去重 + 合法性检查，取前 `sample_count` 个 → 每个都 `base_state.copy()` 并替换未知槽位（**不重新计算同类修正**）。
   - 排列顺序决定了场景集合高度偏向排名第一的候选（例如 2 张未知时，前 24 个排列几乎都是 (c0, cX)）。这是 Python 的实际行为，C# 必须一致。
4. `select_endgame_robust_move`：对每个我方走法计算 avg/worst/safety_ratio/corner_risk，非激进模式优先选 safety_ratio==1 的走法，再按 (safety, final, robust, -corner) 排序；激进模式按 (final, robust, safety, -corner) 排序。
5. 覆盖条件见 `ai_server.py:2171–2196`。

### 1.10 clone / copy / deepcopy 使用情况

| 位置 | 方式 | 频率 |
|---|---|---|
| minimax / `_solve_endgame_exact` | make/undo，**不复制** | — |
| `GameState.copy()` | `deepcopy(board)` + 每个 Player 一次 deepcopy | — |
| `_copy_state_with_unknown_assignment` | copy() + Card.copy() | 每个场景 1 次 |
| `_evaluate_endgame_move_robustly` | `scenario.copy()` | 走法数 × 场景数（≤16×24） |
| `_best_immediate_reply_score` | `state_after_my_move.copy()` | **每个回应走法 1 次** |
| `ai_move` 预测棋盘 | copy() | 1 次 |
| MonteCarlo | 每次模拟 deepcopy 棋盘 + 复制手牌 | 模拟次数 × 走法数 |
| `count_captured_cards` | `card.copy()` + make/undo（undo 后手牌里换成了复制品，D-010） | 1 次（文案） |

### 1.11 每个搜索节点的对象分配（Python）

每个 minimax 节点大约分配：`SearchResult` 对象；`path + [move]` 新 list（O(深度) 复制）；`get_state_hash` 生成约 100+ 个临时对象（3×3 嵌套 list、9× `display_multiline` 的 4 个 f-string、颜色包裹、13 行字符串、join）；TT entry dict；`get_playable_cards` list；**每张可出的牌都重新调用一次** `available_positions()`；每个走法 2 个 tuple；`sorted` 新 list + key list；每次 `evaluate_move`：`card_edges` list、方向 list 字面量、`_calculate_placement_edge_score` 中的 2 个 set + list + listcomp。
每次 `make_move`：`MoveRecord`、`flipped_cards` list、同类规则下 `{f"board_{r}_{c}": …}` 字符串 key dict、`to_flip` dict、`flipped_set` set、`add_flip` 闭包、方向 list、加算的 `plus_list/sum_map/valid_sums`、同数的 `same_list`。
每次 `undo_move`：`move_history` 线性查找 + `list.remove`、对字符串 key 做 `split`。
叶子 `evaluate_state`：每次调用都新建 `position_weights` dict 字面量。

---

## 2. 已发现的性能热点

cProfile（CPython 3.11.4，fixed depth，TT 每次清空）：

| 场景 | 节点 | 总耗时(profiled) | 主要热点（cumtime） |
|---|---|---|---|
| open_same_plus d4 | 12,665 | 3.35 s | `evaluate_move` 80,109 次 = **2.04 s（≈61%）**；其中 `_calculate_placement_edge_score` 1.32 s；`get_modified_value` 72 万次 0.80 s；`get_effective_value` 0.95 s；`_count_occupied_cells` 0.27 s；`evaluate_state` 0.56 s；`make_move` 0.33 s；`Board.__repr__`(TT key) 0.17 s |
| mid3_ascension d5 | 2,322 | 0.81 s | `make_move` 0.24 s（其中 `recalculate_type_modifiers` 0.17 s，`apply_type_modifier` 2.2 万次）；`evaluate_move` 0.24 s；`get_modified_value` 20.6 万次；`evaluate_state` 0.16 s |

结论（Python 当前最可能的 NPS 瓶颈，按影响排序）：
1. **走法排序**：每个内部节点对所有走法都调用 `evaluate_move`（约 6.3 次/节点），且 depth==1 节点也排序。
2. **字符串方向分派**：`getattr(self, f'base_{direction}')` + clamp，每节点约 57 次。
3. **TT key 构造**：每节点把棋盘渲染成带颜色的多行字符串。
4. **make/undo 中的临时容器**（dict/set/闭包/字符串 key），同类规则下全盘重算修正值。
5. 叶子 `evaluate_state` 全盘扫描。
6. 算法层面：TT 命中率极低（open d4：32 次命中 / 15,807 节点）；ID 在超过剩余空格后仍重复搜索；ID 时间预测从不生效。
7. robust 残局：大量 deepcopy（走法 × 场景，以及每个回应走法）。

---

## 3. Baseline（Phase 1，2026-10-07）

### 3.1 环境

- CPU：13th Gen Intel Core i7-13650HX（14C/20T），32 GB RAM，Windows 11 Pro for Workstations 10.0.26200。
- CPython 3.11.4（flask/numpy/rich 已安装）；**PyPy 未安装**，生产环境 PyPy 构建的 baseline 暂缺（见 §9 Q-6）。
- .NET SDK：5.0.405 / 10.0.101 / 10.0.401。

### 3.2 测试

`python -m pytest -q` → **26 passed**（1.26 s）。文件：`tests/test_game_rules.py`(17)、`tests/test_draft_selection_analyzer.py`(6)、`tests/test_npc_deck_recommender_demo.py`(3)。

### 3.3 Benchmark 方法

脚本见附录 A（Phase 1 放在仓库外的临时目录；Phase 2 应提交到 `tools/bench/py_bench.py`）。
- 全部为已知牌（无随机），固定深度测试使用 `TIME_LIMIT=1e9`，每次清空 TT、history 用新的 `{}`，直接调用 `ai.minimax(...)`。
- 3 次运行的节点数/分数/走法 **完全一致**（已断言）→ 固定深度搜索是确定性的。耗时取 3 次中位数。
- 场景（卡牌 id 来自 `幻卡数据库.csv`，`Card(Top,Right,Bottom,Left)`）：

| 场景 | 棋盘 (格子: id/owner) | 红手牌 | 蓝手牌 | 先手 | 规则 |
|---|---|---|---|---|---|
| open_none | 空 | 56,64,27,46,7 | 59,51,35,48,8 | 红(0) | 无 |
| open_same_plus | 空 | 同上 | 同上 | 红 | 同数,加算 |
| mid3_ascension | (1,1)40/红 (0,0)41/蓝 (2,2)52/红 | 42,43,31,15 | 53,54,47,14 | 蓝(1) | 同类强化 |
| mid4_rev_ace | (0,1)1/红 (1,0)2/蓝 (1,2)3/红 (2,1)4/蓝 | 56,64,27 | 59,51,35 | 红 | 逆转,王牌杀手 |
| end5_same_plus | (0,0)56/红 (0,1)59/蓝 (1,1)64/红 (2,0)51/蓝 (2,2)27/红 | 46,7 | 35,48 | 蓝 | 同数,加算 |

### 3.4 固定深度 minimax（CPython，中位数）

| 场景 | depth | nodes | sec | NPS | score | best move [id,r,c] | cutoffs | tt_hits |
|---|---|---|---|---|---|---|---|---|
| open_none | 1 | 46 | 0.003 | 15,333 | 5.325 | [56,2,0] | 0 | 0 |
| open_none | 2 | 214 | 0.016 | 13,375 | -4.56 | [64,2,1] | 42 | 0 |
| open_none | 3 | 4,282 | 0.174 | 24,609 | 4.675 | [46,0,0] | 335 | 0 |
| open_none | 4 | 15,807 | 1.089 | 14,515 | -2.44 | [27,2,2] | 3,497 | 32 |
| open_same_plus | 1 | 46 | 0.001 | — | 5.325 | [56,2,0] | 0 | 0 |
| open_same_plus | 2 | 214 | 0.016 | 13,375 | -4.56 | [64,2,1] | 42 | 0 |
| open_same_plus | 3 | 4,488 | 0.193 | 23,253 | 4.675 | [46,0,0] | 336 | 0 |
| open_same_plus | 4 | 12,665 | 0.933 | 13,574 | -2.44 | [27,2,2] | 2,913 | 34 |
| mid3_ascension | 1 | 25 | 0.001 | — | 14.61 | [53,1,2] | 0 | 0 |
| mid3_ascension | 2 | 68 | 0.007 | 9,714 | 7.81 | [53,1,2] | 23 | 0 |
| mid3_ascension | 3 | 416 | 0.033 | 12,606 | 13.76 | [53,1,2] | 43 | 0 |
| mid3_ascension | 4 | 957 | 0.101 | 9,475 | 6.96 | [53,1,2] | 346 | 28 |
| mid3_ascension | 5 | 2,322 | 0.221 | 10,506 | 11.61 | [53,1,2] | 550 | 73 |
| mid4_rev_ace | 1 | 16 | 0.001 | — | 6.2 | [56,2,0] | 0 | 0 |
| mid4_rev_ace | 2 | 53 | 0.004 | 13,250 | -0.5 | [56,2,0] | 13 | 0 |
| mid4_rev_ace | 3 | 165 | 0.010 | 16,500 | 2.86 | [56,2,0] | 35 | 0 |
| mid4_rev_ace | 4 | 402 | 0.025 | 16,080 | -0.375 | [27,2,2] | 98 | 33 |
| mid4_rev_ace | 5 | 587 | 0.037 | 15,864 | 0.165 | [27,2,2] | 197 | 102 |
| end5_same_plus | 1 | 9 | 0.000 | — | 3.335 | [35,2,1] | 0 | 0 |
| end5_same_plus | 2 | 34 | 0.002 | 17,000 | -0.265 | [35,2,1] | 5 | 0 |
| end5_same_plus | 3 | 75 | 0.005 | 15,000 | 2.585 | [35,2,1] | 19 | 0 |
| end5_same_plus | 4 | 111 | 0.008 | 13,875 | 2.76 | [48,1,0] | 36 | 6 |

（score 为 6 位小数四舍五入，仅供参考；Phase 3 的 golden 数据必须保存 `float.hex()`。）

### 3.5 服务端等价调用（`find_best_move_parallel(max_depth=10, max_time=10)`，单次）

| 场景 | 完成深度 | 总节点 | 耗时 | NPS | 走法 | 各深度节点 |
|---|---|---|---|---|---|---|
| open_none | 5 / 9 | 170,643 | 10.0 s | 17,064 | [27,2,2] | 46, 214, 4281, 17080, 86716 |
| open_same_plus | 5 / 9 | 159,616 | 10.0 s | 15,949 | [27,2,2] | 46, 214, 4460, 12857, 91318 |
| mid3_ascension | 10（剩余只有 6 格） | 31,627 | 3.47 s | 9,106 | [53,1,2] | 25, 68, 418, 912, 2367, 3795, 3689, 5709, 6686, 7958 |

注：ID 中 depth3/4 的节点数与 fixed-depth 不同，原因是 history 在各层之间累积。

**D-002 修复后的更新（Phase 3 golden，Python 与 C# 位级/节点级一致）**：`mid3_ascension` 的 §3.4 固定深度节点数变为 d3 = 403、d4 = 889（cutoffs 327，tt_hits 39）、d5 = 2,404（cutoffs 557，tt_hits 49），根分数与走法不变（见 D-002 修复记录）；§3.5 的 ID（max_depth=10、时间无限）各层节点数变为 [25, 68, 403, 900, 2403, 3520, 3566, 5441, 5949, 8225]，合计 30,500（修复前 31,627）。其余场景不受影响。

### 3.6 精确残局 `_solve_endgame_exact`（从根局面直接求解）

| 场景 | make_move 次数 | cache 条目 | 耗时 | make/s | 值 |
|---|---|---|---|---|---|
| end5_same_plus | 217 | 122 | 0.005 s | 48,099 | 1 |
| mid4_rev_ace | 4,707 | 1,936 | 0.056 s | 83,889 | -2 |

### 3.7 TT 对固定深度结果的影响（D-006 实测）

关闭 TT 写入后，open_none d4、mid3_ascension d5、mid4_rev_ace d5、end5_same_plus d4 的根分数与最佳走法都 **不变**（只是节点数增加：15,839 / 2,619 / 720 / 117）。即在这些局面上，无 bound 的 TT 没有造成错误结果，但理论上仍不安全。

---

## 4. Python/C# Semantic Discrepancy（Python 已知怪异行为）

状态：`OPEN-EMULATE` = C# 按 Python 实际行为复现；`DECISION` = 需用户决定；`FIXED-BOTH` = 已在两边同时修复。

| ID | 现象（均已在 Phase 1 用脚本实测证实，除非注明） | 影响 | 建议 | 状态 |
|---|---|---|---|---|
| D-001 | undo 把卡 `append` 到手牌末尾 → make+undo 后手牌顺序改变（实测 `[56,64,27,46,7]` → 打出 64 再撤销 → `[56,27,46,7,64]`） | 走法生成顺序 → 稳定排序中并列走法的先后 → best move 选择 / cutoff / 节点数 / history | C# 复现（有序手牌 + 移除下标 + 追加） | OPEN-EMULATE |
| D-002 | （**已修复**）原行为：同类规则下，undo 按 **快照时的手牌下标** 恢复 `type_modifier`，但手牌已重排 → 修正值写到错误的卡上（Phase 1 实测：拂晓卡 46 从 +1 变 0，帝国卡 31 从 0 变 +1）。无类型卡也可能被写入非零修正且永远不会被重算。**修复**：make 时按卡牌对象（C# 按卡实例）保存 `type_modifier`，undo 时按身份写回；D-001（undo 追加到手牌末尾）和 D-003（显示边不恢复）**不变**。回归测试：Python `tests/test_undo_type_modifiers.py`（3 个，旧代码上全部失败）、C# `D002_fixed_undo_restores_modifiers_by_card_identity` + 全量 differential（golden generator `py_rules_trace/2`） | 修复前：改变规则结算结果；修复后：同类规则下搜索结果不再依赖访问顺序 | 两边同时修复（用户在 Phase 2 之后批准 Q-1，2026-10-07） | **FIXED-BOTH**
| D-003 | 显示值 `.up/.right/.down/.left` 只在 `apply_type_modifier` 中更新，undo 不恢复；全 0 卡的显示值为 0，而有效值为 1 | `evaluate_move` 的吃子差值、星级特例，以及 `_score_unknown_candidate` 读取的是陈旧值 → 影响排序 | C# 为每个卡实例保存 4 字节 DisplayEdges 并完全复现（开销很小） | OPEN-EMULATE |
| D-004 | 同一手牌里有重复 `card_id`（例如两个 None 占位牌、四边相同的两张牌被识别成同一 id、预测牌重复）时，`make_move` 移除的是第一张同 id 的卡，放到棋盘上的却是传入的那张 → undo 后出现重复引用、原卡丢失（实测） | 状态损坏 | C# 按手牌槽位身份处理；遇到重复 id 输入时报告并在 parity 中排除；是否在 Python 侧修复由用户决定 | **DECISION**（Q-2） |
| D-005 | TT key 中手牌 id 无分隔符拼接：`[1,23]` 与 `[12,3]` 的 key 相同（实测碰撞） | 罕见的错误 TT 命中 | ReferenceCompat 模式复现；Fast 模式改用 Zobrist（无此问题） | OPEN-EMULATE（compat） |
| D-006 | TT 不存 bound 类型；cutoff/超时后的部分结果也按精确值存储，命中条件只看深度 | 理论上不正确（§3.7 的样本中未观察到影响） | ReferenceCompat 复现；Fast 模式使用 exact/lower/upper | OPEN-EMULATE（compat） |
| D-007 | 终局（棋盘满）仍用启发式 `evaluate_state`，没有胜负大分 | 搜索可能偏好"启发式分高但输掉"的线 | 不改，复现 | OPEN-EMULATE |
| D-008 | `ai.ai` 的 TT/统计/计时都是模块级全局，Flask 多线程并发请求会互相干扰 | 并发时结果错误 | C# 每次搜索独立上下文（行为上等价于单请求） | OPEN（C# 天然不复现，记录即可） |
| D-009 | `ai.get_card_star_map()` 用相对 cwd 的路径 | 从别的目录启动时会崩溃 | C# 用程序目录定位数据 | OPEN（集成时处理） |
| D-010 | `count_captured_cards` 用复制品 make/undo → 手牌中的原卡被替换为复制品，并改变顺序 | 只影响文案和输出预测棋盘 | Phase 8 移植文案时复现或修复（届时决定） | OPEN |
| D-011 | 请求解析中 `numL→right`、`numR→left` 左右互换（`ai_server.py:161, :809`） | 协议语义，必须保持 | Phase 7/8 必须复现 | OPEN-EMULATE |
| D-012 | 数据库 id 474/475 为全 0 空行，会进入候选池（有效值视为 1，显示值为 0） | 未知牌候选质量 | 复现；是否过滤由用户决定 | OPEN |
| D-013 | 连携只有一层（连携翻得的卡不再继续连携），与常见的游戏实现可能不同 | 规则语义 | Python 为准，复现；若要改动需用户确认 | OPEN-EMULATE |
| D-014 | ID 在深度超过剩余空格后仍重复搜索；"预测下一层耗时" 的代码从不生效 | 只浪费时间，固定深度结果不受影响 | ReferenceCompat 复现；Fast 模式可以在不改变结果的前提下提前终止（需 benchmark 证明） | OPEN |

---

## 5. C# 目标架构

### 5.1 原则

- 单个可变 `GameState`（sealed class，内部全部是固定长度数组 / InlineArray），由搜索器独占；**GenerateMoves → MakeMove → Search → UndoMove**。
- 两种搜索配置：
  - **ReferenceCompat**：逐节点复现 Python（同样的排序、history、TT 语义 D-005/D-006、ID 流程）。用于证明移植正确，只在固定深度/无限时间下做 parity。允许稍慢，但仍不得在热路径分配。
  - **Fast**：Zobrist + 带 bound 的 TT 等。必须证明与 ReferenceCompat 的根分数一致（D-002 已在两边修复，同类规则下这一点现在可证明）。

### 5.2 数据布局（Phase 2 实现）

| 类型 | 形式 | 说明 |
|---|---|---|
| `CardDef` | `readonly struct`（Id ushort, Up/Right/Down/Left byte, Star byte, TypeId byte） | 静态卡库，按 id 索引的数组 |
| `CardId` | `int`，`NoId = -1` 表示 Python 的 None | 预测牌保留 1000+id |
| `Owner` | `byte`：0=red 1=blue 2=None | 与 Python 三值一致 |
| `RuleSet` | `[Flags] enum : ushort` | Same, Plus, Reverse, FallenAce, Ascension, Descension, Order, Chaos, Draft；OpenMode 单独存 |
| `Move` | `readonly struct`（byte Inst, byte Cell） | 2 字节，存入预分配的 `Move[ply][45]`。**身份是卡实例而不是手牌槽位**（审查 H-3 整改；最初的设计写的是 `HandSlot`，会被 D-001 的手牌重排悄悄改变含义） |
| 卡实例表 | SoA：`BaseEdges[16*4]`、`EffEdges[16*4]`（clamp 缓存）、`DisplayEdges[16*4]`（D-003）、`CardId[16]`、`TypeId[16]`、`Owner[16]`、`TypeMod[16] sbyte`、`CanUse[16]` | 一局最多约 10 张卡，容量 16 |
| 棋盘 | `sbyte Cell[9]`（实例下标或 -1）+ `ushort OccupiedMask` | |
| 手牌 | `byte Hand[2][5]` + `byte HandCount[2]`（有序；移除=左移，undo=追加） | 复现 D-001 |
| 邻接表 | 静态 `NeighborCount[9]`、`Neighbor[9][4]`(cell, mySide, oppSide)，顺序为上、下、左、右 | 与 Python 方向顺序一致 |
| Undo 记录 | 预分配 `UndoRecord[]` 栈（struct）：cell、placedInst、player、placedOrigOwner、`ushort FlipMask`、`bool HasModSnap`、modSnapshot（棋盘 9 + 手牌 2×5，按 Python 的 key 语义） | 翻转只会在两方之间切换 → 9 位 mask 足够 |
| 浮点 | `double`，严格保持 Python 的运算与累加顺序；禁止 `Math.FusedMultiplyAdd`；不得重排表达式 | 位级 parity |
| 排序 | 自写稳定插入排序（≤45 个元素，按分数降序，相等时保持生成顺序） | 等价于 Python `sorted(reverse=True)` |

### 5.3 适合 MakeMove/UndoMove 的逻辑

`make_move`、`resolve_flip_with_record`（含加算/同数/连携/逆转/王牌杀手）、`recalculate_type_modifiers`（含按身份的修正值快照/恢复，D-002 已修复）、`count_cards`、`is_game_over`；`_solve_endgame_exact` 与 robust 评估中的 "copy + make" 都可以改为在单一状态上 make/undo，或从基础状态快速重建（场景状态构造一次，之后只做 make/undo）。

### 5.4 适合固定数组 / bitmask / readonly struct 的数据

- bitmask：棋盘占用（9 bit）、红方所有权（9 bit）→ 用 popcount 统计棋盘张数；翻转集合（9 bit）；规则集合。
- readonly struct：`CardDef`、`Move`、`NeighborEntry`、TT entry（Phase 4）。
- 固定数组：卡实例表、手牌、每层的走法缓冲区 `Move[10][45]` 与分数缓冲区 `double[10][45]`、undo 栈、history 表（`int[cardIdKey][9]`，card_id 的取值域包括 None 和 1000+，用小映射表压缩）。

---

## 6. 暂时不迁移的内容

到 Phase 7 之前 **不迁移**（Phase 8 才处理）：
- Flask HTTP 层、`console_ui.py`（rich 面板）、ASCII 启动画面、PyInstaller/PyPy 构建脚本与 `pypy_launcher.cs`。
- `UnknownCardHandler` 与 `parse_hand` 的随机采样：基于 Python `random`（MT19937 + Python 特有的 `random.sample` 算法），无法、也不应该在 C# 中位级复现；Phase 8 只要求分布等价，并且在 C# 内部可用种子复现。在 Phase 7 中由 Python 完成采样后再交给 C#。
- `ai/monte_carlo.py`（备用求解器，同样依赖随机）。
- 文案/分析：`analyze_opponent_hand`、`calculate_win_probability`、`generate_move_recommendation`、`analyze_*`、`format_*`。
- `app.py` 的 `/api/recommend`、`/api/draft/analyze`、`/api/winrate`，以及 `draft_analyzer_app.py`、`npc_deck_recommender_demo.py`。
- `GameState._estimate_unknown_hand_types` / `_get_card_type_distribution`（没有被调用的死代码，用到 pandas）。

---

## 7. Phase 定义、范围与完成标准

### 阶段划分的调整（相对用户最初建议）及原因

1. **History 与走法排序放进 Phase 3，而不是 Phase 4**：Python 的 `evaluate_move` 直接读取 history 表，minimax 也会写入它；不包含它们就无法做节点级 parity。Phase 3 同时实现 Python 语义的 TT 模拟（ReferenceCompat），因为 TT 命中会改变节点数和 history。
2. **Phase 4 改为 "Fast 搜索"**：Zobrist + 带 bound 的 TT 是行为上的改进，必须以 Phase 3 的 ReferenceCompat 为对照来验证。
3. **Phase 5 加入信息集 robust 残局**（不只是 `_solve_endgame_exact`）：两者在 Python 中是同一条调用路径，并且场景构造部分（合法候选枚举、排序、排列）是确定性的，可以做 parity。
4. **新增 Phase 8 "全量服务移植"**：用户的目标是把整个项目移植到 C#。随机采样、HTTP、文案等部分无法或不必做位级 parity，因此放在引擎被证明正确之后。原来的 CP-SAT 阶段顺延为 Phase 9。
5. **root parallelism 放在 Phase 6 末尾，且需单独批准**（CLAUDE.md §5）。

### Phase 1 — Python 架构 / profiling（DONE）

范围：阅读全部求解相关代码、运行测试、profiling、benchmark，产出 CLAUDE.md 与本文件。
完成标准：✅ 调用链 ✅ 状态组成 ✅ make/undo ✅ copy 使用 ✅ 每节点分配 ✅ hash/TT ✅ ordering/history ✅ 评估成本 ✅ 规则修改的状态 ✅ exact 残局关系 ✅ scenario 构造 ✅ NPS 瓶颈 ✅ 适合 make/undo 的逻辑 ✅ 数据结构建议 ✅ 暂不迁移的内容 ✅ 测试 baseline ✅ benchmark baseline。

### Phase 2 — Card / Move / RuleSet / GameState / MakeMove / UndoMove

**实现范围**
1. 解决方案骨架：`csharp/TTC.Siren.sln`，项目 `TTC.Siren.Core`、`TTC.Siren.Tests`（xUnit）、`TTC.Siren.Bench`（控制台）；`net10.0`，`<Nullable>enable</Nullable>`，`<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`。
2. `CardDb`：读取 `data/幻卡数据库.csv`（UTF-8 BOM，按列名读取，包含 474/475 空行）；id→CardDef、星级、TypeId（类型名 ↔ id 映射）；按四边查 id 的首个匹配（为后续阶段准备，语义与 `get_card_lookup` 一致）。
3. `RuleSet` 与 `RuleSet.ParsePythonRuleString`（等价于 `parse_rules_and_open_mode` 的子串匹配，包括 open_mode）。
4. `GameState`：§5.2 中的布局；可从 "Python 状态描述"（JSON）构造，也能导出为同样格式，用于 parity。
5. `GenerateMoves(Span<Move>)`：手牌当前顺序 ×（秩序/混乱时只取 `can_use`）× 行优先空格。
6. `MakeMove/UndoMove`：完全按 §1.4/§1.5 的语义，包括 D-001、D-003 的复现（用 `// PY-QUIRK(Dxxx)` 标注；D-002 已修复，见 §4）；同类规则只在打出的卡有类型时重算；两条同类规则同时存在时强化优先。
7. `CountCards / IsGameOver / GetWinner`（按 card.owner 统计棋盘和手牌）。
8. 输入校验：同一手牌内 card_id 重复（D-004）时抛出明确的异常（parity 工具会排除这类输入）；手牌数量超过容量也抛异常。
9. Python parity 工具 `tools/parity/py_rules_trace.py`（只 import Python 代码，不修改）：固定种子生成随机局面（随机真实卡牌，包括有类型卡、全 0 卡、预测 id 1000+、None id 但不重复；随机规则子集；随机 `can_use`；随机初始棋盘与 owner；随机先手），执行随机的 make / undo 交错序列，每一步之后导出完整状态（每个实例的 owner、mod、display4、base4、id、type、can_use；9 格；有序手牌；当前玩家；`available_moves` 列表）→ JSONL golden。
10. 把 `tests/test_game_rules.py` 中与规则相关的测试移植为 C# 测试：compare/王牌杀手/逆转、同数正反例、加算使用修正值、连携/不连携、同类强化/弱化计数、undo 恢复 owner。
11. `tools/bench/py_bench.py`（即附录 A 的脚本）提交进仓库；C# Bench 加入 make/undo 吞吐量 microbenchmark。

**验收条件**
- [x] `dotnet build csharp/TTC.Siren.sln -c Release`：0 warning，0 error（`--no-incremental` 全量重建确认，`TreatWarningsAsErrors=true`）。
- [x] `dotnet test`：24 个测试全部通过（0 失败、0 跳过；D-002 修复后复测同样 24/24）；`tests/test_game_rules.py` 中全部 9 个规则相关测试已移植（`PortedRuleTests`）。
- [x] Differential：20,000 条 golden 序列（seed 20261007）、182,910 个 make/undo 步骤（110,937 make + 71,973 undo），每步之后的完整状态、合法走法列表及其顺序、红/蓝张数与胜者、make 翻转的格子集合都与 Python **完全一致**，0 个不一致。8 条结算规则的全部 256 个子集各出现 ≥78 次（含 8 个单独规则和全部 28 个两两组合）。
- [x] 覆盖率计数器（测试输出，按 make 计数；以下为 D-002 修复后、generator `py_rules_trace/2` 的数字，首次交付时数字相近）：基础翻面 44,092；同数翻面 648；加算翻面 4,640；连携翻面 1,937；逆转取胜比较 23,840；王牌杀手 1v10=299、10v1=300、1v10+逆转=340、10v1+逆转=283；强化重算 25,076、弱化重算 12,411；"undo 中有修正值被改动并按身份恢复" 24,272 次。全部 ≥100，测试内以断言保证。
- [x] 分配测试：1,000,002 次 Make/Undo 对（6 个场景，含同类强化、加算、同数、逆转、王牌杀手）→ `GC.GetAllocatedBytesForCurrentThread()` 增量 **0 字节**。
- [x] Make/Undo 吞吐量写入 §10 benchmark 历史（只记录，不设门槛）。
- [x] `python -m pytest -q` 仍为 26 passed；没有任何已跟踪的 Python 文件被修改。
- [x] 本文件更新：Phase 2 状态、新发现的差异、benchmark、设计决定、已知限制。

**不在 Phase 2 范围**：评估、排序、搜索、TT、残局、请求解析、未知牌、HTTP。

#### Phase 2 交付记录（2026-10-07）

**新增文件**（均为未跟踪新文件，未改动任何已有文件）
- `csharp/TTC.Siren.sln`、`csharp/Directory.Build.props`（net10.0、Nullable、`TreatWarningsAsErrors`）、`csharp/.gitignore`
- `csharp/src/TTC.Siren.Core/`：`Primitives.cs`（Owner/Side/RuleSet/OpenMode/RuleParser/Move/CardTypes/InvalidGameStateException）、`CardDb.cs`、`Snapshot.cs`（导入/导出与规范化字符串）、`GameState.cs`、`CoverageCounters.cs`、`ReferenceScenarios.cs`（§3.3 的 5 个基准局面 + 全规则压力局面 + 随机 playout 驱动）
- `csharp/src/TTC.Siren.Bench/Program.cs`：Make/Undo 吞吐量
- `csharp/tests/TTC.Siren.Tests/`：`RulesTraceDifferentialTests`、`PortedRuleTests`、`GameStateBehaviourTests`（D-001/2/3/4、秩序/混乱过滤、规则解析、CardDb 对照 Python）、`AllocationTests`、`GoldenFiles`
- `tools/parity/py_rules_trace.py`（golden 生成器，只 import `core.*` 与 `csv_compat`）、`tools/bench/py_bench.py`（附录 A，已改为输出 `score_hex`）、`tools/parity/golden/.gitignore`

**运行方式**
- 构建 / 测试：`dotnet build csharp/TTC.Siren.sln -c Release`、`dotnet test csharp/TTC.Siren.sln -c Release`
- golden 数据按需生成：测试发现 `tools/parity/golden/rules_trace_s20261007_n20000.jsonl.gz`（≈6.4 MB，被 git 忽略）不存在时会调用 `python tools/parity/py_rules_trace.py` 生成（≈20 s，生成失败则测试失败，不会跳过）。也可手动执行 `python tools/parity/py_rules_trace.py --out <file> --seed 20261007 --count 20000`。
- 吞吐量：`dotnet run -c Release --project csharp/src/TTC.Siren.Bench`

**设计决定（相对 §5.2 的偏差已标明）**
1. 解决方案使用经典 `.sln`（`dotnet new sln --format sln`；SDK 10 默认生成 `.slnx`）。
2. **翻转记录改用两个 9 位 mask（`FlipMask` + `FlipFromNoneMask`）**，而不是 §5.2 假设的单个 mask：Python 的 `owner` 是三值，棋盘上 `owner=None` 的卡会被 `owner != 我` 判定为敌方并被翻面，undo 需恢复为 None。golden 中约 2% 的棋盘卡和约 3% 的手牌卡 owner 为 None（手牌卡另有约 3% 为"错误颜色"），已被 differential 覆盖。
3. **没有实现 `EffEdges`（clamp 缓存）**：有效值 `clamp(base+mod,1,10)` 现场计算（一次加法 + 两次比较）。没有 benchmark 数据证明需要缓存，按 CLAUDE.md §6 不做；Phase 6 若 profiler 指向它再评估。
4. 实例表容量 `MaxInstances = 19`（棋盘 9 + 手牌 2×5），手牌容量 5，undo 深度 16（≥ 9 个空格）。
5. `MakeMove` 对非法走法（手牌下标/格子越界、格子已占用）抛 `InvalidOperationException`（Python 返回 `None`）。搜索只会传入 `GenerateMoves` 生成的走法。
6. D-004：`FromSnapshot` 发现同一手牌内 `card_id` 重复（包括两个 `None`）时抛 `InvalidGameStateException`；golden 生成器保证手牌内 id 唯一（最多一张 `None` id）。
7. 同数/加算/基础翻面的"插入顺序"（基础→加算→同数；dict 保序语义）被逐项复现，包括"同数/加算 reason 不会被后续改写、基础可被覆盖"。最终状态本来与顺序无关（§1.5），但 `LastFlipMask` 与 Python 的 `record.flipped_cards` 集合在 182,910 步中全部一致。
8. 同类强化/弱化的修正值快照**按卡实例**保存（`_snapshot[depth * MaxInstances + inst]`，覆盖全部实例），undo 按实例写回——这是 **D-002 修复后**的语义（Phase 2 首次交付时曾按 Python 旧的下标 key 语义复现 D-002，Q-1 批准后改为修复，见下方"D-002 修复记录"）；显示边 `_display` 作为每实例 4 字节状态保存，仅在 `ApplyTypeModifier` 中更新，undo 不恢复（D-003）。代码中以 `// PY-QUIRK(D-001/003/004/013)` 标注（D-002 已修复，代码注释为 `D-002 (FIXED-BOTH)`）。
9. 统计覆盖率用可选的 `GameState.Coverage`（搜索路径上为 `null`，开销为一次 null 判断）。
10. `used_cards`（Python 在 `play_card` 中追加、undo 不撤销）与搜索无关，C# 不建模。
11. `CardTypes` 为固定表（兽人/拂晓/帝国/蛮神）；未知类型名会抛异常（CardDb 测试保证数据库中只出现这 4 种）。

**新发现的差异 / 观察**（均无需用户决定，未新增 BLOCKED）
- 无新的 Python 缺陷。D-001、D-003、D-004、D-013 的行为均经 golden 逐步验证；D-002 见下方修复记录。
- `CardDb` 与 Python 对照：475 行、星级分布 (0★:2, 1★:53, 2★:98, 3★:170, 4★:83, 5★:69)、类型分布 (无:360, 兽人:35, 拂晓:22, 帝国:28, 蛮神:30)、465 个不同的四边键（`get_card_lookup` 首个匹配）与 §1.1 的记录一致。
- 变异测试：把同数条件 `cnt >= 2` 临时改成 `cnt >= 3` 后，differential 在序列 141 的第 0 步报告不一致（已还原），说明 golden 对规则细节敏感。

#### D-002 修复记录（Q-1，2026-10-07，用户批准）

- **Python**（唯一改动的 Python 业务文件 `core/game_state.py`，+13/−22 行）：`make_move` 把 `type_modifiers` 由 `{"board_r_c"|"hand_i_j": mod}` 改为 `[(card, mod), ...]`（棋盘卡 + 双方手牌，按对象）；`undo_move` 对每个 `(card, mod)` 直接写回 `card.type_modifier`。`MoveRecord.type_modifiers` 由 dict 变为 list（grep 确认仅 `core/game_state.py` 内部使用）。D-001（undo 追加到手牌末尾）与 D-003（显示边不恢复）**未改动**。
- **Python 回归测试**：新增 `tests/test_undo_type_modifiers.py`（3 个：手牌 `[A+1,B,C+2]` 打出 B 再撤销后 `(A,B,C)=(1,0,2)`；含同类卡的真实场景；300 个固定种子的随机 make/undo 往返，修正值逐卡复原）。**在修复前的代码上 3 个全部失败，修复后全部通过**。`python -m pytest -q`：29 passed（= 原 26 + 新增 3；CLAUDE.md §7 中的"26 passed"是 Phase 1 基线，原有 26 个测试均未改动且通过）。
- **C#**：`GameState` 的修正值快照改为按实例（`_snapshot[depth * MaxInstances + inst]`），undo 按实例写回；移除不再需要的 `UndoRecord.HandSlot`。覆盖率计数器 `UndoMisrestore` 改为 `UndoModifierChanged`。测试：`D002_fixed_undo_restores_modifiers_by_card_identity`（期望 `[1,2,0]`）、differential 全量重跑。
- **Golden**：`py_rules_trace.py` 的 `GENERATOR_VERSION` 升为 `py_rules_trace/2`，golden 文件名带 `g2`，C# 测试校验 header 的 generator 版本，旧 golden 不会被误用。修复后的 Python 重新生成 20,000 序列 / 182,910 步，C# 与之 **0 不一致**；build 0 warning / 0 error，24 个 C# 测试全部通过，分配测试仍为 0 字节。
- **对 Phase 1 baseline 的影响**（只影响含同类规则的场景；`py_bench.py fixed`，修复前后各跑一次）：`mid3_ascension` 各深度的 **根分数（`float.hex`）与最佳走法完全不变**，但 d3–d5 的节点数/cutoff/tt_hits 变化：

  | depth | 修复前 nodes / cutoffs / tt_hits | 修复后 nodes / cutoffs / tt_hits |
  |---|---|---|
  | 1 | 25 / 0 / 0 | 25 / 0 / 0 |
  | 2 | 68 / 23 / 0 | 68 / 23 / 0 |
  | 3 | 416 / 43 / 0 | **403** / 43 / 0 |
  | 4 | 957 / 346 / 28 | **889** / **327** / **39** |
  | 5 | 2,322 / 550 / 73 | **2,404** / **557** / **49** |

  §3.4 表中 `mid3_ascension` 的节点/cutoff/tt_hits 属于修复前的 Python；Phase 3 的 golden 必须用修复后的 Python 重新生成。§3.5 的 `mid3_ascension` 服务端等价调用同理需要在 Phase 3 重测。其余 4 个不含同类规则的场景不经过同类快照路径，未受影响（未重测）。

**已知限制**
- parity 覆盖的是规则结算与 make/undo；评估、排序、搜索未实现（Phase 3）。
- golden 生成依赖本机 CPython 3.11（`random.Random` 对同一 seed 在该版本内确定）。
- 手牌容量 5、实例上限 19：超出的输入被拒绝，而不是静默截断。
- Make/Undo 吞吐在含同类规则的局面（`mid3_ascension`、`open_all_rules`）明显更低（见 §10），原因是每步快照 19 个修正值并全盘重算；只记录，未做优化。

### Phase 3 — Evaluator / MoveOrdering / History / Minimax / AlphaBeta / ID（ReferenceCompat）

范围：`evaluate_state`、`evaluate_move`（含激进模式、`_calculate_placement_edge_score`、`_calculate_endgame_exposure_penalty`、星级表与 1000+/None→0）；稳定排序；history 表；Python 语义的 TT 模拟（key 与 `str(board)`+手牌拼接字符串等价，复现 D-005/D-006）；minimax 的控制流（计时检查点、cutoff、history 更新、TT 写入时机）；ID 循环（固定深度测试时时间无限）；`SearchStats`（nodes、cutoffs、tt_hits、tt_cutoffs）。
验收：
- [x] ≥100,000 个 golden 局面上，`evaluate_state`/`evaluate_move`（普通 + 激进）位级相等。**100,000 个局面**（规则取遍 256 个子集，60,001 个带非空 history，激进模式的三个阶段 ≤2/3..5/≥6 个占用格分别 30,798/46,257/22,945 个局面）：`evaluate_state` 400,000 次、`evaluate_move` 1,592,088 次，全部 `float.hex()` 位级一致，0 个不一致。
- [x] ≥500 个局面 × depth 1..5（残局局面一直搜到终局）：根分数位级相等、best move 相等、nodes_searched、alpha_beta_cutoffs、tt_hits、tt_cutoffs、move_evaluations 相等、最终 history 表相等。**900 个随机局面 + 22 个基准项**，共 4,507 次固定深度搜索（激进模式 1,121 次；搜到终局即 depth ≥ 空格数 2,153 次；各深度 1/2/3/4/5 的运行数 905/905/905/905/887；合计 1,152,695 个节点；其中 560 次出现 TT 命中并全部与 Python 的 tt_hits/tt_cutoffs 一致）：0 个不一致。
- [x] ID（max_depth 固定、时间无限）：每一层的 best move、score、节点数都相等。**200 个局面 + 3 个基准项**（mid3_ascension / mid4_rev_ace / end5_same_plus 搜到 depth 10）共 203 次无限时间 ID；另外 600 次"节点预算"ID（把 `ai.is_time_up` 换成确定性的 `nodes_searched > N`，586 次在 max_depth 前被中断，覆盖"超时后丢弃该层""超时时的部分结果""超时后 TT 仍写入"）和 200 次带预算的直接 minimax；其中有 12 次 ID 运行因预算过小而没有任何可接受的层，最终 best_move 为 None（也一致）。每层 score(位级)/move/nodes，以及总 nodes/cutoffs/tt_hits/tt_cutoffs/move_evaluations/depth_completed 全部一致；ID 中 TT 命中 157,896 次，其中仅 16,582 次是 cutoff（其余是"深度不足的命中"，684 次运行出现），均一致。
- [x] §3.4 的 22 个基准项全部复现（`mid3_ascension` 使用 D-002 修复后的 Python）。
- [x] 热路径零分配：`CompatSearcher` 稳态（6 个场景、普通/激进、固定深度与 ID，共 371,212 个节点）`GC.GetAllocatedBytesForCurrentThread()` 增量 **0 字节**；`Evaluator.EvaluateMove/EvaluateState` 单独测试同样为 0。
- [x] 其他：TT key 的等价类与 Python `get_state_hash` 字符串完全一致（3,000 个家族、38,524 个状态、16,733 个不同的 Python key ↔ 16,733 个不同的 C# 摘要，双射；家族中含故意构造的 D-005 碰撞：`[1,23]` vs `[12,3]`、红/蓝手牌边界上的 `1|23` vs `12|3`、`None` id，以及只改 can_use/类型/手牌修正值/显示边的变体）；搜索结束后状态恢复（除 D-001 的手牌顺序与 D-003 的显示边外逐项相等，4,507+803 次搜索均检查）；墙钟时间限制可用（0.5 s 限制内停止并返回可用走法）。
- [x] `dotnet build -c Release`（`--no-incremental`）0 warning / 0 error；`dotnet test` 33 个测试全部通过（0 失败、0 跳过）；`python -m pytest -q` 29 passed。

#### Phase 3 交付记录（2026-10-07）

**新增文件**（全部为未跟踪新文件；Phase 3 没有修改任何已有的 Python 文件，Core 只增加了 `InstanceCount` 与 `HandSlotOf` 两个只读/查询成员）
- `csharp/src/TTC.Siren.Search/`（新项目，引用 Core）：`Evaluator.cs`（`evaluate_state`/`evaluate_move`/落点边值/残局暴露惩罚）、`HistoryTable.cs`、`CompatTranspositionTable.cs`（含 `PythonStateKey`）、`CompatSearcher.cs`（minimax + ID + `SearchStats` + `SearchLimits`）
- `csharp/tests/TTC.Siren.Tests/`：`SearchDifferentialTests`（评估 / 固定深度 / ID+超时 / TT key 四个 differential）、`SearchBehaviourTests`、`SearchAllocationTests`、`SearchGolden`（golden 读取、`float.hex()` 解析、状态规范化）；`GoldenFiles` 增加按需生成 search golden 的入口
- `tools/parity/py_search_trace.py`（golden 生成器，kind = eval / search / id / tt；只读调用 `ai/ai.py` 与 `core/*`）
- `csharp/src/TTC.Siren.Bench`：新增 `search` 模式（固定深度 ReferenceCompat 搜索基准）

**golden 数据**（git 忽略，按需由测试调用 Python 生成；文件名带生成器版本 `g1`，header 的 generator 字段由测试校验）：
- `search_eval_g1_s20261008_n100000.jsonl.gz`（≈20 MB，≈3 min）、`search_search_g1_s20261008_n900.jsonl.gz`（≈0.2 MB，≈3 min）、`search_id_g1_s20261008_n200.jsonl.gz`、`search_tt_g1_s20261008_n3000.jsonl.gz`。首次运行 `dotnet test` 会自动生成（总计约 8–10 min，四个文件串行）；也可手动：`python tools/parity/py_search_trace.py --kind eval --out <file> --seed 20261008 --count 100000`。

**设计决定**
1. **Move 以卡实例而非手牌槽位为身份（重要）**：`Move.HandSlot` 在生成时有效，但 make+undo 会重排手牌（D-001），同一节点里后面的走法的槽位就会指向别的卡。Python 的走法持有 `Card` 对象所以没有这个问题。`CompatSearcher` 在排序后记录每个走法的卡实例，make 前用新增的 `GameState.HandSlotOf` 重新解析槽位。这个错误由 differential 在 `open_none` depth 2 第一次发现（Python -4.56，C# -1.375），修复后全部一致；回归测试 `Move_order_is_recomputed_by_card_identity_after_hand_reordering`。（**审查整改 H-3 已取代这个"每节点 `instances[]` + `HandSlotOf`"的变通方案**：`Move` 现在直接携带卡实例，见下方 Review remediation。）
2. **TT key 用 128 位摘要，不构造 Python 字符串**：Python 的 key 是 `str(board)`（带 ANSI 色码的多行字符串）+ 手牌 id 无分隔拼接 + 当前玩家。信息等价于：每格（owner、id|None、四条 **base** 边值、type_modifier）+ 手牌 id 的字符序列 + 当前玩家；can_use、卡类型、手牌卡的修正值/owner、陈旧的显示边都 **不在** key 里（因此也不在摘要里）。摘要在碰撞概率 2^-128 内与 Python 字符串等价；等价类用 Python 的 SHA-1 做了双射检查（上面第 6 条）。
3. **TT 无界 → 大固定表 + 满了就抛异常**：Python 的 dict 无界。`CompatTranspositionTable` 默认 2^21 项（≈64 MB，负载上限 75%），超过时抛 `TranspositionTableFullException`，**不会**静默替换（替换会改变 tt_hits，破坏 parity）。`NewGeneration()` 以 O(1) 清空（代际号），所以每次搜索不需要 memset。（**审查整改 H-2 已取代"固定表 + 满了就抛异常"**：改为按需翻倍扩容，仅在硬上限处抛异常，见下方 Review remediation。）
4. **D-006 逐字复现**：无 bound 类型；命中条件 `entry.depth >= depth`；每个节点（包括 cutoff、超时中断之后）都无条件覆盖写入；ID 的各层共享一张 TT（只在开始时清空）。
5. **时钟**：`SearchLimits` = 墙钟秒数（`double.PositiveInfinity` = 无时间限制，不读时钟）+ 测试用的节点预算（`NodesSearched > N` 视为超时，与 Python 侧被替换的 `is_time_up` 完全对应）。固定深度/无限时间模式满足 CLAUDE.md §5 的要求。
6. **稳定排序**：每个走法只算一次 `evaluate_move`（按生成顺序），再做插入排序（`scores[j] < key` 才移动，相等保持原顺序），等价于 `sorted(reverse=True)`。
7. **history 表**：按 card id（不是实例）存 `int[ids × 9]`，`min(old + 2**depth, 1_000_000)`；`history * 0.1` 超过 1000.0 就取 1000.0（history ≥ 10,000 后饱和）。
8. **未实现 / 刻意不做**：Python 的 `path`（PV）和 `SearchResult.stats['branching_factor']`、进度回调、verbose 输出、"预测下一层耗时"块（只打印，从不 break，D-014）。`ai_server` 只取 `best_move`（`move, _ = find_best_move_parallel(...)`），所以 PV 不影响结果；Phase 7 如果需要再补。
9. 星级用 `CardDb` 按 id 取（预测牌 1000+、None、数据库外的 id → 0；474/475 → 0），与 `get_card_star_map().get(card_id, 0)` 一致；`evaluate_move` 的卡力/星级特例/吃子差值读取 **显示边**（D-003 陈旧值），吃子判定读有效边——均已按 Python 复现并由 golden 覆盖（含手牌带过 `apply_type_modifier` 或显示边陈旧的局面）。

**观察 / 新发现**（均无需用户决定，未新增 BLOCKED）
- `tt_hits == tt_cutoffs` 在固定深度搜索里恒成立（单次搜索内同一 key 的条目深度只会 ≥ 当前深度）；"命中但不 cutoff" 只在 ID 的更深层出现。
- 根节点不可能被 TT cutoff（ID 每层的根 key 条目深度都 < 当前深度），所以 TT 里保存的 `move` 只影响内部节点的返回值（父节点不使用）。C# 仍保存 (实例, 格子) 以保持语义。
- 变异检查（每次改一处后 differential 必须失败）：`evaluate_move` 残局惩罚系数 1.8→1.7（第 1 个局面就失败）；排序比较 `<`→`<=`（不稳定）、history 增量 `2**depth`→`2**(depth-1)`、TT 命中条件 `>=`→`>` 全部在 mid3_ascension ID 的第 4–5 层被发现（均已还原）。"超时后跳过 TT 写入" 没有被发现——这是等价变异：超时后搜索立刻整体退出，写入的条目没有任何可观察的读者。
- Python 的 `evaluate_move` 对手牌卡使用 `card.owner`（属性）而不是当前玩家颜色来判断"敌方"；golden 中约 10% 的手牌 owner 是随机值（可能是错误颜色或 None），C# 用实例 owner，一致。

**已知限制**
- 只有 ReferenceCompat；没有 Fast 模式（Phase 4）、没有残局精确求解（Phase 5）。
- `CompatTranspositionTable` 的 128 位摘要与 Python 字符串 key 的等价性是概率性的（2^-128），且仅在 id 位数 ≤ 10 位、每方手牌 ≤ 5 张时成立（编码缓冲区 256 字节）。
- 墙钟模式的结果依赖机器速度，无法 golden 化；只有行为测试（Phase 3 用节点预算替代时钟来验证控制流）。
- 大搜索（例如从空盘开始搜到 depth 9）会超过默认 TT 容量并抛异常；Phase 3 的测试规模（≤ 约 10⁵ 个节点/次）不会触发。**（此条已被审查 H-2 证伪并整改：服务端等价调用会触发，现已改为按需扩容。）**

#### Review remediation (H-1/H-2/H-3)（2026-10-07，来自 `docs/reviews/OPUS_PHASE3_REVIEW.md`；目标 `.claude/sonnet_review_fix_goal.md`）

**状态：整改完成，等待复审。**本节只覆盖三个 HIGH 项；MEDIUM/LOW/OPTIONAL 项（M-1…M-3、L-1…L-9、O-*）按任务范围 **未处理**。没有改动任何 Python 文件（`git diff --stat` 仍然只有 D-002 的 `core/game_state.py`，+13/−22，与整改前相同），没有改动任何规则/评估/排序/TT/ID 语义：所有已有 golden 对比的数字与整改前逐位相同（固定深度 4,507 次/1,152,695 节点、ID 803 次、评估 100,000 局面/1,992,088 次检查、规则轨迹 20,000 序列、TT key 3,000 家族、分配测试的 checksum 18846081 都不变）。

**H-1 异常安全**
- `CompatSearcher.RunFixedDepth` 与 `RunIterativeDeepening` 记录 `rootUndo = _g.UndoDepth`，并在 `finally` 里 `UnwindTo(rootUndo)`（`while (UndoDepth > rootUndo) UndoMove()`）。异常继续向上传播，不吞掉；LIFO 回退与正常递归返回做的事完全一样（含 D-001 的追加顺序），所以异常后的状态等同于"提前返回"。热路径（`Node`）没有改动。
- 测试：`SearchExceptionSafetyTests.Search_restores_state_when_an_exception_escapes`——硬上限 64 的 TT 在 `open_none` 上让 `RunFixedDepth(4)`、`RunIterativeDeepening(6)`、`RunFixedDepth(4, aggressive)` 抛 `TranspositionTableFullException`；断言异常冒出、`UndoDepth == 0`、`NormalizedState` 与搜索前相等；再用 `GameState.FromSnapshot(g.ToSnapshot())` 得到 `g2`，在 `g` 与 `g2` 上各用新 TT 做 `RunFixedDepth(3)`，分数位级、走法、节点数、cutoffs 全部相同。**整改前该测试失败（`UndoDepth left at 3`，与审查测得的一致），整改后通过。**

**H-2 TT 按需扩容**
- `CompatTranspositionTable(initialCapacity = 2^21, maxCapacity = 2^25)`：`Store` 在插入新键会超过 75% 负载时，把槽位数翻倍并重新插入当前代的全部条目（旧代条目视为空），然后继续插入；查找比较完整的 128 位 key，所以重排不改变任何语义。增长只发生在 `Store` 触达负载上限时（罕见、摊销），预分配足够大的表时稳态分配仍为 0。硬上限（默认 2^25 槽 = 1 GiB，`EntryBytes` = 32）被触达时抛 `TranspositionTableFullException`（带 `MaxCapacity`、`Entries`，消息给出上限与条目数）；`finally` 保证状态恢复（H-1）。只读诊断：`Capacity`、`Used`、`PeakUsed`、`Grows`、`MaxCapacity`、`MemoryBytes`。
- 测试：`CompatTtGrowthTests`——`Compat_TT_grows_without_changing_results`（初始容量 `1 << 6`，每次运行一张新表；22 个基准项 + 50 个 golden 局面共 270 次固定深度搜索、190,549 节点，其中 82 次发生扩容、共 233 次扩容；分数位级、走法、nodes、cutoffs、tt_hits、tt_cutoffs、move_evaluations、history 全部与 golden 相同）；`Compat_TT_one_shared_table_keeps_results_across_generations_and_growth`（同一张表跨代/跨扩容复用，结果不变）；`Compat_TT_hard_cap_throws_dedicated_exception_and_state_is_restored`（64→256 的上限：恰好扩容 2 次、`Entries == 192`、消息含上限与条目数、状态恢复）；`Server_equivalent_iterative_deepening_completes`（`RunIterativeDeepening(10, false, WithTime(10))`，4 个局面：无异常、`UndoDepth == 0`、状态恢复、`HasMove`、`Levels == DepthCompleted`、TT ≤ 1 GiB；约 30 s，未跳过）。
- **1 GB 停止条件未触发**：服务端等价调用的 TT 峰值为 512 MiB（≤ 2^24 槽），见 §10。

**H-3 `Move` 的身份是卡实例**
- `Move(Inst, Cell)`（保留 `Row`/`Col`/相等性）。`GenerateMoves` 以 **完全相同的顺序**（手牌当前顺序 × 行优先空格，含秩序/混乱 `can_use` 过滤）产出实例型走法；`MakeMove` 内部解析当前手牌槽位（≤ 5 次比较），在 **改动状态之前** 校验并抛 `InvalidOperationException`：实例不在行动方手牌中、格子已占用或越界、undo 栈满。移除手牌（左移）、undo 追加到末尾（D-001）、D-002 修复、D-003 都没有变。`Evaluator.EvaluateMove` 直接读 `move.Inst`。`CompatSearcher.Node` 里的 `instances[]` + `HandSlotOf` 变通方案已删除，`bestInst` 来自 `move.Inst`。新增 `GameState.MoveFromSlot(slot, cell)`（显式命名，按 **调用时** 的手牌解析槽位）供读 golden 的 `slot*9+cell` 编码的测试使用；golden 文件与生成器不变。`HandSlotOf` 的文档注释已更新。`grep` 确认 `src/` 中没有任何把 `Move` 当作手牌槽位解释的搜索路径代码（仅 `MakeMove` 内部解析和显式的 `MoveFromSlot`/`HandSlotOf`）。
- 测试：`MoveIdentityTests`（5 个）——`Move_identity_survives_hand_reordering`（open_none：先生成走法，对第二个走法 make+undo 使手牌变为 `[56,27,46,7,64]`，再下先生成的第一个走法：棋盘上是原来生成的实例 27，手牌为 `[56,46,7,64]`，符合 D-001；旧的槽位语义会下成 46）；`EvaluateMove_depends_on_the_card_not_on_the_current_hand_order`；`MakeMove_rejects_instance_not_in_hand`（对手手牌实例、棋盘上的实例、已占用格子、格子越界、实例号越界：全部抛异常，且 `ToSnapshot().ToCanonicalString()`（含手牌顺序与显示边）逐字不变）；`MakeMove_rejects_a_full_undo_stack_without_changing_state`；`Generated_moves_keep_hand_order_times_row_major_cells_with_can_use_filter`。`Move_order_is_recomputed_by_card_identity_after_hand_reordering` **未改动**（score bits `0xC0123D70A3D70A3E`、牌 64、格 7、214 节点、42 cutoffs 全部仍然成立）。读取 golden 的测试只改了构造走法的方式（`g.MoveFromSlot(slot, cell)` / `g.HandSlotOf(...)`），断言值没有变。

**验证（整改后）**：`dotnet build -c Release --no-incremental` 0 warning / 0 error；`dotnet test` **43 通过 / 0 失败 / 0 跳过**（原 33 + H-1 1 + H-2 4 + H-3 5）；`python -m pytest -q` 29 passed；所有 Python 生成的 golden 测试使用既有缓存文件通过（未重新生成）；`AllocationTests` 与 `SearchAllocationTests` 仍为 0 字节。

**性能（H-3 前后，同机器、Release、各 3 次独立运行取中位数）**：`Bench search`（43,003 节点，节点数不变）总耗时 0.0483 s → 0.0486 s（+0.6%，在噪声内）；`Bench makeundo` 6 场景中位数均值 9.231 M → 9.243 M pairs/s（+0.1%）。均未超过 10% 的报告阈值。详细数字见 §10。H-1/H-2 不改变热路径（H-1 只在入口加 `try/finally`；H-2 的增长只在 `Store` 触达负载上限时发生）。整改起点（H-1 之前）的基线是：`Bench search` 总耗时中位数 0.0517 s（0.0526/0.0517/0.0459），`Bench makeundo` 中位数 8.19 M（6.71/9.01/8.19）；H-1/H-2 完成后（H-3 之前）为 0.0483 s 与 9.23 M。两组之间的差异是这台机器的运行间噪声（同一代码连续运行就能差 ±10%），不能归因于任何一项整改。

**已知限制 / 备注**
- 服务端等价测试依赖墙钟，各次运行的完成层数会有波动（例如 `open_same_plus` 在两次运行中分别完成 8 和 9 层）；断言只要求行为正确、TT ≤ 1 GiB、≥ 3 层。
- 默认硬上限 2^25 槽（1 GiB）；扩容瞬间旧表与新表同时存在，峰值约为新表大小的 1.5 倍。
- `MoveFromSlot` 与 `HandSlotOf` 是只为 golden 读取/诊断保留的槽位接口，不应出现在搜索路径中；Phase 4 的 TT 走法、history、killer 必须使用 `Move`（实例）。
- 审查门禁仍为 `BLOCK_PHASE_4`，等待复审把它改为 PASS（本整改没有修改门禁状态）。

### Phase 4 — Zobrist / TranspositionTable / Fast 搜索

范围：增量 Zobrist（在 Make/Undo 中更新）；固定大小、2 的幂、bound-aware（exact/lower/upper）、深度优先替换的 TT；Fast 模式的搜索。是否在 Fast 模式中改变并列走法的选择（例如 TT move 优先）需在本阶段开始时由用户确认。
验收：不含同类规则的 golden 局面上，Fast 与关闭 TT 的 ReferenceCompat 根分数位级相等；best move 相等，或为同分走法（逐条列出）；含同类规则的局面：Q-1 已决定修复 D-002，因此同样要求位级相等。给出节点数/NPS 对比。

### Phase 5 — 精确残局 / 信息集 robust 残局 / Python-vs-C# benchmark

范围：`_solve_endgame_exact`（memo key 语义）、`_best_immediate_reply_score`、`_evaluate_endgame_move_robustly`、`select_endgame_robust_move`、`_calculate_corner_safety_risk`、覆盖逻辑；场景构造中确定性的部分（`_build_legal_unknown_candidates`、`_unique_unknown_candidates`、`_score_unknown_candidate` 排序、排列 + 合法性检查 + 截断）。完整 benchmark：§3 的全部项目 + 残局（多个 occupancy、未知牌 0/1/2 张）。
验收：exact 值、场景集合（顺序）、每个走法的 (final, safety, robust, corner) 位级相等；最终选择相等；benchmark 报告。

### Phase 6 — Profiler 驱动优化

范围：用 dotnet-trace / BenchmarkDotNet 找热点，每次改动都走 CLAUDE.md §6 的流程；可能的方向（必须先测量）：增量评估、ordering 计算缓存、TT 大小与布局、残局 α-β（值必须保持精确）。单线程完成后，经用户批准才做 root parallelism。
验收：每次改动都有 Baseline/After 数据，parity 全部保持。

### Phase 7 — 集成策略

建议方案：C# 引擎做成常驻子进程（stdin/stdout JSON 行协议）或 localhost HTTP；Python `ai_server` 仍然负责请求解析和未知牌采样，然后把完整确定的状态交给 C#（minimax + robust 残局），出错或超时时回退到 Python 求解器（受开关控制）。
验收：录制的真实 `/ai_move` 请求（固定 Python 随机种子）端到端结果一致；延迟报告；回退路径有测试。

### Phase 8 — 全量服务移植

范围：ASP.NET Core minimal API 替代 Flask（`/ai_move`、`/search_progress`、`/api/recommend`、`/api/draft/analyze`、`/api/winrate`）；请求解析（D-011）；UnknownCardHandler 与 MC（分布等价 + C# 内部可用种子复现）；文案输出；控制台 UI；单文件发布。
验收：协议兼容（相同 JSON 字段）；对确定性部分做 parity；随机部分做统计检验；由用户确认后才能替换 Python。

### Phase 9 — 可选 CP-SAT / scenario

范围：用 OR-Tools CP-SAT 枚举/抽样合法隐藏牌组合（星级配额、选拔、NPC 卡池），改进 deck optimization 与 scenario generation；只为搜索提供 "可能世界"，不取代 Minimax。

---

## 8. 重要设计决定（ADR）

| # | 决定 | 原因 | 状态 |
|---|---|---|---|
| A-1 | Python 为 reference；已知怪异行为在用户决定前由 C# 复现 | 只有这样 parity 才可验证 | 已定 |
| A-2 | 两种搜索配置：ReferenceCompat（节点级 parity）+ Fast | Python 的 TT 有缺陷（D-005/D-006），Fast 不应继承；但验证必须以 compat 为基准 | 已定 |
| A-3 | 卡实例用小整数下标 + SoA 数组，GameState 为单个可变实例 | 零分配 make/undo | 已定 |
| A-4 | 浮点使用 double，保持 Python 运算顺序，parity 按位级比较 | 结果必须完全一致 | 已定 |
| A-5 | 自写稳定排序 | `Array.Sort` 不稳定 | 已定 |
| A-6 | 目标框架 net10.0 | 本机已安装 SDK 10.0.401（LTS） | 已定 |
| A-7 | 随机采样类逻辑不追求位级 parity | Python RNG 与 `random.sample` 的实现细节不应复刻 | 已定 |
| A-8 | 集成先用子进程/本地 HTTP 桥接，再做全量替换 | 降低风险，可回退 | 提议（Phase 7 确认） |

---

## 9. BLOCKED / 待用户决定

当前 **没有阻塞 Phase 2 的项目**。以下问题需要用户在对应阶段之前做出决定：

| ID | 问题 | 默认做法（未回复时） | 必须在何时之前决定 |
|---|---|---|---|
| Q-1 | ~~D-002 是否同时修复 Python 和 C#？~~ **已决定：修复（用户，2026-10-07），已完成，D-002 = FIXED-BOTH。** | — | 已关闭 |
| Q-2 | D-004（手牌内 card_id 重复导致状态损坏）：在 Python 侧修复（按卡身份移除），还是只在输入层拒绝/报告？ | C# 按卡身份处理，遇到重复输入抛异常，parity 中排除 | Phase 7 前 |
| Q-3 | D-013 连携只有一层：确认这是有意设计吗？ | 保持 Python 行为 | Phase 8 前（不影响 parity） |
| Q-4 | D-012 卡库 474/475 全 0 空行是否应从候选池中排除？ | 保持 | Phase 5 前 |
| Q-5 | Fast 模式中，同分并列时是否允许选择与 Python 不同的走法？ | 不允许，除非逐条报告 | Phase 4 开始前 |
| Q-6 | 是否需要 PyPy 的 baseline（生产构建使用 PyPy，本机未安装）？ | 只用 CPython baseline | Phase 5 前 |

---

## 10. Benchmark 历史

| 日期 | Phase | 实现 | 项目 | 结果 | 备注 |
|---|---|---|---|---|---|
| 2026-10-07 | 1 | Python (CPython 3.11.4) | 固定深度套件（§3.4） | open d4 15,807 节点 / 1.089 s（14.5k NPS）；mid3_ascension d5 2,322 / 0.221 s | 3 次中位数，结果确定性已验证 |
| 2026-10-07 | 1 | Python | 服务端等价调用 10 s（§3.5） | 开局完成深度 5/9，约 16–17k NPS | 单次 |
| 2026-10-07 | 1 | Python | `_solve_endgame_exact`（§3.6） | 48k–84k make/s | 单次 |
| 2026-10-07 | 2 | C# net10.0 Release（Core，Make+Undo 随机 playout） | open_none | 7.9–8.5 M pairs/s | 每次 2,000,000 对、5 次取中位数；两次独立进程运行的范围；alloc 0 B |
| 2026-10-07 | 2 | C# | open_same_plus | 6.2–6.7 M pairs/s | 同上 |
| 2026-10-07 | 2 | C# | mid3_ascension（同类强化） | 1.79–1.85 M pairs/s | 同上；快照 + 全盘重算 |
| 2026-10-07 | 2 | C# | mid4_rev_ace | 11.0–11.7 M pairs/s | 同上 |
| 2026-10-07 | 2 | C# | end5_same_plus | 8.4–8.5 M pairs/s | 同上 |
| 2026-10-07 | 2 | C# | open_all_rules（同数+加算+逆转+王牌杀手+同类强化） | 1.25–1.26 M pairs/s | 同上 |
| 2026-10-07 | 2 | C# | 6 场景中位数均值 | 6.1–6.4 M pairs/s | 机器见 §3.1；Phase 3 起以此为"无搜索开销的上限"参考 |
| 2026-10-07 | 2 | C#（D-002 修复后，快照改为按实例） | 6 场景 | 单次运行（2,000,000 对 × 5 次中位数，alloc 0 B）：open_none 8.44 M；open_same_plus 6.96 M；mid3_ascension 3.04 M；mid4_rev_ace 17.4 M；end5_same_plus 12.6 M；open_all_rules 2.12 M；均值 8.43 M pairs/s | 仅此一次修复后测量；不含同类规则的场景（如 mid4_rev_ace 11→17 M）也变快，说明两次运行之间机器负载不同，**不能**把同类场景的提升归因于修复。需要结论时须同负载下 A/B 测量（§6） |
| 2026-10-07 | 3 | Python（CPython 3.11.4，D-002 修复后） | §3.4 的 22 个固定深度项合计 | 43,003 节点 / 2.965 s（中位数，3 次：2.96/2.96/3.08）= 14.5 k NPS | `py_bench.py fixed` 逐项耗时之和 |
| 2026-10-07 | 3 | C# net10.0 Release（ReferenceCompat，`Bench search`） | §3.4 的 22 个固定深度项合计 | 43,003 节点 / 0.045 s（两次独立运行：0.0451 / 0.0448）= 954k / 960k NPS；alloc 0 B | 每项 5 次取中位数（另加 1 次预热）；节点数、分数位级、走法与 Python 完全相同；**≈ 66× 总耗时**（微小项受计时精度影响） |
| 2026-10-07 | 3 | C# ReferenceCompat | open_none d4（15,807 节点） | 0.0231 s = 685k NPS（Python 1.089 s = 14.5k NPS，≈ 47×） | 同上，单项 |
| 2026-10-07 | 3 | C# ReferenceCompat | open_same_plus d4（12,665 节点） | 0.0084 s = 1.51 M NPS（Python 0.933 s，≈ 111×） | 同上；TT 命中使 C# 的计时受缓存影响更大，只作参考 |
| 2026-10-07 | 3 | C# ReferenceCompat | mid3_ascension d5（2,404 节点，D-002 修复后） | 0.0016 s = 1.52 M NPS | 同上；Python 修复后 d5 的耗时未单独测量 |
| 2026-10-07 | 3 整改 H-2 | C# ReferenceCompat，服务端等价调用 `RunIterativeDeepening(10, false, WithTime(10))`（默认 TT：初始 2^21 槽、硬上限 2^25；每个局面一张新表；单次运行，墙钟相关） | open_none | 完成 10 层；15,944,848 节点；TT 条目 7,279,403；扩容 3 次；容量 16,777,216 槽 ≈ 512 MiB；8.86 s | 整改前：2.7 s 后抛 `TranspositionTableFullException`（第 8 层、4.2 M 节点，审查 V5） |
| 2026-10-07 | 3 整改 H-2 | 同上 | open_same_plus | 完成 9 层；16,746,862 节点；TT 条目 8,246,136；扩容 3 次；16,777,216 槽 ≈ 512 MiB；10.00 s | 另一次运行（测试中）完成 8 层、10,783,958 节点：层数随机器负载波动 |
| 2026-10-07 | 3 整改 H-2 | 同上 | mid3_ascension | 完成 10 层；30,500 节点；TT 条目 8,665；扩容 0 次；2,097,152 槽 = 64 MiB；0.02 s | 与 D-002 修复后的 ID 节点数一致 |
| 2026-10-07 | 3 整改 H-2 | 同上 | open_all_rules | 完成 7 层；10,647,148 节点；TT 条目 3,803,965；扩容 2 次；8,388,608 槽 ≈ 256 MiB；10.00 s | 整改前：5.2 s 后抛异常（第 8 层、5.0 M 节点）。最大 TT 峰值为 512 MiB，低于 1 GiB 的停止线 |
| 2026-10-07 | 3 整改 H-3 | C# ReferenceCompat，`Bench search`（22 项合计，43,003 节点，节点数前后不变） | H-3 之前 / 之后，各 3 次独立运行 | 之前 0.0491 / 0.0483 / 0.0464 s → 中位数 **0.0483 s**（≈ 889 k NPS）；之后 0.0521 / 0.0457 / 0.0486 s → 中位数 **0.0486 s**（≈ 885 k NPS） | +0.6%，低于 10% 阈值；每次运行内部是 5 次取中位数；机器运行间噪声约 ±10% |
| 2026-10-07 | 3 整改 H-3 | C# Core，`Bench makeundo`（6 场景，每次 2,000,000 对 × 5 次中位数；alloc 0 B） | H-3 之前 / 之后，各 3 次独立运行（6 场景中位数均值） | 之前 9.23 / 9.24 / 9.02 M → 中位数 **9.231 M pairs/s**；之后 9.11 / 9.48 / 9.24 M → 中位数 **9.243 M pairs/s** | +0.1%，低于 10% 阈值（`MakeMove` 多了一次 ≤ 5 次比较的槽位解析） |
| 2026-10-07 | 2 | Python（参考） | `_solve_endgame_exact` 复测 | end5: 217 make / 0.004 s；mid4_rev_ace: 4,707 make / 0.055 s（≈85k make/s） | 与 Phase 1 一致；Python 的 make/s 含 minimax 与 memo 开销，不是纯 make/undo，不可直接比 |

---

## 11. 下一步行动

0. **审查门禁 `PASS_TO_PHASE_4`**（复审 2026-10-07，`docs/reviews/OPUS_PHASE3_REVIEW.md` §15；初审结果为 `BLOCK_PHASE_4`）：H-1/H-2/H-3 已 RESOLVED，没有回归。下一步：用户决定 Q-5（建议见 §15.6）并明确授权 Phase 4。MEDIUM/LOW 项（M-1 到 M-3、L-1 到 L-9）未处理，不阻塞，建议在 Phase 4 前处理 M-1 到 M-3。
1. Phase 2、Phase 3 已完成并验收（§7）。**Phase 4 尚未开始，等待用户明确授权。**
2. 工作区状态：Phase 2/3 的产物都是未跟踪的新文件，外加对 `core/game_state.py`（D-002 修复）的一处修改和新增的 `tests/test_undo_type_modifiers.py`；未 commit、未 push。是否 commit（建议拆成：文档 / D-002 的 Python 修复与回归测试 / Python parity 与 bench 工具 / C# Core / C# Search / C# 测试与 Bench）由用户决定。
3. Phase 4 开始前需要用户决定：**Q-5**（Fast 模式并列走法能否与 Python 不同；默认不允许，除非逐条报告）。Q-1 已关闭（D-002 FIXED-BOTH）。
4. Phase 4 的第一步建议：以 Phase 3 的 `CompatSearcher` 为基准，在 `TTC.Siren.Search` 中新增 Fast 模式（增量 Zobrist + bound-aware TT）；用 §3.4 的 22 项和 `search` golden 中不含 TT 差异的局面做对照（Fast 对 "关闭 TT 的 ReferenceCompat"）。注意：ReferenceCompat 目前没有"关闭 TT"开关，需要先加（只影响 Phase 4 的对照，不改变默认行为）。

---

## 附录 A — Phase 1 benchmark 脚本（Phase 2 时作为 `tools/bench/py_bench.py` 提交）

```python
# Usage: python py_bench.py <repo_root> [fixed|exact|server|all]
import sys, os, time, json
ROOT = sys.argv[1]; sys.path.insert(0, ROOT); os.chdir(ROOT)
import ai.ai as A
from ai_server import get_all_cards, _solve_endgame_exact
from core.board import Board
from core.player import Player
from core.game_state import GameState

DB = {c.card_id: c for c in get_all_cards()}
def card(cid, owner):
    c = DB[cid].copy(); c.owner = owner; return c
def state(board_items, red_ids, blue_ids, cur, rules):
    b = Board()
    for (r, c), cid, owner in board_items: b.place_card(r, c, card(cid, owner))
    players = [Player('red', [card(i, 'red') for i in red_ids]),
               Player('blue', [card(i, 'blue') for i in blue_ids])]
    s = GameState(b, players, current_player_idx=cur, rules=list(rules))
    if '同类强化' in rules or '同类弱化' in rules: s.recalculate_type_modifiers()
    return s

RED5 = [56, 64, 27, 46, 7]; BLUE5 = [59, 51, 35, 48, 8]
SCEN = {
  'open_none':      lambda: state([], RED5, BLUE5, 0, []),
  'open_same_plus': lambda: state([], RED5, BLUE5, 0, ['同数', '加算']),
  'mid3_ascension': lambda: state([((1,1),40,'red'), ((0,0),41,'blue'), ((2,2),52,'red')],
                                  [42,43,31,15], [53,54,47,14], 1, ['同类强化']),
  'mid4_rev_ace':   lambda: state([((0,1),1,'red'), ((1,0),2,'blue'), ((1,2),3,'red'), ((2,1),4,'blue')],
                                  [56,64,27], [59,51,35], 0, ['逆转', '王牌杀手']),
  'end5_same_plus': lambda: state([((0,0),56,'red'), ((0,1),59,'blue'), ((1,1),64,'red'),
                                   ((2,0),51,'blue'), ((2,2),27,'red')], [46,7], [35,48], 1, ['同数', '加算']),
}
def fmt_move(m):
    if m is None: return None
    c, (r, col) = m; return [c.card_id, r, col]
def run_fixed_depth(name, depth):
    s = SCEN[name](); A.TRANSPOSITION_TABLE = {}; A.SEARCH_STATS.reset()
    A.START_TIME = time.time(); A.TIME_LIMIT = 1e9
    t0 = time.perf_counter()
    res = A.minimax(s, depth, float('-inf'), float('inf'), True, s.current_player_idx, False, True, {})
    dt = time.perf_counter() - t0; n = A.SEARCH_STATS.nodes_searched
    return dict(scenario=name, depth=depth, nodes=n, sec=dt, score_hex=float(res.eval_score).hex(),
                move=fmt_move(res.best_move), cutoffs=A.SEARCH_STATS.alpha_beta_cutoffs, tt_hits=A.SEARCH_STATS.tt_hits)
def run_server_like(name, max_time=10, max_depth=10):
    s = SCEN[name](); t0 = time.perf_counter()
    mv, _ = A.find_best_move_parallel(s, max_depth=max_depth, max_time=max_time)
    return dict(scenario=name, depth_completed=A.SEARCH_STATS.depth_completed, nodes=A.SEARCH_STATS.nodes_searched,
                sec=time.perf_counter() - t0, move=fmt_move(mv), nodes_per_depth=A.SEARCH_STATS.nodes_per_depth)
def run_exact(name):
    s = SCEN[name](); n = [0]; orig = s.make_move
    def counted(*a, **k): n[0] += 1; return orig(*a, **k)
    s.make_move = counted; cache = {}; t0 = time.perf_counter()
    v = _solve_endgame_exact(s, s.current_player_idx, cache)
    return dict(scenario=name, make_moves=n[0], cache=len(cache), sec=time.perf_counter() - t0, score=v)

if __name__ == '__main__':
    which = sys.argv[2] if len(sys.argv) > 2 else 'all'
    if which in ('all', 'fixed'):
        for name, depths in [('open_none',[1,2,3,4]), ('open_same_plus',[1,2,3,4]), ('mid3_ascension',[1,2,3,4,5]),
                             ('mid4_rev_ace',[1,2,3,4,5]), ('end5_same_plus',[1,2,3,4])]:
            for d in depths: print(json.dumps(run_fixed_depth(name, d), ensure_ascii=False), flush=True)
    if which in ('all', 'exact'):
        for name in ['end5_same_plus', 'mid4_rev_ace']: print(json.dumps(run_exact(name)), flush=True)
    if which in ('all', 'server'):
        for name in ['open_none', 'open_same_plus', 'mid3_ascension']:
            print(json.dumps(run_server_like(name), ensure_ascii=False), flush=True)
```
（Phase 1 实际运行的版本输出的是 6 位小数的 score，而不是 `score_hex`，其余逻辑相同。）
