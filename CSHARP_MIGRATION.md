# CSHARP_MIGRATION.md — TTC_Siren Python → C# 迁移进度（唯一事实来源）

> 每个 Claude 会话：先读 `CLAUDE.md`（长期规则），再读本文件。
> 本文件记录：Python 架构、性能热点、baseline、C# 目标架构、Phase 定义与状态、设计决定、语义差异、benchmark 历史、BLOCKED、下一步。
> 任何 Phase 完成后必须更新本文件。
>
> **新会话快速上手：先读 `CLAUDE.md`，再读本文件 §12「会话交接」（当前状态、如何构建/测试、用户已做的决定、未解决问题、下一步），然后按需读 §0 与 §7。**
> 当前一句话状态：Phase 1–4 完成（Phase 4 于 2026-10-07 完成，Q-5 = 选项 (b)，验收见 §7 Phase 4）；Phase 5 **未开始**；Phase 4 独立审查门禁为 **`PASS_TO_PHASE_5`**（H4-1 已 RESOLVED，复审见 `docs/reviews/OPUS_PHASE4_REVIEW.md` §13），等待用户明确授权；工作在分支 `csharp-solver-migration`（Phase 1–3 已推送；Phase 4 的改动**尚未提交、未推送**；**没有**对 `main` 的 PR——PR #1 已按用户要求关闭）。

---

## 0. 状态总览

| Phase | 名称 | 状态 |
|---|---|---|
| 1 | Python 架构分析 / profiling / baseline | **DONE**（2026-10-07） |
| 2 | Card / Move / RuleSet / GameState / MakeMove / UndoMove + 规则 parity 工具 | **DONE**（2026-10-07，验收见 §7 Phase 2） |
| 3 | Evaluator / MoveOrdering / History / Minimax / AlphaBeta / ID（ReferenceCompat，节点级 parity） | **DONE**（2026-10-07，验收见 §7 Phase 3） |
| 4 | Zobrist / 带 bound 的 TranspositionTable / Fast 搜索模式 | **DONE**（2026-10-07，Q-5 = 选项 (b)，验收见 §7 Phase 4） |
| 5 | 精确残局 + 信息集 robust 残局 + Python-vs-C# 全量 benchmark | NOT STARTED — Phase 4 审查门禁 **`PASS_TO_PHASE_5`**（复审 2026-10-07，`docs/reviews/OPUS_PHASE4_REVIEW.md` §13；技术上可以开始，仍需用户明确授权） |
| 6 | Profiler 驱动优化（单线程）→ 可选 root parallelism | NOT STARTED |
| 7 | 集成策略：Python 服务调用 C# 引擎（桥接 + 回退） | NOT STARTED |
| 8 | 全量服务移植（请求解析、未知牌采样、MC、分析文案、app.py 子系统、HTTP） | NOT STARTED |
| 9 | 可选：CP-SAT 隐藏牌组合 / deck optimization / scenario generation | NOT STARTED |

**Phase 4 审查门禁（2026-10-07）**：结果 **`BLOCK_PHASE_5`**，报告 `docs/reviews/OPUS_PHASE4_REVIEW.md`。Zobrist（覆盖范围与增量更新）、bound 语义（按原始窗口分类、fail-soft）、同深度复用、被中断的搜索不写表、TT 走法校验、history 都经独立验证为正确：在新种子 golden 上 Phase 4 的 15 个测试全部通过；对抗性用例 1,756 次运行（含 4 项的极小表）与 oracle 0 不一致。唯一的阻塞项是 **H4-1**：TT 表项不带有效上下文（GameState、AI 方、评估模式），公开参数 `clearTable:false` 在换边、切换 aggressive、跨 GameState 共表时会 **静默返回错误分数**（审查探针分别为 6/6、6/6、3/32 错误）。修复任务：`.claude/sonnet_review_fix_goal.md`。复审改为 PASS 之前不得开始 Phase 5。
**整改进展（2026-10-07）**：H4-1 **已修复，等待复审**（记录见 §7 Phase 4 的 "Review remediation (H4-1)"；测量见 §10）：`FastTranspositionTable` 现在记录当前代的 `SearchContext`（GameState 对象、AI 方、`EvalMode`），`clearTable:false` 在上下文不同或表未绑定时在 `Begin` 里抛 `InvalidOperationException`（不改变任何东西）；62/62 测试通过。门禁状态 **仍为 `BLOCK_PHASE_5`**，由复审者而不是整改者改为 PASS；Phase 5 仍为 NOT STARTED。M4-1 到 M4-3、L4-1 到 L4-5、O4-1 到 O4-3 按任务范围未处理。
**复审结果（2026-10-07，`docs/reviews/OPUS_PHASE4_REVIEW.md` §13）**：H4-1 经独立验证为 **RESOLVED**，门禁改为 **`PASS_TO_PHASE_5`**（这是现行状态，上面两行是记录）。验证内容：62/62 测试通过，pytest 29，0 warning，没有改动 Python；审查者原来的 H4-1 探针（换边、切换 aggressive、跨 GameState、第二个 searcher、ID、逻辑克隆）51/51 被拒绝，0 个静默错误，拒绝后 状态/哈希/UndoDepth/表 均不变；变异检查：去掉 `RequireContext` → 3 个拒绝测试失败，单独去掉 AI 方 / EvalMode / GameState 的比较 → 分别只有对应的测试失败；热路径（`Node`/`TryGet`/`Store`）没有改动，分配为 0。剩余 CRITICAL 0、HIGH 0；新增 LOW：L4-6（`TryGet`/`Store` 仍是公开且不带上下文的接口，Phase 5 若新增求解器直接使用表，必须用自己的 `EvalMode` 绑定）、L4-7（复用测试的"命中"指标偏高，不影响正确性）。Phase 5 仍需用户明确授权；建议先处理 M-1/M-2（不阻塞）。

> 以下三段是按时间顺序追加的门禁历史：初审 `BLOCK_PHASE_4` → 整改完成 → 复审 `PASS_TO_PHASE_4`。**现行状态是第三段（PASS_TO_PHASE_4）**；前两段保留作为记录，其中"不得开始 Phase 4"的限制已随复审解除。

**审查门禁（Phase 2+3 独立审查，2026-10-07）**：结果 **`BLOCK_PHASE_4`**。报告：`docs/reviews/OPUS_PHASE3_REVIEW.md`。阻塞项为 3 个 HIGH：H-1 搜索异常时 GameState 未恢复；H-2 ReferenceCompat TT 在服务端等价配置下溢出；H-3 `Move` 以手牌槽位为身份。规则、评估、搜索与 Python 的 parity 已被独立复核（新种子重新生成全部 golden，外加对抗性用例，0 不一致）。修复任务：`.claude/sonnet_review_fix_goal.md`。修复完成并经复审改为 PASS 之前，**不得开始 Phase 4**（即使 `.claude/phase_goal.md` 按 §0 选中了 Phase 4）。Phase 4 开始前仍需用户决定 Q-5。
**整改进展（2026-10-07）**：H-1、H-2、H-3 的整改已完成，**等待复审**（记录见 §7 Phase 3 的 "Review remediation (H-1/H-2/H-3)"；测量见 §10）。门禁状态 **仍为 `BLOCK_PHASE_4`**，由复审者而不是整改者改为 PASS；Phase 4 仍为 NOT STARTED。
**复审结果（2026-10-07，`docs/reviews/OPUS_PHASE3_REVIEW.md` §15）**：门禁改为 **`PASS_TO_PHASE_4`**。H-1、H-2、H-3 均经独立验证为 **RESOLVED**，没有引入回归：43/43 测试通过；在原 golden 与新种子 golden 上 parity 计数完全不变；对抗性用例 0 不一致；"过期走法" fuzz 115,955 次 0 错误；中途抛异常（含调用方已有走法、ID 第 2 层以后抛出）时状态都能恢复；服务端等价调用 4 个局面全部完成，TT 峰值 512 MiB。M-1 到 M-3、L-1 到 L-12、O-1 到 O-6 仍未处理，均不阻塞（建议在 Phase 4 依赖新 golden 之前处理 M-1 到 M-3）。Phase 4 仍需用户明确授权；Q-5 的建议见审查报告 §15.6（选项 b：只允许在分数完全相等的走法之间不同，并逐条验证、报告）。

分析基线：分支 `csharp-solver-migration`，commit `eb60a02`（也是 `main` 的当前 HEAD）。Phase 1 未修改任何 Python 代码，只新增了 `CLAUDE.md` 和本文件。
Phase 2 首次交付时没有修改任何已跟踪文件；**之后经用户批准（Q-1）修改了唯一一个已跟踪的 Python 文件 `core/game_state.py`（D-002 修复，+13/−22）并新增 `tests/test_undo_type_modifiers.py`**。Phase 1–3 的全部产物现已提交为 5 个提交并推送到 `origin/csharp-solver-migration`（见 §12.1）。

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
| D-005 | TT key 中手牌 id 无分隔符拼接：`[1,23]` 与 `[12,3]` 的 key 相同（实测碰撞） | 罕见的错误 TT 命中 | ReferenceCompat 模式复现；**Fast 模式已改用 Zobrist（Phase 4，无此问题）** | OPEN-EMULATE（compat；Fast 不继承） |
| D-006 | TT 不存 bound 类型；cutoff/超时后的部分结果也按精确值存储，命中条件只看深度 | 理论上不正确。§3.7 的 4 个样本中未观察到影响；**Phase 4 在 4,507 次固定深度运行中实测：3 次（0.07%）Python（TT 开）的根分数 ≠ 精确 minimax 值，且 Python 的走法不在精确并列集合内**（`i=186 d5`、`i=383 d5 激进`、`i=397 d4`，见 §7 Phase 4 的 Q-5 报告；ReferenceCompat 与 Python 一致，所以这三次是 TT 语义造成的，D-005 碰撞与 D-006 无 bound 在此未分开归因） | ReferenceCompat 复现；**Fast 模式使用 exact/lower/upper，只在同深度复用，不存被中断节点（Phase 4 已实现并验证）** | OPEN-EMULATE（compat；Fast 不继承） |
| D-007 | 终局（棋盘满）仍用启发式 `evaluate_state`，没有胜负大分 | 搜索可能偏好"启发式分高但输掉"的线 | 不改，复现 | OPEN-EMULATE |
| D-008 | `ai.ai` 的 TT/统计/计时都是模块级全局，Flask 多线程并发请求会互相干扰 | 并发时结果错误 | C# 每次搜索独立上下文（行为上等价于单请求） | OPEN（C# 天然不复现，记录即可） |
| D-009 | `ai.get_card_star_map()` 用相对 cwd 的路径 | 从别的目录启动时会崩溃 | C# 用程序目录定位数据 | OPEN（集成时处理） |
| D-010 | `count_captured_cards` 用复制品 make/undo → 手牌中的原卡被替换为复制品，并改变顺序 | 只影响文案和输出预测棋盘 | Phase 8 移植文案时复现或修复（届时决定） | OPEN |
| D-011 | 请求解析中 `numL→right`、`numR→left` 左右互换（`ai_server.py:161, :809`） | 协议语义，必须保持 | Phase 7/8 必须复现 | OPEN-EMULATE |
| D-012 | 数据库 id 474/475 为全 0 空行，会进入候选池（有效值视为 1，显示值为 0） | 未知牌候选质量 | 复现；是否过滤由用户决定 | OPEN |
| D-013 | 连携只有一层（连携翻得的卡不再继续连携），与常见的游戏实现可能不同 | 规则语义 | Python 为准，复现；若要改动需用户确认 | OPEN-EMULATE |
| D-014 | ID 在深度超过剩余空格后仍重复搜索；"预测下一层耗时" 的代码从不生效 | 只浪费时间，固定深度结果不受影响 | ReferenceCompat 复现；**Fast 模式已在整棵博弈树搜完（depth ≥ 空格数）或根方无合法走法后停止（Phase 4；每一层的分数与精确值位级相等，已验证）** | OPEN-EMULATE（compat；Fast 已处理） |

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
| Undo 记录 | 预分配 `UndoRecord[]` 栈（struct）：cell、placedInst、player、placedOrigOwner、`ushort FlipMask` + `ushort FlipFromNoneMask`、`bool HasSnapshot`；修正值快照在独立的 `sbyte[depth × MaxInstances]` 数组里，**按卡实例**保存（D-002 修复后；最初设计是"按 Python 的 key 语义"，已被取代） | 翻转的原 owner 只可能是"另一方"或 `None`，所以用两个 9 位 mask（见 Phase 2 设计决定 2） |
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

范围：增量 Zobrist（在 Make/Undo 中更新）；固定大小、2 的幂、bound-aware（exact/lower/upper）、深度优先替换的 TT；Fast 模式的搜索。**Q-5 已决定：选项 (b)**——Fast 的根走法只允许在 **分数位级完全相等** 的走法之间与 Python/compat 不同，并且每一处差异都要验证、报告（用户 2026-10-07）。
验收（原文）：不含同类规则的 golden 局面上，Fast 与关闭 TT 的 ReferenceCompat 根分数位级相等；best move 相等，或为同分走法（逐条列出）；含同类规则的局面：Q-1 已决定修复 D-002，因此同样要求位级相等。给出节点数/NPS 对比。

**验收结果（全部满足，2026-10-07）**
- [x] `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`：0 warning / 0 error。
- [x] `dotnet test`：Phase 4 完成时 **58 通过 / 0 失败 / 0 跳过**（Phase 3 整改后的 43 + Phase 4 新增 15；H4-1 整改后为 62，见 "Review remediation (H4-1)"）；`python -m pytest -q`：29 passed；没有任何 Python 文件被修改（`git status` 无 `.py`）。
- [x] **Fast（固定深度、无时间限制）对 oracle（ReferenceCompat，TT 关闭）**：4,507 次运行（900 个随机局面 + §3.4 的 22 个基准项，depth 1..5；激进模式 1,121 次；含同类强化/弱化规则 3,354 次；搜到终局 2,153 次；根方无合法走法 85 次），**根分数全部位级相等**；根走法全部属于精确并列集合；用 16 项的极小表重复（强替换压力）结果相同；两次运行完全相同（确定性）。含同类规则的局面同样位级相等。另有 6 个基准场景的更深搜索（open_none d≤5、open_same_plus d≤5、mid3_ascension d≤8、mid4_rev_ace d≤8、end5_same_plus d≤6、open_all_rules d≤4）同样位级相等。
- [x] oracle 本身用独立实现验证：无 α-β、无排序、无 TT 的全宽 minimax 在 4,470 次运行（2,153 次搜到终局；6,709,723 个叶子；另有 37 次因树过大跳过）上与 oracle 根分数位级相等。
- [x] **Q-5 并列集合验证**：对每次运行，用 compat（TT 关闭）以全窗口计算 **每个根走法** 的精确值，并列集合 = 分数与根分数位级相等的走法；Fast 的根走法必须在其中，否则测试失败（视为 BLOCKED，不是并列）。188 次运行存在 ≥2 个并列最优走法（并列规模直方图 0:85、1:4,234、2:175、3:10、4:3；0 表示根方无合法走法）。
  - **固定深度：Fast 的走法与 oracle 的走法完全相同（0 处差异，16 项小表也是 0 处差异）。** 原因：固定深度的根排序在 history 为空且无 TT 走法时与 compat 相同，并列时取排序中第一个。
  - **与 Python（TT 开）的 golden 走法不同的有 3 次，三次都是 D-006 的实例**（Python 的 TT 结果 ≠ 精确值，走法不在精确并列集合内；Fast 与 oracle 在这三次上一致）。清单：`i=186 d5`（Fast/oracle 143@(1,2)，Python 474@(2,2)）、`i=383 d5 激进`（5003@(1,2) vs 5004@(1,2)）、`i=397 d4`（5007@(1,2) vs 5007@(0,2)），三次的精确并列集合大小都是 1。
  - **迭代加深：** 在 203 次无限时间的 ID 运行、701 个层上，每一层的分数与 oracle 的同深度分数位级相等；29 个层存在并列；**5 个层的走法与 oracle 的固定深度走法不同，全部是精确并列集合的成员**（下一层的根排序先试上一层的最佳走法，因此并列时可能选另一个）。完整清单（label | 层 | 激进 | 分数 | 并列规模 | oracle 走法 | Fast 走法；走法格式 卡id,行,列）：
    - `end5_same_plus` | 4 | 否 | 2.76 (`0x4006147AE147AE15`) | 2 | 48,1,0 | 35,2,1（Phase 1 的 Python 基线 d4 走法是 [48,1,0]，同分的 [35,2,1] 与之并列）
    - `i=0` | 5 | 否 | −7.86 (`0xC01F70A3D70A3D71`) | 2 | 362,1,0 | 87,1,0
    - `i=58` | 4 | 否 | −5.91 (`0xC017A3D70A3D70A4`) | 2 | 1270,0,0 | 5002,0,2
    - `i=58` | 5 | 否 | −5.91 (`0xC017A3D70A3D70A4`) | 2 | 1270,0,0 | 5002,0,2
    - `i=103` | 4 | 否 | 12.165 (`0x4028547AE147AE14`) | 2 | 5003,0,1 | 376,1,2
  - 两份报告由测试在每次运行时重新生成（git 忽略）：`tools/parity/golden/phase4_q5_tie_report.txt`（固定深度）、`phase4_q5_id_tie_report.txt`（ID）。
- [x] Zobrist：Python golden 的 20,000 条序列（110,937 次 make、71,973 次 undo，共 202,910 次检查）每一步增量哈希 == 从头计算，undo 精确恢复 make 前的哈希；每条序列内 `位置 ⇔ 哈希` 是双射（115,425 个不同位置，0 个碰撞、0 个同位置不同哈希）。手牌重排（D-001）、显示边（D-003）不影响哈希；同一局面的不同走子顺序哈希相同；同类强化下修正值的变化被哈希捕获。
- [x] 被中断的搜索（节点预算，600 次 ID 运行，其中 206 次在最后一层被中断）不留下可被复用的条目：在同一张表上继续搜被中断的那一层（`clearTable:false`），根分数与 oracle 位级相等（183 次继续搜索实际命中了表）。
- [x] 热路径零分配：Fast 在 6 个场景（普通/激进、固定深度、ID、节点预算、墙钟限制）共 2,094,824 个节点上 `GC.GetAllocatedBytesForCurrentThread()` 增量 **0 字节**；含 Zobrist 的 Make/Undo（500,000 对）同样 0 字节。
- [x] 墙钟限制：0.3 s 限制内停止并返回走法（完成 6 层 / 360,448 节点），状态与哈希恢复；非法深度（<0 或 >100）抛 `ArgumentOutOfRangeException` 且不动状态。
- [x] 节点数 / NPS 对比：见 §10 与下面的设计决定 9。

**变异检查**（每次改一处，对应测试必须失败，之后已还原）：被中断节点也写表 → `An_interrupted_search_...` 失败；所有 bound 都存成 Exact → 固定深度 differential 失败；允许 `entry.depth >= depth - 1` 复用 → ID 与中断测试失败；翻转卡时不更新哈希 → Zobrist 两个测试失败。

#### Phase 4 交付记录（2026-10-07）

**新增 / 修改的文件**（Phase 4 没有修改任何 Python 文件）
- 新增 `csharp/src/TTC.Siren.Core/ZobristKeys.cs`（SplitMix64、固定种子的键表）；修改 `GameState.cs`（`ZobristHash`、`ComputeZobristHash()`；`UndoRecord.HashBefore`；make 时在放置、翻转、`ApplyTypeModifier` 处异或，undo 时直接恢复保存的哈希）。
- 新增 `csharp/src/TTC.Siren.Search/FastTranspositionTable.cs`、`FastSearcher.cs`；修改 `CompatSearcher.cs`（`UseTranspositionTable` 开关，默认 `true`，行为不变；`RunFixedDepthAs(depth, maximizing, aiPlayer, …)`，`RunFixedDepth` 现在委托给它）。
- 新增测试 `ZobristTests.cs`（4 个）、`FastSearchTests.cs`（`Oracle` 辅助 + 8 个 Fast 测试 + 3 个 `FastTranspositionTableTests`）。`csharp/src/TTC.Siren.Bench/Program.cs` 新增 `fast` 模式。

**设计决定**
1. **Zobrist 覆盖**：棋盘每格的卡实例、每个卡实例（棋盘与手牌）的 owner 与 type_modifier、行动方；手牌成员关系由棋盘隐含。**不含**手牌顺序（D-001）、显示边（D-003）、can_use——它们只影响排序，不影响分数（`EvaluateState` 只读 owner 与有效边）。同类规则下手牌卡的修正值会影响之后的走子，所以修正值进入哈希。64 位键；表项比较完整的 64 位键 + 桶下标。未做哈希碰撞的二次校验：按 10^8 次探测、2 项/桶估算，错误命中期望 ≈ 10^8×2/2^64 ≈ 10^-11 次/搜索（计算值，非实测）；走法在使用前要验证在当前局面合法，所以碰撞最多造成一个错误的分数，不会造成非法走法。
2. **undo 恢复保存的哈希** 而不是反向异或：undo 已经逐字段恢复了所有被哈希的状态，所以直接写回 make 前的值，O(1) 且不可能漂移；测试对每一步 undo 断言。
3. **Bound 按原始窗口分类（fail-soft）**：`best <= alpha_in` → Upper，`best >= beta_in` → Lower，否则 Exact；命中时 Exact 直接返回，Lower 仅在 `score >= beta`、Upper 仅在 `score <= alpha` 时返回。没有做 TT 窗口收窄（未测量其收益，且会增加推理负担）。
4. **只在同深度复用分数（`entry.depth == depth`，不是 `>=`）**：一个局面只会出现在固定的 ply，所以固定深度/ID 的同一层里同深度是唯一会发生的情形；这样即使表里还有更深搜索的旧条目（`clearTable:false`），根分数仍是精确的同深度值。更浅或更深的条目仍然提供最佳走法用于排序。
5. **根节点从不被 TT 截断**（需要根走法）；TT 走法先试，且必须在生成列表中（碰撞不能注入走法）；其余走法用与 compat 相同的 `evaluate_move` + history 稳定排序。
6. **被中断的节点不写表**（compat 的 D-006 会写）；一旦超时/节点预算耗尽，`_stopped` 置位并粘住，之后的节点立即返回。ID 丢弃被中断的层（第 1 层除外，与 Python 相同）。
7. **时钟**：墙钟每 1024 个节点读一次（compat 每节点都读）；节点预算（测试用）每个节点检查。墙钟模式本来就不确定。
8. **ID 提前停止（D-014）**：某层的 depth ≥ 空格数（整棵树已搜完）或根方无合法走法时停止；更深的层不会改变结果。因此 Fast 的 `Stats.Levels` 可以少于 compat 的。
9. **depth-1 的非根节点不排序（O-6）**：子节点都是叶子，顺序不影响节点值，排序只是多付 n 次 `evaluate_move`。这是 Phase 4 中唯一一次"以性能为目的"的搜索改动，按 §6 流程测量：Fast d≥4 合计 1.7819 s → 1.4746 s（3 次中位数，见 §10），每次运行都更快，保留；分数与全部 parity 测试不变。（Fast 的节点数因此从 2.82 M 升到 3.63 M——叶子现在都被计数——所以 NPS 在不同变体之间不可比，时间才是指标。）
10. **表**：固定大小、2 的幂、2 项一桶；存储 = 同键覆盖，否则空/旧代槽，否则替换较浅者（新项总能存入）；`NewGeneration()` 用 1 字节代际号 O(1) 清空，回绕时才清数组；项 24 字节（键 8、分数 8、深度/bound/走法/代际 5）；默认 2^22 项 = 96 MiB；**不扩容、不抛异常**。
11. **oracle**：ReferenceCompat + `UseTranspositionTable=false`（即无 TT 的稳定排序 α-β，根分数 = 精确的深度限制 minimax 值），并以独立的全宽 minimax 验证；每个根走法的精确值通过 `RunFixedDepthAs(depth-1, false, rootPlayer, …)` 全窗口得到。
12. **刻意没做**（没有测量依据，留给 Phase 6）：aspiration window、PVS、killer、TT 窗口收窄、多线程、Zobrist 之外的增量评估。

**已知限制**
- Fast 在并列时的选择取决于根排序（history、上一层的 TT 走法、D-001 的手牌顺序），所以与 Python 不一定相同；Phase 7 的端到端比较必须比较 **分数与并列集合成员关系**，而不是字面走法。
- 无 bound 的 Python TT 在 0.07% 的运行里给出非 minimax 的值（D-006）；因此 Fast 在这些局面上 **有意** 与 Python 的走法不同。
- 64 位键没有二次校验（见设计决定 1）；Fast 的表在满时只替换，不会失败。
- 含 Zobrist 的 Make/Undo 原始吞吐约 −5%（§10）；这是 Fast 模式的必要代价，compat 搜索基准没有可测回归。
- 墙钟模式不可确定、不可 golden 化（只有行为测试）；节点预算模式是确定的。
- M-1 到 M-3、L-1 到 L-12、O-1 到 O-5 仍未处理（Phase 4 没有改变它们）。M-1（golden 未绑定参考源码）仍然意味着 Python 语义变化时必须手动提升生成器版本。
- Phase 4 的改动没有经过独立审查。

#### Review remediation (H4-1)（2026-10-07，来自 `docs/reviews/OPUS_PHASE4_REVIEW.md` §6；目标 `.claude/sonnet_review_fix_goal.md`）

**状态：整改完成，等待复审。** 只处理 H4-1；M4-*、L4-*、O4-* 与 Phase 3 遗留项按任务范围 **未处理**。没有改动任何 Python 文件（`git diff --stat -- '*.py'` 为空）；没有改动规则、评估、排序、bound 语义、同深度复用规则、中断处理或 ID 控制流，也没有改动每节点热路径（上下文检查只在 `Begin`，每次搜索一次）。

**问题**：`FastTranspositionTable` 的条目没有有效上下文。存储的分数来自固定的 AI 视角并取决于评估模式（`aggressive`），而 Zobrist 键只区分 **一个 `GameState` 对象内部** 的局面（实例下标在不同状态间会复用；卡牌内容、规则、`can_use` 不在哈希里）。`clearTable:false`（以及在多个 searcher/状态之间共享一张表）会无检查地复用旧条目。审查者的探针：(a) 搜完后调用方走一步再 `RunFixedDepth(D−1, …, clearTable:false)`：6/6 场景错误（如 `end5_same_plus` Fast +2.76，精确值 −2.76）；(b) 同一根、同一深度但 `aggressive` 翻转：6/6 错误；(c) 一张表先后用于 `open_none` 与 `open_same_plus`：32 个深度/模式组合中 3 个错误。相同 `GameState`、相同 AI 方、相同评估模式的复用是 **正确的**（审查 V3：90/90，根前进两步）。

**修改**（新增 `csharp/src/TTC.Siren.Search/SearchContext.cs`；修改 `FastTranspositionTable.cs`、`FastSearcher.cs`，以及 `Core/ZobristKeys.cs` 的文档）
- `enum EvalMode { Heuristic, HeuristicAggressive }`：Phase 5 的精确残局必须新增自己的值，而不是复用启发式值（本整改没有实现任何 Phase 5 内容）。
- `readonly struct SearchContext(GameState State, int AiPlayer, EvalMode Mode)`；`DescribeMismatch` 按 状态 → AI 方 → 评估模式 的顺序返回第一个不同的部分。状态身份用对象引用（`ReferenceEquals`）：一个 `GameState` 的卡牌、规则和 `can_use` 在其生命期内不变。**没有** 把卡牌内容放进每节点的键。
- `FastTranspositionTable`：记录当前代的上下文（`IsBound`、`Context`）；新建的表是 **未绑定** 的；`NewGeneration(in SearchContext)` 开始新一代并绑定；无参的 `NewGeneration()` 保留，但会把表标为未绑定；`RequireContext(in SearchContext)` 在未绑定或上下文不同时抛 `InvalidOperationException`（消息点名不匹配的部分：`different GameState` / `AI side` / `evaluation mode` / `not bound`），不改变任何东西。
- `FastSearcher.Begin`：上下文 = (`_g`，根处的行动方，`aggressive` 对应的 `EvalMode`)。`clearTable:true` → `NewGeneration(context)`；`clearTable:false` → `RequireContext(context)`。该检查在 `Begin` 的最前面，早于对统计、history、时钟的任何修改，所以被拒绝的调用不改变 状态、哈希、统计、history、表、时钟。`RunFixedDepth`、`RunIterativeDeepening`、`FastTranspositionTable`、`ZobristKeys` 的文档注释已写明哈希域与复用规则（"同一个 GameState、同一个 AI 方、同一个评估模式；根可以是任意局面"）。
- `CompatSearcher` / `CompatTranspositionTable` 不在范围内（compat 每次搜索都清表，模拟 Python 的每请求 dict）。
- 取舍：抛异常而不是静默自动清表——静默清表同样正确但会掩盖调用方的 bug（审查建议的保守默认），因此没有新增统计项。

**测试**（新增 `FastTableContextTests.cs`，4 个；对 `ReferenceScenarios` 的 6 个场景与固定种子的 golden 局面，失败信息带 label、D、模式与 golden 种子 20261008）
- `Table_reuse_with_a_different_ai_side_is_rejected`：探针 (a)，6 个场景 × `RunFixedDepth`/`RunIterativeDeepening` = 12 个案例，全部抛 `InvalidOperationException`（消息含 "AI side"）；之后 `UndoDepth`、`ZobristHash == ComputeZobristHash()`、规范化快照（含手牌顺序与显示边）、`Stats`、history、表的 `Stores` 计数与 `IsBound` 都与调用前相同；随后的 `clearTable:true` 搜索与 oracle 位级相等。
- `Table_reuse_with_a_different_evaluation_mode_is_rejected`：探针 (b)，6 个场景 × 2 种方法 × 2 个方向（普通→激进、激进→普通）= 24 个案例，同样的后置条件；同一模式的 `clearTable:false` 复用仍与 oracle 位级相等。
- `Table_shared_between_game_states_is_rejected`：探针 (c)，`open_none → open_same_plus` 与 `open_none → open_all_rules` × 2 种方法 = 4 个案例（消息含 "different GameState"；表仍属于状态 A 且 A 可以继续有效复用；B 可以用 `clearTable:true` 重新开始并与 oracle 位级相等）；另有"新建的未绑定表"与"无参 `NewGeneration()` 之后"各 × 2 种方法（消息含 "not bound"）= 2 个案例；共 6 个案例。
- `Valid_table_reuse_matches_the_oracle`：6 个场景 + `search_id_g1_s20261008_n200` 中 112 个 ≥4 个空格的局面，每个局面在普通与激进模式下：搜 D = min(5, 空格数)，下 Fast 的走法与一个回应（生成顺序中的中间走法），在 `clearTable:false` 下搜 D−2；再回到原根搜 D−1（`clearTable:false`）。448 次复用搜索，其中 396 次实际命中了表，**根分数全部与 oracle（compat，TT 关闭）位级相等，396 个走法全部在精确并列集合内**。
- 变异检查：去掉 `Begin` 里对 `RequireContext` 的调用 → 三个拒绝测试全部失败（已还原）。
- 既有测试 **断言未改动**：`An_interrupted_search_stores_nothing_that_later_searches_could_reuse`（有效复用）与其余所有测试通过；Fast 的分配测试仍为 0 字节（2,072,797 个节点）。
- 记录：新测试的辅助函数里曾有一个我自己写的断言把 `UndoDepth` 硬编码为 0，而 AI 方测试里调用方合法地走了一步（`UndoDepth == 1`）；那是新测试自己的错误，已改为与调用前的值比较（`Probe` 已包含 `UndoDepth`），不是对既有断言的放宽。

**验证（整改后）**：`dotnet build -c Release --no-incremental` 0 warning / 0 error；`dotnet test` **62 通过 / 0 失败 / 0 跳过**（58 + 新增 4）；`python -m pytest -q` 29 passed；`git diff --stat -- '*.py'` 为空。所有既有 Fast-对-oracle 结果不变：`Bench fast` 的节点数完全相同（compat TT 开 3,431,185、TT 关闭 3,771,758、Fast 3,625,620），所有行的分数仍为 same。

**性能**（`Bench fast 3`，d≥4 合计，3 次独立运行；详见 §10）：节点数不变；同一批内 Fast 对 compat TT 关闭 的比值整改前中位数 1.53×，整改后 1.54×；机器负载在两批之间变化很大，绝对时间不可比；整改前最快一批 Fast 1.5308 s 对整改后中位数 1.5680 s = +2.4%。**没有超过 5% 的回归**（检查只在 `Begin`，每次搜索一次）。

**已知限制 / 备注**
- 状态身份是对象引用：对同一局面用 `FromSnapshot` 新建的另一个 `GameState` 会被当作不同状态（保守，需要 `clearTable:true`）。
- 上下文只覆盖今天存在的评估模式。Phase 5 的精确残局 **必须** 新增自己的 `EvalMode` 值（见 §12.8）。
- 审查门禁仍为 `BLOCK_PHASE_5`，等待复审把它改为 PASS（本整改没有修改门禁状态）；Phase 5 仍为 NOT STARTED。

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
| A-9 | D-002 在 Python 和 C# 两边同时修复（按卡身份恢复 `type_modifier`） | 用户批准（Q-1）；否则同类规则下的搜索结果依赖访问顺序，Fast 模式的 parity 无法定义 | 已定（FIXED-BOTH） |
| A-10 | `Move` 的身份是卡实例 `(Inst, Cell)`，不是手牌槽位 | D-001 会在每次 make/undo 后重排手牌；Phase 3 真实出现过由此导致的 parity 错误；Phase 4 的 TT 走法/history/killer 会跨节点保存走法 | 已定（审查 H-3 整改） |
| A-11 | ReferenceCompat TT：128 位摘要代替字符串 key，按需翻倍扩容 + 硬上限（默认 2^25 槽），代际号清空 | Python 的 dict 无界且 key 是字符串；等价类已用 SHA-1 双射验证；扩容不改变结果 | 已定（审查 H-2 整改） |
| A-12 | 搜索入口用 `try/finally` 回退 `UndoDepth`，热路径不加 `try/finally` | 异常后共享的 `GameState` 必须恢复；保持热路径零开销 | 已定（审查 H-1 整改） |
| A-13 | golden 数据不入库：由测试按需调用 Python 生成，文件名与文件头带生成器版本 | 体积大（≈ 26 MB）且可重新生成；缺点是未绑定参考源码内容（审查 M-1） | 已定（M-1 待改进） |
| A-14 | 每个阶段由用户用 `/goal @.claude/*.md` 单独授权；独立审查产生"门禁"（`BLOCK_PHASE_4` / `PASS_TO_PHASE_4`），门禁只能由复审者改，整改者不能自己改 | 控制范围，防止越权进入下一阶段 | 已定 |
| A-15 | **Q-5 = 选项 (b)**：Fast 的根分数必须与精确的深度限制 minimax 值位级相等；根走法只允许在精确并列集合内与 compat/Python 不同，每个差异都要验证并报告；Fast 必须确定性 | 值级结果可精确验证，同时允许 TT 走法优先等排序改进；Python 的 TT（D-006）本身已不等于精确 minimax，所以"与 Python 一致"早已缩小为"与精确值一致" | 已定（用户，2026-10-07） |
| A-16 | Fast 的 oracle = ReferenceCompat 关闭 TT（`UseTranspositionTable=false`），并用独立的全宽 minimax 交叉验证；Fast 的 TT 只在同深度复用分数，不存被中断节点，bound 按原始窗口分类 | 避免 oracle 与被测对象共享同一个缺陷；D-006 使 compat（TT 开）不能当精确值 | 已定（Phase 4） |
| A-17 | Zobrist 哈希在 make 中增量更新，undo 恢复保存的哈希；哈希只覆盖决定局面价值的信息（格子→实例、owner、修正值、行动方），不含手牌顺序与显示边 | 排序相关状态（D-001/D-003）不影响分数；O(1) 的 undo 不会漂移 | 已定（Phase 4） |

---

## 9. BLOCKED / 待用户决定

当前 **没有 BLOCKED 项**（Phase 2、3、审查整改与 Phase 4 期间都没有出现无法解释的 Python/C# 差异；Phase 4 发现的 3 次 Python 与精确值不同，已归因于 D-006，见 §4）。以下问题需要用户在对应阶段之前做出决定（Q-5 已关闭）：

| ID | 问题 | 默认做法（未回复时） | 必须在何时之前决定 |
|---|---|---|---|
| Q-1 | ~~D-002 是否同时修复 Python 和 C#？~~ **已决定：修复（用户，2026-10-07），已完成，D-002 = FIXED-BOTH。** | — | 已关闭 |
| Q-2 | D-004（手牌内 card_id 重复导致状态损坏）：在 Python 侧修复（按卡身份移除），还是只在输入层拒绝/报告？ | C# 按卡身份处理，遇到重复输入抛异常，parity 中排除 | Phase 7 前 |
| Q-3 | D-013 连携只有一层：确认这是有意设计吗？ | 保持 Python 行为 | Phase 8 前（不影响 parity） |
| Q-4 | D-012 卡库 474/475 全 0 空行是否应从候选池中排除？ | 保持 | Phase 5 前 |
| Q-5 | ~~Fast 模式中，同分并列时是否允许选择与 Python 不同的走法？~~ **已决定：选项 (b)（用户，2026-10-07）——只允许在分数位级完全相等的走法之间不同，并逐条验证、报告；已在 Phase 4 完成，见 §7。** | — | 已关闭 |
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
| 2026-10-07 | 4 | C# Core `Bench makeundo`（加入增量 Zobrist 之前 / 之后，各 3 次独立运行，6 场景中位数均值；alloc 0 B） | 之前 9.275 / 9.420 / 8.513 M → 中位数 **9.275 M pairs/s**；之后 8.331 / 7.667 / 8.267 M → 中位数 **8.267 M** | 同机同时段的交错 A/B（HEAD 版 Core vs 含 Zobrist，各 4 次交替运行）：基线 8.553 / 8.445 / 8.441 / 8.316 → 中位数 **8.443 M**；含 Zobrist 7.946 / 8.138 / 8.069 / 7.902 → 中位数 **8.008 M**，**约 −5.2%**（原始 Make/Undo 微基准；是 Fast 模式所需的代价）。尝试"修正值未变时跳过哈希异或"：交替 4 次，基线中位数 8.483 M vs 该变体 8.06 M，无可测改善，已还原（§6） |
| 2026-10-07 | 4 | C# ReferenceCompat `Bench search`（加入 `UseTranspositionTable` 开关与 `RunFixedDepthAs` 之前 / 之后，22 项，43,003 节点，各 3 次运行） | 之前 0.0437 / 0.0474 / 0.0473 s → 中位数 **0.0473 s**；之后 0.0379 / 0.0438 / 0.0410 s → 中位数 **0.0410 s** | 无回归（运行间噪声 ±10%；compat 的行为与节点数不变，43 个 Phase 3 测试全部通过） |
| 2026-10-07 | 4 | C# `Bench fast`，固定深度，depth ≥ 4 的 22 个项合计（open_none / open_same_plus d4–7，mid3_ascension d4–8，mid4_rev_ace d4–7，end5_same_plus d4–5，open_all_rules d4–6），各 3 次运行，每项 3 次取中位数再求和 | **变体 A（Fast = Zobrist + bound TT + TT 走法，所有节点排序）**：Fast 1.7620 / 1.8178 / 1.7819 s → 中位数 **1.7819 s**（2,823,926 节点）；同批 compat TT 关闭 2.3115 s（3,771,758 节点）、compat TT 开 2.3324 s（3,431,185 节点） | 变体 A 相对 TT 关闭 1.30×，节点数为 0.749× |
| 2026-10-07 | 4 | 同上 | **变体 B（最终：A + depth-1 非根节点不排序）**：Fast 1.5272 / 1.4381 / 1.4746 s → 中位数 **1.4746 s**（3,625,620 节点，叶子被计数）；同批 compat TT 关闭 2.2668 / 2.1782 / 2.2076 → **2.2076 s**；compat TT 开 2.3574 / 2.3104 / 2.2853 → **2.3104 s** | **Fast 比 compat TT 关闭快 1.50×，比 compat TT 开（Python 语义）快 1.57×**；B 相对 A −17% 时间，每次运行都更快，保留。所有项的根分数与 oracle 位级相等（`Bench fast` 每行打印 same/DIFF，全部 same） |
| 2026-10-07 | 4 | 同上，最终二进制再测一次（`Bench fast 5`，每项 5 次中位数） | Fast 1.5285 s / compat TT 关闭 2.3098 s / compat TT 开 2.6670 s（d≥4 合计）；单项如 open_none d7：Fast 0.2937 s、TT 关闭 0.4854 s、TT 开 0.7854 s；open_same_plus d7：0.4398 / 0.7064 / 0.6740 s；open_all_rules d6：0.3815 / 0.5100 / 0.5442 s | 1.51× / 1.74×；机器负载变化大（同一项在不同批次相差可达 ±30%），因此以同一批内的比值为准；NPS 随变体的节点计数方式而变，不作为结论；与 Python 的 NPS 对比留给 Phase 5 |
| 2026-10-07 | 4 整改 H4-1 | C# `Bench fast 3`，d≥4 合计，各 3 次独立运行（Fast / compat TT 关闭 / compat TT 开，秒） | 整改前：Fast 2.2973 / 2.0970 / 1.5308，TT 关闭 3.5098 / 3.1323 / 2.3803，TT 开 3.5902 / 3.4099 / 2.4902；整改后：Fast 1.6121 / 1.5671 / 1.5680（中位数 **1.5680 s**），TT 关闭 2.4802 / 2.4941 / 2.3806，TT 开 2.5850 / 2.4428 / 2.5200 | 节点数完全不变（3,431,185 / 3,771,758 / 3,625,620）。机器负载在两批之间变化大（整改前前两次比第三次慢约 40%），绝对时间不可比；同批比值 Fast/TT 关闭：整改前 1.53× / 1.49× / 1.55×（中位数 1.53×），整改后 1.54× / 1.59× / 1.52×（中位数 1.54×）；整改前最快一批 1.5308 s 对整改后中位数 1.5680 s = +2.4%，**无超过 5% 的回归**（检查只在 `Begin`，每次搜索一次） |
| 2026-10-07 | 2 | Python（参考） | `_solve_endgame_exact` 复测 | end5: 217 make / 0.004 s；mid4_rev_ace: 4,707 make / 0.055 s（≈85k make/s） | 与 Phase 1 一致；Python 的 make/s 含 minimax 与 memo 开销，不是纯 make/undo，不可直接比 |

---

## 11. 下一步行动

0. **Phase 1–4 完成。Phase 5（精确残局 + 信息集 robust 残局 + Python-vs-C# 全量 benchmark）尚未开始，等待用户明确授权**（CLAUDE.md：一次只授权一个阶段；可以用 `/goal @.claude/phase_goal.md` 授权，它会选中 Phase 5）。审查门禁 `PASS_TO_PHASE_4` 已完成其使命；Phase 4 的改动没有经过独立审查——如果需要，可以像 Phase 2+3 那样让审查者复核 Phase 4（重点：`FastSearcher` 的 bound 语义、Zobrist 覆盖范围、Q-5 并列集合验证）。
1. Phase 5 开始前需要用户决定：**Q-4**（474/475 空行是否从候选池排除；默认保持）、**Q-6**（是否需要 PyPy baseline；默认只用 CPython）。均有默认值，未回复时按默认。
2. Phase 5 的注意事项：精确残局的终局值必须是 AI 视角的牌数差（`_solve_endgame_exact`），**不能** 复用 `EvaluateState`/`EvaluateMove` 当值；Fast 的 `FastTranspositionTable`/Zobrist 可以复用于残局 memo（Python 的 memo key 包含手牌顺序，见 §1.8，复现时要注意）。
3. 工作区状态：Phase 4 的全部改动 **尚未提交**（`git status`：`CSHARP_MIGRATION.md`、`GameState.cs`、`CompatSearcher.cs`、`Bench/Program.cs` 已修改；`ZobristKeys.cs`、`FastSearcher.cs`、`FastTranspositionTable.cs`、`ZobristTests.cs`、`FastSearchTests.cs` 为新文件）。用户没有要求就不要 commit/push（CLAUDE.md §8）。PR #1 已关闭，`main` 仍是 `eb60a02`。
4. 仍未处理的审查遗留项：M-1 到 M-3、L-1 到 L-12、O-1 到 O-5（见 §12.7）。

---

## 12. 会话交接（Session handoff，2026-10-07 会话结束时）

目的：让一个全新的 Claude Code 会话 **不依赖对话历史** 就能继续。本节汇总前面各节的结论；与前面各节冲突时以前面各节的细节为准，并请修正本节。

### 12.1 仓库与 Git 状态

- 仓库 `D:\SirenPVP\TTC_Siren`（GitHub `extrant/TTC_Siren`），工作分支 **`csharp-solver-migration`**，本会话开始时与 `main` 同为 `eb60a02`。
- 分支上现有 5 个提交（已推送到 `origin/csharp-solver-migration`，HEAD `d0ef721`）：
  1. `aa3fc23` docs：`CLAUDE.md`、`CSHARP_MIGRATION.md`、`.claude/*.md`（目标文件）、`docs/reviews/OPUS_PHASE3_REVIEW.md`
  2. `e6385c0` fix：D-002（`core/game_state.py` + `tests/test_undo_type_modifiers.py`）
  3. `5c70ae3` tools：`tools/parity/py_rules_trace.py`、`py_search_trace.py`、`tools/bench/py_bench.py`、`tools/parity/golden/.gitignore`
  4. `495786d` csharp：`Directory.Build.props`、`.gitignore`、`src/TTC.Siren.Core`、`src/TTC.Siren.Search`
  5. `d0ef721` csharp：`TTC.Siren.sln`、`src/TTC.Siren.Bench`、`tests/TTC.Siren.Tests`
  前 4 个提交单独 checkout 时无法构建整个解决方案（`.sln` 在第 5 个提交里）。
- **PR #1（`csharp-solver-migration` → `main`）已创建后又被用户要求不要提到 `main`，现已关闭**（状态 CLOSED，可 reopen）。`origin/main` 仍是 `eb60a02`，从未被修改。用户的原话意思是："如果已经提交到 `csharp-solver-migration` 了，就不要再提 `main`"。因此提交保留在分支上；**撤销它们需要强制推送，没有得到授权**。若用户之后明确要求撤销，选项是：(a) 本地与远端回退到 `eb60a02`（需要用户明确授权强制推送）；(b) 追加反向提交。
- **（Phase 4 之后）未提交的改动**：`CSHARP_MIGRATION.md`、`csharp/src/TTC.Siren.Core/GameState.cs`、`csharp/src/TTC.Siren.Search/CompatSearcher.cs`、`csharp/src/TTC.Siren.Bench/Program.cs`（已修改）；`ZobristKeys.cs`、`FastSearcher.cs`、`FastTranspositionTable.cs`、`ZobristTests.cs`、`FastSearchTests.cs`（新文件）。CLAUDE.md §8：用户没有要求就不要 commit/push。
- 本机 git 会提示 "LF will be replaced by CRLF"（`core.autocrlf`），无害。

### 12.2 已完成的阶段与用户决定（Phase 4 的行见表末）

| 项 | 状态 | 用户决定 / 备注 |
|---|---|---|
| Phase 1 分析与 baseline | DONE | — |
| Phase 2 规则与 make/undo | DONE | 用户用 `/goal @.claude/phase_goal.md` 授权（该文件内容 = "只做下一个未完成的阶段，不要越界"） |
| **Q-1 / D-002 修复** | DONE，`FIXED-BOTH` | **用户明确批准"修复"**：Python（`core/game_state.py`）和 C# 同时改为按卡牌身份恢复 `type_modifier`。这是唯一被改动的 Python 业务文件 |
| Phase 3 评估/排序/history/minimax/ID（ReferenceCompat） | DONE | 同上用 phase goal 授权 |
| 独立审查 Phase 2+3 | 初审 `BLOCK_PHASE_4`（3 个 HIGH） | 报告 `docs/reviews/OPUS_PHASE3_REVIEW.md` |
| 审查整改 H-1/H-2/H-3 | DONE | 用户用 `/goal @.claude/sonnet_review_fix_goal.md` 授权，范围仅限这三项 |
| 复审 | **`PASS_TO_PHASE_4`** | 复审报告 §15（H-1/2/3 均 RESOLVED，无回归） |
| 创建 PR → 撤回 | 见 12.1 | 用户要求"撤销提交；已推送到分支就不要提到 main" → 关闭 PR #1，保留分支提交 |
| **Q-5** | **决定：选项 (b)** | 用户，2026-10-07：Fast 只允许在分数完全相等的走法之间与 Python 不同，逐条验证、报告 |
| **Phase 4** Zobrist / bound-aware TT / Fast | **DONE**（2026-10-07，`/goal @.claude/phase_goal.md` + Q-5 = (b)） | 验收见 §7 Phase 4；改动未提交 |
| **Phase 5** | **未开始** | 需要用户明确授权 |

### 12.3 代码地图（`csharp/`，.NET `net10.0`，`Nullable`、`TreatWarningsAsErrors`）

- `src/TTC.Siren.Core`：
  - `Primitives.cs`：`OwnerId`、`Side`、`RuleSet`（flags）、`RuleParser`（`ParsePythonRuleString` 等价于 `parse_rules_and_open_mode`）、`Move(Inst, Cell)`、`CardTypes`（固定 4 种）、`InvalidGameStateException`
  - `CardDb.cs`：读 `data/幻卡数据库.csv`（475 行，含全 0 的 474/475）；`FindFirstIdByEdges` 等价 `get_card_lookup`
  - `GameState.cs`：单个可变对象；SoA 卡实例表；`GenerateMoves`、`MakeMove`、`UndoMove`、`RecalculateTypeModifiers`、`CountCards`、`GetWinner`；`MoveFromSlot`/`HandSlotOf`（只给 golden 读取与诊断用）；容量 手牌 5、实例 19、undo 深度 16
  - `Snapshot.cs`（`StateSnapshot`/`CardSnapshot`：导入导出与规范化字符串）、`CoverageCounters.cs`（测试用的规则事件计数；搜索路径上为 null）、`ReferenceScenarios.cs`（§3.3 的 5 个基准局面 + `open_all_rules` + 无分配的随机 playout 驱动）
- `src/TTC.Siren.Search`（ReferenceCompat，与 `ai/ai.py` 节点级一致）：
  - `Evaluator.cs`（`evaluate_state`、`evaluate_move`、落点边值、残局暴露惩罚）、`HistoryTable.cs`（按 card id）
  - `CompatTranspositionTable.cs`（Python TT 的模拟：128 位 `PythonStateKey` 摘要；无 bound 类型；按需翻倍扩容，硬上限默认 2^25 槽 = 1 GiB，超过抛 `TranspositionTableFullException`；代际号 O(1) 清空）
  - `CompatSearcher.cs`（`RunFixedDepth`、`RunFixedDepthAs`、`RunIterativeDeepening`、`SearchStats`、`SearchLimits`（墙钟 + 测试用节点预算）；入口处 `try/finally` 回退状态；`UseTranspositionTable=false` 为 Phase 4 的 oracle）
  - **Phase 4**：`FastTranspositionTable.cs`（固定大小、2 项一桶、bound-aware、代际号）、`SearchContext.cs`（H4-1：`SearchContext`(GameState, AI 方, `EvalMode`) 与 `EvalMode`；表只在相同上下文下可复用）、`FastSearcher.cs`（Fast：Zobrist 键 + bound TT + TT 走法优先；`RunFixedDepth`/`RunIterativeDeepening`，`clearTable` 参数）；Core 里的 `ZobristKeys.cs` 与 `GameState.ZobristHash`/`ComputeZobristHash()`
- `src/TTC.Siren.Bench`：`makeundo`（Make/Undo 吞吐）与 `search`（22 项固定深度）两种模式。
- `tests/TTC.Siren.Tests`（xUnit，**62 个测试**；H4-1 整改新增 `FastTableContextTests`(4)；Phase 4 新增 `ZobristTests`(4)、`FastSearchTests`(8，含 `Oracle` 辅助)、`FastTranspositionTableTests`(3)）：`RulesTraceDifferentialTests`、`PortedRuleTests`、`GameStateBehaviourTests`（D-001/3/4、CardDb 对照 Python 等）、`SearchDifferentialTests`（评估 / 固定深度 / ID+超时 / TT key 四个）、`SearchBehaviourTests`、`SearchExceptionSafetyTests`（H-1）、`CompatTtGrowthTests`（H-2，含 30 s 的服务端等价测试）、`MoveIdentityTests`（H-3）、`AllocationTests`/`SearchAllocationTests`（0 字节）；辅助 `GoldenFiles`、`SearchGolden`。
- Python 侧工具：`tools/parity/py_rules_trace.py`、`py_search_trace.py`（golden 生成器，只 import 参考实现）、`tools/bench/py_bench.py`（基准脚本，用法 `python tools/bench/py_bench.py <repo_root> fixed|exact|server|all`）。
- 审查与目标文件：`docs/reviews/OPUS_PHASE3_REVIEW.md`、`.claude/phase_goal.md`（通用阶段目标）、`.claude/sonnet_review_fix_goal.md`（整改目标，已完成）。

### 12.4 构建、测试、基准、golden 数据（命令与注意事项）

```
dotnet build csharp/TTC.Siren.sln -c Release --no-incremental        # 期望 0 warning / 0 error
dotnet test  csharp/TTC.Siren.sln -c Release                           # 期望 62 通过 / 0 跳过，约 30 s（缓存的 golden 存在时）
python -m pytest -q                                                    # 期望 29 passed（26 原有 + 3 个 D-002 回归）
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search [runs=5]
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- makeundo [pairs=2000000] [runs=5]
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- fast [runs=5]     # Phase 4：compat TT 开 / compat TT 关（oracle）/ Fast
```

- **golden 数据**在 `tools/parity/golden/`（被 git 忽略）。测试发现文件缺失时会自动调用 `python tools/parity/py_*_trace.py` 生成（失败则测试失败，不跳过）。文件名：`rules_trace_g2_s20261007_n20000.jsonl.gz`（≈6 MB，≈20 s）、`search_eval_g1_s20261008_n100000.jsonl.gz`（≈20 MB，≈3 min）、`search_search_g1_s20261008_n900.jsonl.gz`、`search_id_g1_s20261008_n200.jsonl.gz`、`search_tt_g1_s20261008_n3000.jsonl.gz`、`carddb.json`；全部生成约 10 分钟，需要 `PATH` 上有 Python 3.11。目录里可能还有一个不再使用的 `search_search_g1_s20261008_n600.jsonl.gz`（旧计数，可删）。
- **golden 失效规则**：Python 参考实现（`core/*`、`ai/ai.py`）的语义改变时，必须同时提升生成器版本（`GENERATOR_VERSION`，规则 golden 为 `py_rules_trace/2`，搜索 golden 为 `py_search_trace/1`）和 `GoldenFiles.cs` 里对应的常量/文件名里的 `g2`/`g1`；测试会校验文件头里的版本号。**但 golden 没有和参考源码内容绑定（审查 M-1）**：改了 Python 却忘了提升版本号，测试会继续对着旧数据通过。
- 基准机器噪声大：同一代码连续运行就能差 ±10%。比较前后性能必须在同一时段、同一机器、至少 3 次独立运行取中位数；单次差异不能归因于改动。
- 开发环境陷阱（本会话踩过）：
  - PowerShell 里 `R`、`Rp` 是内置别名（`Invoke-History`、`Remove-ItemProperty`），不能用作自定义函数名（导致批量文本替换静默失败）。
  - Bash 工具禁止前台 `sleep` 等待；要等后台任务就用 `run_in_background` 加 `until` 循环，或直接轮询日志文件。
  - `dotnet new sln` 在 SDK 10 默认生成 `.slnx`；本仓库用经典 `.sln`（`--format sln`）。
  - C# 里 `Span<T>`（`stackalloc`）不能被 lambda 捕获；测试里要用普通循环。
  - 修改 `.md` 里的大段中文用 Edit 工具（或 UTF-8 无 BOM 的 `[IO.File]::WriteAllText`），不要用会改写编码的重定向。
- 最近一次完整验证（会话结束前）：build 0 warning/0 error；`dotnet test` 43/43；`pytest` 29 passed；golden 计数：固定深度 4,507 次/1,152,695 节点（900 局面 + 22 基准项）、ID 203 + 600（586 次被节点预算中断）+ 200 次直接带预算 minimax、评估 100,000 局面 / 400,000 + 1,592,088 次检查、规则轨迹 20,000 序列/182,910 步、TT key 3,000 家族/16,733↔16,733；分配测试 0 字节（1,000,002 次 make/undo；371,212 个搜索节点）。

### 12.5 关键设计决定（汇总；细节见 §7 各 Phase 记录与 §8、§4）

1. **Python 是参考实现**；已知怪异行为由 C# 复现并以 `// PY-QUIRK(Dxxx)` 标注：D-001（undo 把卡追加到手牌末尾）、D-003（显示边不随 undo 恢复，`evaluate_move` 读显示边）、D-004（手牌内重复 id → C# 在载入时抛异常）、D-005/D-006（TT key 拼接冲突、TT 无 bound）、D-007（终局仍用启发式）、D-013（连携只有一层）、D-014（ID 不会提前停）。**D-002 已修复并同步到两边（FIXED-BOTH）。**
2. **浮点按位比较**：`float.hex()` ↔ `BitConverter.DoubleToInt64Bits`；严格保持 Python 的运算与累加顺序；不用 epsilon、不用 FMA。
3. **Move 的身份是卡实例 `(Inst, Cell)`**，不是手牌槽位（D-001 会在每次 make/undo 后重排手牌）。这个错误在 Phase 3 真实发生过（`open_none` depth 2：Python −4.56，C# −1.375），审查 H-3 把它从类型层面消除。Phase 4 的 TT 走法、history、killer 必须用 `Move`。
4. **翻转记录用两个 9 位 mask**（`FlipMask` + `FlipFromNoneMask`），因为 Python 的 owner 是三值（`None` 的卡也会被翻）。
5. **同类强化/弱化的修正值快照按卡实例保存**（D-002 修复）；显示边（D-003）仍是每实例 4 字节的陈旧状态。
6. **有效值现场计算**（`clamp(base + mod, 1, 10)`），没有 `EffEdges` 缓存——没有 benchmark 证据说明需要。
7. **TT**：compat 的 TT key 等价于 Python `get_state_hash`（棋盘每格：owner、id/None、四条 base 边值、type_modifier；手牌 id 字符串无分隔拼接；当前玩家；**不含** can_use、卡类型、手牌卡的修正值/owner、显示边），用 128 位自制混合哈希代替字符串，等价类已用 Python SHA-1 做双射检查；条目 32 字节；按需翻倍扩容，硬上限抛异常；`NewGeneration()` 用代际号 O(1) 清空；一次 ID 内各层共享同一张 TT。
8. **搜索入口异常安全**：`RunFixedDepth`/`RunIterativeDeepening` 在 `finally` 里把 `UndoDepth` 回退到入口值；每节点热路径没有 `try/finally`。`MakeMove` 在改动状态之前完成所有校验。
9. **时钟**：`SearchLimits`（无限时间 = 不读时钟；墙钟；测试用的节点预算 `NodesSearched > N`，与 Python 侧被替换的 `is_time_up` 一一对应）。固定深度 + 无限时间是 parity 模式（CLAUDE.md §5）。
10. **不实现**：Python 的 `path`/PV（`ai_server` 只取 `best_move`）、`branching_factor`、进度回调、verbose 输出、"预测下一层耗时"块（只打印，D-014）。
11. 搜索算法实际是 **fail-soft**（返回未截断的最佳子分数），不是文档/注释里写的 fail-hard（审查 L-3，注释尚未更正）。

### 12.6 基准结果汇总（原始数据与方法见 §3、§10）

- Python（CPython 3.11.4，D-002 修复后）§3.4 的 22 个固定深度项合计：43,003 节点 / 2.965 s ≈ 14.5 k NPS。
- C# ReferenceCompat 同 22 项：43,003 节点 / ≈ 0.045–0.052 s ≈ 0.85–0.96 M NPS（≈ 60–66× 总耗时），节点数、分数位级、走法与 Python 完全相同；热路径 0 分配。open_none d4：0.0231 s（Python 1.089 s，≈ 47×）。
- Make/Undo（Core，随机 playout，6 场景中位数均值）：≈ 6–9.5 M pairs/s；含同类规则的场景更慢（≈ 1.2–3 M）。H-3 前后：9.231 → 9.243 M（+0.1%）；search 0.0483 → 0.0486 s（+0.6%），均在噪声内。
- 服务端等价调用 `RunIterativeDeepening(10, false, WithTime(10))`（C#，默认 TT）：open_none 10 层/15.9 M 节点/TT 7.28 M 条/512 MiB；open_same_plus 9 层/16.7 M 节点/8.25 M 条/512 MiB；mid3_ascension 10 层/30,500 节点/64 MiB；open_all_rules 7 层/10.6 M 节点/3.80 M 条/256 MiB（墙钟相关，层数会波动）。Python 同样 10 s 只完成约 5 层。
- 还没有的：PyPy baseline（Q-6）、`_solve_endgame_exact`/robust 残局的 C# 版本（Phase 5）、任何 Fast 模式数据（Phase 4）、profiler 数据（Phase 6）。**没有数据支撑的性能结论不得写入文档。**

### 12.7 未解决的问题与待办（均不阻塞 Phase 4，除非标明）

**需要用户回答**
- ~~Q-5~~ 已决定：选项 (b)（用户，2026-10-07），Phase 4 已实现并验证（§7 Phase 4）。
- Q-2（D-004 重复手牌 id，Phase 7 前）、Q-3（连携只有一层是否有意，Phase 8 前）、Q-4（474/475 空行是否排除，Phase 5 前）、Q-6（是否需要 PyPy baseline，Phase 5 前）：默认值见 §9。
- 是否把 `csharp-solver-migration` 合并到 `main`（目前用户不希望提 PR）；是否撤销已推送的提交（需要强制推送授权）。

**审查遗留项（`docs/reviews/OPUS_PHASE3_REVIEW.md`；未处理）**
- M-1：golden 没有绑定参考源码（建议在文件头写入 `core/*.py`、`ai/ai.py`、`csv_compat.py`、生成器、CSV 的 SHA-256，并在测试里重算；校验 python 主次版本）。**建议在 Phase 4 依赖新 golden 之前处理。**
- M-2：搜索后的手牌顺序和显示边没有和 Python 对比（建议在生成器里每次搜索后导出 `export_state`，C# 比较完整规范化字符串）。**同上。**
- M-3：搜索 golden 偏向后期局面（85% 占用 ≥ 4；节点上限 5,000）；缺少永久的"开局 + 同类规则、双方手牌同 id、服务端风格原始 Card、474/475 与 1000+ id"的 golden（审查的对抗性生成器 V2 没有入库）。**同上。**
- L-1：`maxDepth > 128` 会 `IndexOutOfRangeException`（`SearchStats.MaxLevels`）；L-2：`default(SearchLimits)` = "已超时"，NaN/负时限未校验；L-3：把 "fail-hard" 注释/文档改为 fail-soft；L-4：`PythonStateKey` 的哈希注释引用了不存在的 `PythonStateKeyTests`（真实测试名 `Transposition_key_partition_equals_pythons_get_state_hash`），"2^-128" 说法偏乐观；L-5：`AllocationTests` 没有断言状态恢复；L-6：文档措辞过期（`.claude/phase_goal.md` 里的 "26 passed"、`CLAUDE.md` §7 的 26 基线应说明现为 29；`CLAUDE.md` 没有"人工决策协议"章节）；L-7：Q-2 决定前 Phase 7 的重复 id 输入会在 C# 抛异常；L-8：（已处理：现已提交并推送到分支）；L-9：`CardSnapshot.FromJson` 把 `"r"` 以外的 owner 都当蓝方；L-10：`MakeMove_rejects_a_full_undo_stack_…` 实际测的是"格子已占用"，undo 栈满分支没有直接测试；L-11：扩容瞬间新旧数组并存（≈ 1.5× 峰值内存）；L-12：服务端等价测试依赖墙钟（≈ 30 s，断言 ≥ 3 层且 < 20 s），在高负载 CI 上可能不稳定。
- O-1…O-6（只有在 CLAUDE.md §6 的 Baseline→…→Compare 流程下才可做）：每节点 `stackalloc` 清零、`IsTimeUp` 每节点读时钟、同类规则下每步快照 19 个修正值并全盘重算、`EvaluateState` 每个叶子全盘重算、`PythonStateKey` 每个内部节点构造 ≈ 220 字节缓冲区并哈希（**Fast 模式已用增量 Zobrist 取代**）、depth-1 节点也做完整排序（**Fast 已在 Phase 4 省略，经测量保留；compat 不变**）。

### 12.8 下一步行动（按顺序）

0. **（2026-10-07 复审：H4-1 RESOLVED，门禁 `PASS_TO_PHASE_5`，见 `docs/reviews/OPUS_PHASE4_REVIEW.md` §13；阻塞已解除。以下为复审前的原文，保留作记录。）** **（H4-1 已整改，等待复审；下面是整改前的原文，保留作记录）** 整改记录见 §7 "Review remediation (H4-1)"：复审者复核 H4-1 的 diff 后把门禁改为 `PASS_TO_PHASE_5`；在此之前仍然 **不得开始 Phase 5**。
   **Phase 4 审查门禁 `BLOCK_PHASE_5`（2026-10-07，`docs/reviews/OPUS_PHASE4_REVIEW.md`）**：先执行 `/goal @.claude/sonnet_review_fix_goal.md`，只修 H4-1，然后复审。门禁改为 PASS 之前 **不得开始 Phase 5**，即使 `.claude/phase_goal.md` 按 §0 选中了 Phase 5。注意下面第 3 条中"`FastTranspositionTable` 可复用"的说法：只有在 **同一个 GameState、同一 AI 方、同一评估模式** 下复用才正确（H4-1）；不同场景的 GameState、精确残局的计分都 **不能** 与启发式搜索共用一代表项。
1. **等待用户**：明确授权 Phase 5（可用 `/goal @.claude/phase_goal.md`，它会选中下一个未完成的阶段 = Phase 5；开始前先重读 §0 与 §7 Phase 5）。Q-4、Q-6 有默认值（§9）。
2. （建议，不强制）Phase 5 之前：M-1 到 M-3（golden 绑定参考源码哈希；导出并比较搜索后的完整状态；把对抗性用例并入永久 golden），以及对 Phase 4 做一次独立审查。这些都是测试/工具改动，不改变搜索语义。
3. **Phase 5 的设计约束**（来自 Phase 4 的经验）：
   - 精确残局里终局值是 AI 视角牌数差，**不能** 用 `EvaluateState`/`EvaluateMove` 当值；`EvaluateMove` 读陈旧显示边，只能用于排序。
   - **（H4-1，已实现的机制）Fast 表的上下文**：精确残局的评分必须在 `EvalMode` 里新增 **自己的值**（不能复用 `Heuristic`/`HeuristicAggressive`），或者使用自己的表；robust 层的每个场景 `GameState` 都是不同的状态对象，**场景之间不得共享同一代表**（`FastTranspositionTable.RequireContext` 会拒绝，需要 `clearTable:true`）。在同一个场景 `GameState`、同一个 AI 方、同一个评估模式内复用是正确的。
   - Fast 的 `GameState.ZobristHash` 与 `FastTranspositionTable` 可复用；但 Python 的 `_endgame_state_key` 含手牌**顺序**、can_use 与 rules（§1.8）。Phase 5 的验收是 exact 值位级相等（整数牌数差），所以对 Python 的 memo 语义只需要"结果相同"，实现可以不同。
   - 比较 Fast/compat 与 Python 的走法时，按 Q-5 (b)：比较分数与并列集合成员关系，不比较字面走法；已知 3/4,507 的 D-006 差异（§4）。
   - 任何以性能为目的的改动都要走 CLAUDE.md §6（Baseline → Modify → Build → Test → Benchmark → Compare），不要引入未测量的 `unsafe`、SIMD、并行。
4. Phase 5 完成后：更新本文件（状态、设计决定、测试、benchmark、已知限制、下一步），并同步本节（§12）。
5. 后续阶段顺序见 §0/§7：Phase 6（profiler 驱动优化）→ 7（Python 服务调用 C# 引擎，需先定 Q-2）→ 8（全量服务移植）→ 9（可选 CP-SAT）。

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
