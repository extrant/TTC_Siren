# CSHARP_MIGRATION.md — TTC_Siren Python → C# 迁移进度（唯一事实来源）

> 每个 Claude 会话：先读 `CLAUDE.md`（长期规则），再读本文件。
> 本文件记录：Python 架构、性能热点、baseline、C# 目标架构、Phase 定义与状态、设计决定、语义差异、benchmark 历史、BLOCKED、下一步。
> 任何 Phase 完成后必须更新本文件。
>
> **新会话快速上手：先读 `CLAUDE.md`，再读本文件 §12「会话交接」（当前状态、如何构建/测试、用户已做的决定、未解决问题、下一步），然后按需读 §0 与 §7。**
> **Phase 8A 已完成，独立审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`（2026-10-11，`docs/reviews/OPUS_PHASE8A_REVIEW.md`；0 CRITICAL/HIGH；M8-1 发布命令失败、M8-2 依赖时间的调度测试，上线前需修 M8-1 或按审查 §8 变通；任务 `.claude/phase8a_goal.md`，未开始 8B/8C/8D，未推送）**：后端成为产品服务器——原生 C# `/ai_move`（请求解析、未知牌采样与 CPython `random` 逐位一致，960 个请求与 Python 服务器端到端逐值相同）、蒙特卡洛下线、解释事实 v2 + 中文模板文案、可重放对局日志、`/v1/live` 事件流、API 密钥/限流/HTTPS/`/api/*` 代理、O8-1 调度；报告 `docs/reports/PHASE8A_REPORT.md`，交付记录见 §7 Phase 8A，非位级一致项 D-020…D-025。
> 当前一句话状态：Phase 1–7 完成（Phase 7 于 2026-10-10 完成，独立审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（`docs/reviews/OPUS_PHASE7_REVIEW.md`；0 CRITICAL / 0 HIGH，M7-1…M7-3 不阻塞 Phase 8，但上线前建议修复或按审查 §9 配置），验收见 §7 Phase 7、`docs/reports/PHASE7_INTEGRATION.md`、`docs/DEPLOY_WINDOWS.md`；Phase 6 门禁 `PASS_WITH_NONBLOCKING_FINDINGS`）；**Phase 7 加固（`.claude/phase7_hardening_goal.md`，M7-1/M7-2/M7-3/L7-1/L7-2/L7-3 + A-20 取消自动回退）已于 2026-10-10 完成，独立审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`**（`docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md`；0 CRITICAL/HIGH/MEDIUM，M7-1…M7-3、L7-1…L7-3 已解决；`docs/reports/PHASE7_HARDENING.md`；现在 C# 引擎是 minimax 的唯一求解器，出错返回 503 + `engine_error`，仅保留人工开关 `TTC_CS_ENGINE=off`）。Phase 8 **未开始**，需用户授权。分支 `csharp-solver-migration`：Phase 1–6 与 Phase 6 收尾文档已推送到 `origin`（`03b4704`，用户授权的唯一一次推送，原样快进、未压缩）；**Phase 7 的提交与审查文件已随 Step A 推送（`origin/csharp-solver-migration` = `45a9b28`，原样快进）；Phase 7 加固的提交只在本地，未推送**；**没有**对 `main` 的 PR——PR #1 已按用户要求关闭。

---

## 0. 状态总览

| Phase | 名称 | 状态 |
|---|---|---|
| 1 | Python 架构分析 / profiling / baseline | **DONE**（2026-10-07） |
| 2 | Card / Move / RuleSet / GameState / MakeMove / UndoMove + 规则 parity 工具 | **DONE**（2026-10-07，验收见 §7 Phase 2） |
| 3 | Evaluator / MoveOrdering / History / Minimax / AlphaBeta / ID（ReferenceCompat，节点级 parity） | **DONE**（2026-10-07，验收见 §7 Phase 3） |
| 4 | Zobrist / 带 bound 的 TranspositionTable / Fast 搜索模式 | **DONE**（2026-10-07，Q-5 = 选项 (b)，验收见 §7 Phase 4） |
| 5 | 精确残局 + 信息集 robust 残局 + Python-vs-C# 全量 benchmark | **DONE**（2026-10-08，验收见 §7 Phase 5；先行解决了 M4-1/M4-2；benchmark 报告 `docs/reports/PHASE5_BENCHMARK.md`） |
| 6 | Profiler 驱动优化（单线程）→ 可选 root parallelism | **DONE**；独立审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（2026-10-10，`docs/reviews/OPUS_PHASE6_REVIEW.md`）（2026-10-10；root parallelism 未授权、未做）— 以下为授权前的审查记录： Phase 5 benchmark 独立审查结论 **`BENCHMARK_VALID_WITH_CAVEATS`**（`docs/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md`：没有正确性回归；49.7× 是冷 JIT 数字，热态约 110×；精确求解的 263× 不可复现，约 165×）。建议 Phase 6 之前先执行 `.claude/sonnet_review_fix_goal.md`（只修 benchmark 工具与报告），否则 §6 的 Baseline→Compare 不可靠。**整改复审（2026-10-08，审查文档 §12）**：B-1 到 B-5 全部 RESOLVED，结论仍为 `BENCHMARK_VALID_WITH_CAVEATS`，更正后的数字在固定到 P 核的复测中得到确认；新发现 **A-7**（混合架构 CPU 的 P/E 核调度不受控，未固定亲和性时比值可偏高约 1.6–1.7×，C# 自身在 E 核上慢约 1.45×），Phase 6 之前需先执行新的 `.claude/sonnet_review_fix_goal.md`（只含 A-7）。**A-7 复审（2026-10-08，审查文档 §13）**：A-7 RESOLVED（`tools/bench/pin.py` + `run_interleaved.py --affinity`，两边固定在同一 P 核集，独立复测一致），B-1 到 B-5 无回归，结论 `BENCHMARK_VALID_WITH_CAVEATS`；可信范围（对 CPython 3.11.4、两边都在 P 核）：Python 语义搜索 warm 113–118×、cold ≈52–91×（双峰）；robust 流水线 warm 169–180×、cold 109–123×；精确求解 159–176×。**从 benchmark 角度可以开始 Phase 6**（仍需用户明确授权；Phase 6 的所有 Baseline→Compare 必须固定亲和性并记录掩码）。**用户已授权 Phase 6（2026-10-09）**：任务 `.claude/phase6_goal.md`（A：提交并推送 Phase 5 —— 已完成，`b6e4ea4`；B：Q-2/D-004 双边修复 —— 已完成，`a0eb193`，本地；C：Phase 6 的性能工作 —— 已完成，本地）。**Phase 6 状态：DONE，等待独立审查**（见 §7 Phase 6 交付记录与 `docs/reports/PHASE6_OPTIMIZATION.md`）；B、C 的提交未推送，Phase 7 未开始 |
| 7 | 集成策略：Python 服务调用 C# 引擎（桥接 + 回退） | **用户已授权（2026-10-10）**：任务 `.claude/phase7_goal.md`（A：Phase 6 收尾并推送现有本地提交 → B：Phase 7 集成，提交只在本地）；用户决定：C# 引擎是只监听 127.0.0.1 的独立 HTTP 服务（ASP.NET Core，依赖服务器上安装的 .NET 10 运行时）、请求级并行（引擎内工作线程池，不做 root parallelism）、默认开启并在出错/超时时自动回退 Python、搜索参数沿用 Python（`max_depth`=10、`max_time`=10 s）、部署目标 Windows Server（E5-2698B v3，16 GB）；**DONE**，独立审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（2026-10-10，`docs/reviews/OPUS_PHASE7_REVIEW.md`）（2026-10-10；验收见 §7 Phase 7；报告 `docs/reports/PHASE7_INTEGRATION.md`、部署 `docs/DEPLOY_WINDOWS.md`；Phase 7 的提交只在本地，未推送；Phase 8 未开始；不自设审查门禁）；**Phase 7 加固已获用户授权（2026-10-10）**：任务 `.claude/phase7_hardening_goal.md`（先推送 Phase 7，再修 M7-1/M7-2/M7-3/L7-1/L7-2/L7-3，并取消自动回退 Python，见 A-20、Q-8）；**Phase 7 加固 DONE**，审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（2026-10-10，`docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md`）（2026-10-10；Step A：审查者文件提交 `45a9b28` 并推送 `03b4704..45a9b28`；Step B 加固提交只在本地，未推送；见 §7 Phase 7 加固交付记录与 `docs/reports/PHASE7_HARDENING.md`；不自设审查门禁）；Phase 8 未开始 |
| 8 | 全量服务移植（请求解析、未知牌采样、MC、分析文案、app.py 子系统、HTTP） | **按用户决定（2026-10-11，Q-9）拆成 8A→8B→8C→8D，逐段授权与审查**，设计见 `docs/design/PHASE8_PLAN.md`。**8A（后端成为产品服务器：/ai_move 原生 C#、解析与采样按原逻辑移植并与 Python 位级一致、MC 下线、解释事实 v2 + 模板文案、对局日志、实时事件流、API 密钥/限流/HTTPS、O8-1 调度）已获授权**：任务 `.claude/phase8a_goal.md`；**已完成（2026-10-11），审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`（`docs/reviews/OPUS_PHASE8A_REVIEW.md`）**：Step A–D 全部完成，见 §7 Phase 8A 交付记录与 `docs/reports/PHASE8A_REPORT.md`。8B（Web UI：实时求解界面、实战跟随、3D 棋桌与 3D 翻牌、日志回放、看板娘）、8C（牌组优化 + 移植 /api/*）、8D（对局指正、自动复盘报告、残局谜题、看板娘台词库）：未授权 | **（2026-10-11 更新，Q-10）8B 已获用户授权**：任务 `.claude/phase8b_goal.md`（A：提交并推送现有全部提交与看板娘原图；B：修 8A 遗留 M8-1/M8-2/L8-1…L8-4 + Cloudflare 适配；C：访问分级与算力分级（插件密钥最高优先、网页用密钥换会话 Cookie、匿名只读并设上限，A-27）；D：Web UI，中英双语，A-29）；进行中 → 完成后等待独立审查。新增 **8E：NPC 通杀库**（A-28，规划中，未授权）；顺序 8B → 8C → 8D → 8E |
| 9 | 可选：CP-SAT 隐藏牌组合 / deck optimization / scenario generation | NOT STARTED |

**Phase 4 审查门禁（2026-10-07）**：结果 **`BLOCK_PHASE_5`**，报告 `docs/reviews/OPUS_PHASE4_REVIEW.md`。Zobrist（覆盖范围与增量更新）、bound 语义（按原始窗口分类、fail-soft）、同深度复用、被中断的搜索不写表、TT 走法校验、history 都经独立验证为正确：在新种子 golden 上 Phase 4 的 15 个测试全部通过；对抗性用例 1,756 次运行（含 4 项的极小表）与 oracle 0 不一致。唯一的阻塞项是 **H4-1**：TT 表项不带有效上下文（GameState、AI 方、评估模式），公开参数 `clearTable:false` 在换边、切换 aggressive、跨 GameState 共表时会 **静默返回错误分数**（审查探针分别为 6/6、6/6、3/32 错误）。修复任务：`.claude/sonnet_review_fix_goal.md`。复审改为 PASS 之前不得开始 Phase 5。
**整改进展（2026-10-07）**：H4-1 **已修复，等待复审**（记录见 §7 Phase 4 的 "Review remediation (H4-1)"；测量见 §10）：`FastTranspositionTable` 现在记录当前代的 `SearchContext`（GameState 对象、AI 方、`EvalMode`），`clearTable:false` 在上下文不同或表未绑定时在 `Begin` 里抛 `InvalidOperationException`（不改变任何东西）；62/62 测试通过。门禁状态 **仍为 `BLOCK_PHASE_5`**，由复审者而不是整改者改为 PASS；Phase 5 仍为 NOT STARTED。M4-1 到 M4-3、L4-1 到 L4-5、O4-1 到 O4-3 按任务范围未处理。
**复审结果（2026-10-07，`docs/reviews/OPUS_PHASE4_REVIEW.md` §13）**：H4-1 经独立验证为 **RESOLVED**，门禁改为 **`PASS_TO_PHASE_5`**（这是现行状态，上面两行是记录）。验证内容：62/62 测试通过，pytest 29，0 warning，没有改动 Python；审查者原来的 H4-1 探针（换边、切换 aggressive、跨 GameState、第二个 searcher、ID、逻辑克隆）51/51 被拒绝，0 个静默错误，拒绝后 状态/哈希/UndoDepth/表 均不变；变异检查：去掉 `RequireContext` → 3 个拒绝测试失败，单独去掉 AI 方 / EvalMode / GameState 的比较 → 分别只有对应的测试失败；热路径（`Node`/`TryGet`/`Store`）没有改动，分配为 0。剩余 CRITICAL 0、HIGH 0；新增 LOW：L4-6（`TryGet`/`Store` 仍是公开且不带上下文的接口，Phase 5 若新增求解器直接使用表，必须用自己的 `EvalMode` 绑定）、L4-7（复用测试的"命中"指标偏高，不影响正确性）。Phase 5 仍需用户明确授权；建议先处理 M-1/M-2（不阻塞）。

> 以下三段是按时间顺序追加的门禁历史：初审 `BLOCK_PHASE_4` → 整改完成 → 复审 `PASS_TO_PHASE_4`。**现行状态是第三段（PASS_TO_PHASE_4）**；前两段保留作为记录，其中"不得开始 Phase 4"的限制已随复审解除。

**审查门禁（Phase 2+3 独立审查，2026-10-07）**：结果 **`BLOCK_PHASE_4`**。报告：`docs/reviews/OPUS_PHASE3_REVIEW.md`。阻塞项为 3 个 HIGH：H-1 搜索异常时 GameState 未恢复；H-2 ReferenceCompat TT 在服务端等价配置下溢出；H-3 `Move` 以手牌槽位为身份。规则、评估、搜索与 Python 的 parity 已被独立复核（新种子重新生成全部 golden，外加对抗性用例，0 不一致）。修复任务：`.claude/sonnet_review_fix_goal.md`。修复完成并经复审改为 PASS 之前，**不得开始 Phase 4**（即使 `.claude/phase_goal.md` 按 §0 选中了 Phase 4）。Phase 4 开始前仍需用户决定 Q-5。
**整改进展（2026-10-07）**：H-1、H-2、H-3 的整改已完成，**等待复审**（记录见 §7 Phase 3 的 "Review remediation (H-1/H-2/H-3)"；测量见 §10）。门禁状态 **仍为 `BLOCK_PHASE_4`**，由复审者而不是整改者改为 PASS；Phase 4 仍为 NOT STARTED。
**复审结果（2026-10-07，`docs/reviews/OPUS_PHASE3_REVIEW.md` §15）**：门禁改为 **`PASS_TO_PHASE_4`**。H-1、H-2、H-3 均经独立验证为 **RESOLVED**，没有引入回归：43/43 测试通过；在原 golden 与新种子 golden 上 parity 计数完全不变；对抗性用例 0 不一致；"过期走法" fuzz 115,955 次 0 错误；中途抛异常（含调用方已有走法、ID 第 2 层以后抛出）时状态都能恢复；服务端等价调用 4 个局面全部完成，TT 峰值 512 MiB。M-1 到 M-3、L-1 到 L-12、O-1 到 O-6 仍未处理，均不阻塞（建议在 Phase 4 依赖新 golden 之前处理 M-1 到 M-3）。Phase 4 仍需用户明确授权；Q-5 的建议见审查报告 §15.6（选项 b：只允许在分数完全相等的走法之间不同，并逐条验证、报告）。
**Phase 5（2026-10-08）**：用户授权 Phase 5；按用户要求先解决 Phase 4 审查的 M4-1（哈希充分性的独立测试）与 M4-2（golden 绑定到 Python 源码），再做 Phase 5。Phase 5 **完成**（见 §7 Phase 5）：精确残局、信息集 robust 残局、确定性的场景构造与覆盖逻辑与 Python 位级/逐条一致，benchmark 报告已产出。Phase 5 的改动没有经过独立审查；Phase 6 未开始。

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
  1. `hand_card = 手牌中第一张 card_id 相同的卡`（不是按对象身份！见 D-004）**【2026-10-10 更正：D-004 已在两边修复（FIXED-BOTH）——现在先按对象身份找，找不到才退回按 card_id；这里描述的是修复前的行为】**；格子非空或找不到则返回 None。
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
| D-004 | （**已修复**）原行为：同一手牌里有重复 `card_id`（两个 None 占位牌、四边相同的两张牌被识别成同一 id、预测牌重复）时，`make_move` 移除的是第一张同 id 的卡，放到棋盘上的却是传入的那张 → undo 后出现重复引用、原卡丢失（Phase 1 实测）。**修复**（用户批准 Q-2，2026-10-09）：Python `make_move` 先按对象身份在手牌里找（`c is card`），找不到才退回按 `card_id` 匹配（`count_captured_cards` 传入 `card.copy()`，仍走回退）；`undo_move` 不变（D-001、D-003 保持）。C#：`GameState.FromSnapshot` 不再拒绝重复 id（每个卡实例本来就有自己的身份，`Move(Inst, Cell)`、哈希、undo 无需改动）。回归测试：Python `tests/test_duplicate_hand_ids.py`（5 个；其中 3 个在旧代码上失败：同 id 两张的第二张、两张 None、300 个固定种子的随机 make/undo 往返），C# `Duplicate_hand_ids_are_supported`（取代 `D004_duplicate_hand_ids_are_rejected`，后者断言的正是被批准改变的行为）+ 全量 differential（golden 现含重复 id，见 §7 "D-004 修复记录"） | 修复前：状态损坏；修复后：重复 id 的手牌是合法输入 | 两边同时修复 | **FIXED-BOTH** |
| D-005 | TT key 中手牌 id 无分隔符拼接：`[1,23]` 与 `[12,3]` 的 key 相同（实测碰撞） | 罕见的错误 TT 命中 | ReferenceCompat 模式复现；**Fast 模式已改用 Zobrist（Phase 4，无此问题）** | OPEN-EMULATE（compat；Fast 不继承） |
| D-006 | TT 不存 bound 类型；cutoff/超时后的部分结果也按精确值存储，命中条件只看深度 | 理论上不正确。§3.7 的 4 个样本中未观察到影响；**Phase 4 在 4,507 次固定深度运行中实测：3 次（0.07%）Python（TT 开）的根分数 ≠ 精确 minimax 值，且 Python 的走法不在精确并列集合内**（`i=186 d5`、`i=383 d5 激进`、`i=397 d4`，见 §7 Phase 4 的 Q-5 报告；ReferenceCompat 与 Python 一致，所以这三次是 TT 语义造成的，D-005 碰撞与 D-006 无 bound 在此未分开归因） | ReferenceCompat 复现；**Fast 模式使用 exact/lower/upper，只在同深度复用，不存被中断节点（Phase 4 已实现并验证）** | OPEN-EMULATE（compat；Fast 不继承） |
| D-007 | 终局（棋盘满）仍用启发式 `evaluate_state`，没有胜负大分 | 搜索可能偏好"启发式分高但输掉"的线 | 不改，复现 | OPEN-EMULATE |
| D-008 | `ai.ai` 的 TT/统计/计时都是模块级全局，Flask 多线程并发请求会互相干扰 | 并发时结果错误 | C# 每次搜索独立上下文（行为上等价于单请求） | OPEN（C# 天然不复现，记录即可） |
| D-009 | `ai.get_card_star_map()` 用相对 cwd 的路径 | 从别的目录启动时会崩溃 | C# 用程序目录定位数据 | OPEN（集成时处理） |
| D-010 | `count_captured_cards` 用复制品 make/undo → 手牌中的原卡被替换为复制品，并改变顺序 | 只影响文案和输出预测棋盘 | Phase 8 移植文案时复现或修复（届时决定）→ **8A 决定：复现**（备选走法的选取依赖手牌顺序，端到端比较需要；见 D-024） | OPEN-EMULATE |
| D-011 | 请求解析中 `numL→right`、`numR→left` 左右互换（`ai_server.py:161, :809`） | 协议语义，必须保持 | Phase 7/8 必须复现 | OPEN-EMULATE |
| D-012 | 数据库 id 474/475 为全 0 空行，会进入候选池（有效值视为 1，显示值为 0） | 未知牌候选质量 | 复现；是否过滤由用户决定 | OPEN |
| D-013 | 连携只有一层（连携翻得的卡不再继续连携），与常见的游戏实现可能不同 | 规则语义 | Python 为准，复现；若要改动需用户确认 | OPEN-EMULATE |
| D-014 | ID 在深度超过剩余空格后仍重复搜索；"预测下一层耗时" 的代码从不生效 | 只浪费时间，固定深度结果不受影响 | ReferenceCompat 复现；**Fast 模式已在整棵博弈树搜完（depth ≥ 空格数）或根方无合法走法后停止（Phase 4；每一层的分数与精确值位级相等，已验证）** | OPEN-EMULATE（compat；Fast 已处理） |
| D-015 | `ai_move` 里的 robust 选择对 **走子方手牌顺序** 敏感（并列时稳定排序取先出现者），而 `base_state` 的手牌顺序是标准 minimax 搜索 **之后** 的顺序（`Player.hand` 与 `opp_hand` 是同一个列表，搜索的 `undo_move` 把牌追加到末尾，D-001）。实测：反转走子方手牌顺序，选中的走法在 532 次调用中有 10 次（≈1.9%，普通与激进模式各 10 次）改变 | Phase 7 想与 Python 的最终走法一致，必须复现搜索后的手牌顺序 | Phase 5 的 parity 以 Python 导出的状态（含手牌顺序）为输入；Phase 7：用 ReferenceCompat 做标准搜索并把搜索后的状态交给 robust 选择，或规定规范手牌顺序（需用户决定） | OPEN-EMULATE（Phase 7 处理）→ **用户决定（2026-10-09）：Phase 7 的标准搜索用 Fast，robust 选择前把手牌恢复为请求的原始顺序（规范顺序）**；因此最终走法只在完全同分时可能与 Python 不同，端到端比较按 Q-5 (b) 比较分数与并列集合（Phase 7 实施）→ **Phase 7 已实现（2026-10-10）**：引擎请求在任何 Python 搜索之前按请求手牌顺序构造，Fast 标准搜索与 robust 层分别在两份状态副本上运行（robust 层看到请求顺序的手牌）；404 条端到端记录中 209 个 robust 候选表与 Python（在请求顺序状态上运行）逐位相同。**M4-3 由此关闭**：搜索后的手牌顺序与显示边不再是引擎路径的输入，robust 层只读有效值 |
| D-016 | `_evaluate_endgame_move_robustly` 的"非精确"分支（一步回应 + 角落提前返回）在生产中不可达：进入 robust 模式要求占用 ≥ 5，`use_exact_solver` 与 corner 风险读同一块棋盘（占用 ≥ 5 才非零），所以"非精确且 corner_risk ≥ 5"永远为假 | 无（死代码） | C# 忠实复现并以 PY-QUIRK 标注；golden 覆盖非精确路径并断言该路径 corner 风险恒为 0 | OPEN-EMULATE |
| D-017 | `_copy_state_with_unknown_assignment` 的替换牌 `type_modifier` = 0 且不重算同类修正，含同类规则时场景里的未知牌没有强化/弱化，直到有类型的牌被打出触发重算 | 场景内的精确值 | 复现；golden 逐条验证 | OPEN-EMULATE |
| D-018 | `select_endgame_robust_move` 在激进模式总是、非激进模式在没有安全走法时 **原地排序** `scored_moves`（别名副作用），返回列表的顺序依分支而异 | 只影响返回列表的顺序 | 复现；golden 逐条比较顺序 | OPEN-EMULATE |
| D-019 | 机器人层（`_best_immediate_reply_score`、`_evaluate_endgame_move_robustly`）把每个走法的牌按 **card_id 取手牌里的第一张**（`next(c for c in hand if c.card_id == card.card_id)`），而不是按身份。手牌里有重复 id（D-004）时，所有重复都映射到该 id 的第一张卡；对真实数据无害，因为同 id = 同一张卡的定义，但对 None 占位牌（四边可以不同）会改变被评估的走法集合 | 只在手牌内有重复 id 时；golden 有 71 条 robust 记录（走子方手牌 50 条、场景手牌 71 条）与 187 条 reply 记录覆盖 | C# 复现：在入口处（Make/Undo 重排手牌之前）按手牌顺序解析"第一张同 id"（Python 每次都从 `copy()` 开始，看不到 D-001 的重排），变异检查：去掉映射 → reply 测试在记录 #5 失败；改成每次迭代重新解析 → robust 测试在记录 #17 失败 | OPEN-EMULATE |
| D-020 | **8A, 非精确（已记录）**：请求体不是合法 JSON / Content-Type 不是 JSON 时，Python（Flask）在 `ai_move` 的 `except Exception` 里返回 HTTP 500 加 Flask 的 BadRequest 文本；C# 返回 HTTP 400 `{"error":"Invalid input data"}`，并且不检查 Content-Type（任何类型的 JSON 体都接受） | 只影响畸形请求；插件不会发送 | 保持 C# 行为（更正确）；已写入 `API.md` | DOCUMENTED-DIFFERENCE |
| D-021 | **8A, 非精确（已记录）**：请求字段类型异常（边值是字符串、`pos` 不是数对、`numU` 是小数……）时 Python 抛出各种 Python 异常，错误文本不全部复现。**逐字复现并有 golden 测试**：缺键（`'numU'`、`'pos'`、`'owner'`）、卡牌不在数据库（`Board card not found in database: U.. R.. D.. L..` / `Hand card ...`）、`list index out of range`、选拔规则下棋盘上星级为 0 的行（id 474/475）触发的 `KeyError: 0`（错误文本 `0`）。整数值的小数（`6.0`）与布尔按 Python 的相等/哈希语义当作整数；`npcIds` 的元素按 `int()` 的读法（空白、符号、下划线、小数截断）；超过 2^63 的 id 被跳过（Python 保留大整数，随后引擎拒绝） | 只影响畸形输入 | 复现常见错误，其余给出 Python 风格的文本 | DOCUMENTED-DIFFERENCE |
| D-022 | **8A**：Python 的未知牌采样使用进程全局的 `random`（所有线程共用一个流）；C# 每个请求一个 `PythonRandom`，种子是新的 64 位密码学随机数并写入对局日志。相同种子下逐位一致（5,400 个采样用例、2,684 个解析请求、960 个端到端请求已验证），但生产环境中请求不再是同一条长随机流中的片段 | 设计如此（A-23）：可重放、线程无关 | 测试用 `TestSeedHeader`（仅配置开启）按 `random.seed(n)` 播种 | EXACT-PER-SEED |
| D-023 | **8A**：`ai_move` 在回答前做 `new_state = game_state.copy(); new_state.play_move(row, col, card)`（给控制台画预测棋盘），传入的是**原来的** `card` 对象；在同类强化/弱化下这会重算修正值并**改写原卡的 `type_modifier`**，而响应里的 `card` 描述（`(兽人+2)`）读的正是这个值。端到端比较的第一个差异即由此发现 | 影响 `card` 字段文案 | C# 复现（`LGameState.Copy` + `PlayMove`，`PY-QUIRK`） | OPEN-EMULATE |
| D-024 | **8A**：旧版分析文案（`calculate_win_probability`、`generate_move_recommendation`、`count_captured_cards`）对同一个 `game_state` 做 make/undo，使手牌顺序改变（D-001）、显示值陈旧（D-003）、手牌里的原卡被复制品取代（D-010）；这些会影响备选走法的选取与文案 | 影响 `recommendation`/`card` 字段 | C# 用 Python 对象模型的忠实移植（`TTC.Siren.Compat.Legacy*`，只在请求期运行一次，不在搜索热路径）按 Python 的调用顺序复现；D-010 由此决定为"复现" | OPEN-EMULATE |
| D-025 | **8A**：`parse_hand` 在没有任何特殊规则（`rules` 为空）时绕过未知牌处理器，用 `random.sample(全库, ...)` 兜底；生成的牌保留真实 id、没有"生成"标记，因此后续逻辑把它们当作已知牌（`analyze_opponent_hand` 给它们置信度 1.0 "已知卡牌"，残局层不把它们当未知槽位）；另外 `_sample_strategic_same_number_cards` 等函数里"排除已选卡牌"的过滤比较的是 1000+id 与真实 id，永远不过滤任何牌（已选牌可再次被抽到） | 采样质量；日后改进采样时的基线 | C# 逐字复现（`PY-QUIRK`），golden 覆盖；**不改进**（8A 不授权） | OPEN-EMULATE |

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
8. 输入校验：同一手牌内 card_id 重复（D-004）时抛出明确的异常（parity 工具会排除这类输入）；手牌数量超过容量也抛异常。**【2026-10-10 更正：重复 id 的拒绝已移除（D-004 FIXED-BOTH，Q-2），C# 现在接受重复 id；手牌容量检查仍在】**
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
6. D-004：`FromSnapshot` 发现同一手牌内 `card_id` 重复（包括两个 `None`）时抛 `InvalidGameStateException`；golden 生成器保证手牌内 id 唯一（最多一张 `None` id）。**（已被 Q-2 取代：D-004 在两边修复，C# 现在接受重复 id，见 §7 "D-004 修复记录"。）**
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

**状态：DONE（2026-10-08）。** 用户（2026-10-08）授权 Phase 5，并要求先解决 M4-1 与 M4-2（Phase 4 审查的两个非阻塞项，因为 Phase 5 的 golden 依赖它们），不扩大为 Phase 4 的整体清理；Q-4、Q-6 保持默认（见 §9）。

#### 前置：M4-1 与 M4-2（在生成任何 Phase 5 golden 之前完成）

**M4-2（golden 绑定参考源码）**
- 新增 `tools/parity/golden_sources.py`：每个 golden 文件的 header 写入 `python_version`（major.minor）与 `source_sha256`（这份 golden 依赖的每个文件的 SHA-256：`core/*.py`、`csv_compat.py`、`data/幻卡数据库.csv`、生成器本身与 `golden_sources.py`；search 另加 `ai/ai.py`；endgame 另加 `ai_server.py`、`ai/unknown_card_handler.py`、`config/unknown_card_config.py`）。`py_rules_trace.py`（`/3`）与 `py_search_trace.py`（`/2`）的 header 与 `carddb.json` 都带上这些字段（只改了生成器脚本，没有改任何 Python 业务代码；生成的记录内容不变）。
- C#：`GoldenSources.cs`（文件集与 golden_sources.py 一一对应）重新计算工作区文件的 SHA-256 与 Python 版本；`GoldenFiles.Ensure` 发现 header 缺少绑定、文件集不同、任一哈希不同或 Python 版本不同就 **删除旧文件并重新生成**，所以 golden 不可能比它生成时的 Python 源码活得更久（Python 语义改动 → 若 C# 没跟上，会表现为 parity 不一致，而不是静默通过旧数据）。文件名与生成器版本升级（`g3`/`g2`），首次运行把所有 golden 重新生成了一遍（约 5 分钟），全部测试在新数据上通过。
- 测试 `GoldenBindingTests`（3 个）：在临时复制的源码树上，匹配的 header 通过；对每个绑定文件改一个字节都会被检测并点名；缺文件、多文件、缺 `python_version`/`source_sha256`、Python 版本不同都判为过期；篡改后的 `carddb.json` 被判过期；真实生成的 rules/carddb/search golden 的绑定与工作区一致（生成器与 C# 的文件集一致）。

**M4-1（哈希充分性，独立于哈希自己的成分列表）**
- 新增 `ZobristSufficiencyTests`（2 个）。它们遍历完整的博弈树（Make/Undo，不经过任何搜索代码），在一个 `GameState` 内要求：(1) 哈希相同 ⇒ `ToSnapshot()` 规范化状态相同（只允许手牌顺序 D-001 与显示边 D-003 不同）：841 个 golden 局面、2,490,856 个节点、1,152,410 次换位（transposition）访问，0 个不一致；(2) 哈希相同 ⇒ 全宽暴力 minimax 的精确值位级相等（两种评估模式）：706 个局面（其中 531 个含同类规则）、605,264 个节点、162,924 次换位访问的值被比较，0 个不一致。
- 变异检查：把 owner 从哈希里去掉（增量与重算 **同时** 去掉，使 Phase 4 的"镜像式"测试仍然自洽）→ 新测试失败（值不同、快照不同），旧的 `Incremental_hash_equals_from_scratch…` 也失败。另一个观察：type_modifier 在哈希里是 **冗余的**——一个局面里修正值由棋盘决定（审查 §12 也这么说，"保险"）——所以去掉它是等价变异，没有任何测试能（也不应该）检测到。
- 当 `GameState` 新增字段时：先把它加进快照（这样 (1) 能覆盖），保留 (2)。

#### 验收结果（全部满足）
- [x] `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`：0 warning / 0 error。
- [x] `dotnet test`：**78 通过 / 0 失败 / 0 跳过**（H4-1 整改后的 62 + M4-2 的 3 + M4-1 的 2 + Phase 5 的 11）；`python -m pytest -q`：29 passed；已跟踪的 `.py` 里只有两个 parity **生成器** 脚本被改（M4-2 的 header），没有任何 Python 业务代码被改动。
- [x] **exact 值位级相等（整数）**：700 条 golden（1,400 个根值，5,376 个"第一步之后"的值，0–4 个空格，根值覆盖 −9..9）与 Python 的 `_solve_endgame_exact` 完全一致；另有独立的无 memo 暴力 minimax 交叉验证 C# 的带 memo 求解器（3 个基准局面 + 250 个 golden 局面，两个 AI 方），一致。
- [x] **Python 自己的 memo 是可靠的**（"memo-key 语义"）：Python 在整个 `_evaluate_endgame_move_robustly` 里共享一个 cache；golden 中 5,376 个第一步值的"共享 cache"与"新 cache"结果 **0 个不同**（生成器额外在 400 条独立随机记录上做过同样的检查：4,198 个值，0 个不同）。所以 memo key 是实现细节，C# 用 Zobrist memo 得到同样的值。
- [x] **场景集合与顺序**：500 条 golden（0–3 张未知牌，22 个无未知牌；164 条限制 NPC 卡池；369 条触达场景数上限；6,936 个场景）：合法候选 id 列表（数据库顺序）、60,960 个候选的威胁分数（`_score_unknown_candidate`，位级）、全部场景状态（规范化字符串，含手牌顺序与显示边）与 Python 逐个相同；`_unique_unknown_candidates` 800 条（去掉 2,310 个重复）顺序与内容相同。
- [x] **每个走法的 (final, safety, robust, corner) 位级相等 + 最终选择**：400 条 golden × 2 种模式 = 800 次 `select_endgame_robust_move`，共 8,760 组走法分数（578 次精确求解路径、222 次启发式一步回应路径；普通模式有"完全安全走法"的 238 次、没有的 158 次；0/1/2/3+ 张未知牌：37/199/126/38；每条 1..24 个场景），**返回列表的顺序（Python 在某些分支原地排序 `scored_moves`）与选中的走法都与 Python 逐条相同**；分别用 Python 构造的场景与 C# 构造的场景各跑一遍，结果相同；`Select` 不改动 base 状态。
- [x] `_calculate_corner_safety_risk`：1,500 条 golden、6,727 个 (牌, 格) 检查（3,722 个非零，占用 0..8），位级相等。`_best_immediate_reply_score`：1,500 条 × 2 个 AI 方，位级相等。
- [x] **覆盖逻辑**：`ai_move` 里的覆盖块是内联在 Flask 处理函数里的，不能 import；golden 生成器用 AST 从 `ai_server.py` 的源码里 **提取这一段并直接执行**（找不到或不唯一就报错；源文件哈希已绑定到 golden），6,000 条合成候选表（含阈值边界值、标准走法/robust 走法不在表里、`None`）：C# `DecideOverride` 与它逐条一致（916 次覆盖，其中激进 65 次；3,904 次保持；1,180 次没有标准走法；458 次走法不在表内）。
- [x] **未知牌场景的确定性部分与 Python 的随机部分的边界**：Python 在"合法候选为空"时会改用 `handler.generate_opponent_cards`（随机）。C# 不复现（A-7）：`ScenarioBuild.RequiresFallback = true`，调用方（Phase 7）可传入这些牌（`fallbackCandidates`）。golden 不含这种记录（生成器跳过）。
- [x] 热路径零分配：精确求解器在 3 个局面 × 2 个 AI 方上共 8,684,790 个节点 `GC.GetAllocatedBytesForCurrentThread()` 增量 **0 字节**（`Bench exact` 每个局面 alloc 0 B）；Phase 3/4 的搜索分配测试仍为 0 字节。
- [x] **设计约束**：精确评分有自己的 `EvalMode.ExactCardDifference`；记忆表按 (GameState, AI 方, 模式) 绑定，场景 `GameState` 之间从不共享一代表（`ExactEndgameSolver.Value` 在未绑定、AI 方不同、状态不同、评估模式不同时抛 `InvalidOperationException`，测试覆盖，并验证启发式搜索与精确求解不能互相复用同一代表）；终局值是 AI 视角的整数牌数差（不是 `EvaluateState`）。
- [x] **变异检查**（9 处，每次改一处，对应的 Python-parity 测试必须失败，之后已还原）：corner 系数 2.2→2.1、`SafetyMargin` 0.75→0.7、robust 权重 0.45/0.55 对调、覆盖阈值 +0.05→+0.06、排序并列时 id 符号、加算组权重 8→6、候选上限差一、精确求解"无可走"局面计 0、去重保留最后一个而不是第一个：**9/9 被检测**（失败的测试见各自的 parity 测试）。
- [x] benchmark 报告：`docs/reports/PHASE5_BENCHMARK.md`，数字见 §10。

#### Phase 5 交付记录（2026-10-08）

**新增 / 修改的文件**（没有修改任何 Python 业务代码）
- `csharp/src/TTC.Siren.Search/`：`ExactEndgameSolver.cs`（精确求解 + 绑定上下文的 memo）、`EndgameScenarios.cs`（合法候选、去重、威胁分数、排序、排列与截断、场景状态）、`EndgameRobust.cs`（corner 风险、一步回应分数、robust 选择、`ShouldUse`、覆盖决策）；`SearchContext.cs` 新增 `EvalMode.ExactCardDifference`。
- `csharp/tests/TTC.Siren.Tests/`：`EndgameDifferentialTests.cs`（8 个 Python-parity 测试：场景、unique、corner、reply、exact、robust×2、override）、`ExactEndgameSolverTests.cs`（3 个：独立暴力 minimax、上下文规则、零分配）、`ZobristSufficiencyTests.cs`（2）、`GoldenBindingTests.cs`（3）、`GoldenSources.cs`；`GoldenFiles.cs` 增加绑定检查与 endgame golden。
- `tools/parity/golden_sources.py`、`tools/parity/py_endgame_trace.py`（新）；`py_rules_trace.py`/`py_search_trace.py`（header 绑定）。
- `tools/bench/py_endgame_bench.py`（新）、`csharp/src/TTC.Siren.Bench/Program.cs`（新增 `exact`、`endgame`、`server` 模式）、`docs/reports/PHASE5_BENCHMARK.md`。

**设计决定**
1. **精确求解器的 memo = Zobrist 键的 `FastTranspositionTable`，上下文 (GameState, AI 方, `ExactCardDifference`)**，每个场景 `GameState` 一个 memo 代；同一个场景状态的所有第一步共享该 memo（相同状态、AI 方、模式，所以是有效复用，也与 Python "一个 cache 跨一个走法的所有场景"的目的一致；值是局面的纯函数，所以 Python 的结构化键与这里的哈希键不必相同）。没有走法的局面（秩序/混乱把手牌全部过滤掉）按当前局面计分（Python 无"过"规则）。
2. **`EndgameRobust.Select` 对场景先克隆、对每个 (场景, 走法) 做 Make/Undo**，base 状态与场景不被改动；循环顺序是场景在外、走法在内，但每个走法的聚合仍然按 **场景顺序** 从左到右累加（`sum(list)/len(list)`，CPython 3.11 的 `sum()` 是普通累加；3.12 起改为补偿求和，所以 golden 绑定 Python 版本；精确路径是整数求和，不受影响）。整数路径的平均值是 `(double)sum / count`（int/int 真除法）。
3. **排序**：Python `sorted(reverse=True)` 是稳定的；C# 用稳定插入排序，键为 (safety, final, robust, −corner)（激进模式 (final, robust, safety, −corner)）；`scored_moves` 在激进模式、以及非激进模式 `safe_moves` 为空时被 **原地排序**（Python 的别名副作用），返回顺序按此复现并由 golden 逐条比较。
4. **场景构造**：Python 的 `_is_generated` / `_is_prediction` 属性与 id ≥ 1000 是未知牌的标记，C# `GameState` 没有这些属性，所以调用方传入 Python 的 `_is_generated_unknown_card` 对每个手牌槽的结果（`bool[2][]`）；场景由 `ToSnapshot()` 替换未知槽的卡后 `FromSnapshot` 得到（替换牌：基础边、无同类修正、显示边 = clamp(base)，owner 与 can_use 取自被替换的牌；与 Python `Card.copy()` 一致）。无未知牌、或没有任何排列合法时返回 base 状态本身（同一个对象），与 Python 一致。
5. **覆盖逻辑**通过 AST 提取执行 Python 源码里的原块（而不是在生成器里再抄一遍公式），golden 因此绑定 `ai_server.py` 的内容。
6. **Q-4**：保持，id 474/475 全 0 空行留在候选池里；golden 里的合法候选 id 列表（含 474/475）与 Python 逐项相同。**Q-6**：基线只用 CPython 3.11.4（本机没有 PyPy）。**Q-5 (b)** 对 Phase 5 的影响：robust 选择是 Python 的确定性移植，所以 **字面走法逐条相等**（不需要放宽到并列集合）；Fast/compat 搜索产生的"标准走法"来自哪里仍遵循 Q-5 (b)。
7. golden 记录的规模为让 Python 在可接受时间内生成：精确/robust 记录的占用格数 ≥ 5（≤4 个空格）；占用 5 且有未知牌的 robust 记录用 2–4 个场景（Python 在更大的手牌上一次精确求解要好几秒）；未知槽 ≤ 3（随机牌工厂的合成 id ≥ 1000 会被 Python 当成"未知"，生成器过滤掉超过 3 个的位置，否则 Python 要排列几千万个组合）。这些都是生成器的选择，不是被测代码的限制。

**新发现的差异 / 观察**（均已写入 §4；没有无法解释的差异，没有新增 BLOCKED）
- **D-015**（OPEN-EMULATE，Phase 7 必须处理）：robust 选择对 **走子方手牌顺序** 敏感（并列时稳定排序取先出现的）。而 `ai_move` 里的 `base_state` 手牌顺序是 **标准 minimax 搜索之后** 的顺序（`Player.hand` 与 `opp_hand` 是同一个列表，搜索的 `undo_move` 把牌追加到末尾，D-001）。实测：把走子方手牌顺序反转，选中的走法在 532 次调用中有 10 次（≈1.9%，普通与激进模式各 10 次）发生变化。所以 Phase 7 想与 Python 的最终走法一致，必须用 ReferenceCompat（复现 D-001）做标准搜索并把 **搜索之后的状态** 交给 robust 选择，或明确规定一个规范手牌顺序；Fast 的遍历不同，会得到不同的手牌顺序。Phase 5 的 parity 以 Python 导出的状态（含手牌顺序）为输入，不受此影响。
- **D-016**（观察，不影响结果）：`_evaluate_endgame_move_robustly` 的"非精确"分支（一步回应 + 角落提前返回）在生产中不可达：进入 robust 模式要求占用 ≥ 5，而 `use_exact_solver` 与 corner 风险都读取同一块棋盘（`occupancy ≥ 5`）；角落提前返回需要"非精确且 corner_risk ≥ 5"，而 corner 风险在占用 < 5 时恒为 0，所以这个提前返回 **永远不会执行**。C# 为忠实复现保留了该分支，并有 PY-QUIRK 注释；golden 覆盖了非精确路径（222 次选择），并断言该路径下 corner 风险恒为 0。
- **D-017**（OPEN-EMULATE，复现）：`_copy_state_with_unknown_assignment` 把替换牌的 `type_modifier` 设为 0 且 **不重算** 同类修正，所以场景里的未知牌在含同类规则时没有强化/弱化，直到某张有类型的牌被打出触发重算（§1.9 已记录，现已有 golden 逐条验证）。
- **D-018**（观察）：`select_endgame_robust_move` 在激进模式下总是原地排序 `scored_moves`，非激进模式只在没有安全走法时才这样（别名副作用）；返回的列表顺序因此依分支而不同，已复现。
- 观察：Python 3.12 把 float `sum()` 改为补偿求和；只有在非精确（生产不可达）路径上才会因此改变数值，golden 绑定 3.11（M4-2）。
- 观察：Python 的 memo 对共享 cache 是可靠的（见上，0 个不一致）。

**已知限制**
- 随机回退（合法候选为空时 Python 用 `handler.generate_opponent_cards`）不在 C# 内复现（A-7）；`RequiresFallback` + `fallbackCandidates` 接口留给 Phase 7。
- 请求解析（`parse_hand` 的随机采样、`parse_board` 等）、`ai_move` 的调度（含"标准走法"的来源）与文案不在 Phase 5 范围；覆盖逻辑只移植了决策函数。
- Python 一侧的耗时只有 CPython；没有 PyPy 基线（Q-6 默认）。
- M4-3（搜索后的手牌顺序/显示边与 Python 对比；golden 偏向后期局面）、L4-*、O4-* 与 Phase 3 遗留项 **未处理**（Phase 5 不依赖它们；D-015 说明了与 M4-3 的关系：手牌顺序对 robust 选择有影响，Phase 7 需要处理）。
- Phase 5 的改动没有经过独立审查。

#### Benchmark review remediation（2026-10-08，`docs/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md`，结论 `BENCHMARK_VALID_WITH_CAVEATS`；目标 `.claude/sonnet_review_fix_goal.md`）

正确性没有问题（节点数、分数位级、走法、72/72 选择全部不变），但 benchmark 数字有偏差，已更正。只改了 benchmark 工具与文档，**没有改动** `TTC.Siren.Core` / `TTC.Siren.Search` / 任何测试断言 / Python 业务代码 / parity 生成器，也没有做任何性能优化。
- **A-1（分层 JIT）**：`Bench` 现在默认 warm（整组项目预热 1500 ms 后才计时），`--cold` 保留旧行为，所有表标注 warm/cold 并给出 min/median/max；`fast` 轮换三个变体的测量顺序；计时区域只含搜索调用。22 项固定深度合计：warm 0.0225–0.0233 s，cold 0.051–0.053 s。"49.7×" 是冷 JIT 数字（cold 53–63×，warm 119×）。
- **A-2（263× 撤回）**：那次 Python 计时 15.58 s 被后台 golden 生成污染；交替重测 Python 10.24–10.67 s，C# 0.0622–0.0651 s，**161–167×**，其中 1.235× 是 memo 键（工作量 297,935 → 241,264），每次 make_move 约 131×。与评审的 9.95–10.21 s 相差 1–5%。
- **A-3/L5-3**：新增 `tools/bench/run_interleaved.py`（Python/C# 交替，≥ 3 轮）；`py_bench.py` 与 `py_endgame_bench.py` 打印 min/median/max。**A-4**：逐项以 µs 打印，< 50 µs 标 `*`，每项都有 NPS。**A-6**：`fast` 轮换顺序（Fast/TT 关 仍为 1.49×）。
- **工作量对比**：`tools/bench/count_robust_work.py` 与 `Bench endgame-work`：robust 流水线 Python 418,990 对 C# 389,990 次 make_move（1.074×）。**L5-6**：`csharp/.gitignore` 加入 `.vs/`。
- 详细数字、可信范围与全部声明见 `docs/reports/PHASE5_BENCHMARK.md` 的 "Corrections after the Opus benchmark review" 一节与 §10。Phase 6 未开始。

- **A-7（CPU 核心放置，2026-10-08，评审 §12.4）**：混合 CPU 上 P/E 核放置会让同一运行相差 1.3–2×，已加 `tools/bench/pin.py` 与 `run_interleaved.py --affinity`（两边都固定在同一 CPU 集，默认检测 P 核集，本机 `0xFFF`），harness 表头打印实际亲和性；可信比值现在都是"两边固定在 P 核"的值（fixed warm 116–118×、exact 161–162×、robust warm 169–175×），记录见 §10 与报告的 "A-7" 一节。**暂定，待用户确认（用户对"中位数还是范围"只回复了"继续"，没有给出选择；助手按最保守的方式处理，这不等于用户批准了某个数字，可一句话改动）**：fixed cold 的 11 轮比值 57.5–91.0×（中位数 70.1×），个别轮超出评审的 55–73× 范围 15% 以上；口径是**同时写出完整范围与中位数**——"57–91×（中位数 70×）"——而不是单个数字。Phase 6 未开始。

#### D-004 修复记录（Q-2，2026-10-09，用户批准；`.claude/phase6_goal.md` 步骤 B）

- **Python**（唯一改动的 Python 业务文件 `core/game_state.py`，+8/−2）：`make_move` 先 `next((c for c in hand if c is card), None)`，找不到才退回旧的按 `card_id` 匹配。所有现有调用者的行为不变（搜索传入的是手牌里的对象；robust 层传入手牌里的"第一张同 id"，身份能找到；`count_captured_cards` 传入 `card.copy()`，走回退）。`undo_move` 未改（D-001、D-003 保持）。
- **C#**：`GameState.FromSnapshot` 去掉重复 id 的拒绝并更新 PY-QUIRK 注释；`MakeMove`/`UndoMove`、哈希（按实例）、`Move(Inst, Cell)` 无需改动。检查过：`HistoryTable`（按 id，重复共享条目，与 Python 相同）、`PythonStateKey`（id 拼接，与 Python 相同）、Zobrist（按实例）、残局代码。**D-019**：`EndgameRobust.BestImmediateReplyScore` 与 `Select` 现在在入口处（任何 Make/Undo 之前）把每个走法映射到手牌里"该 id 的第一张"，与 Python 的 `next(c for c in hand if c.card_id == …)`（每次从 `copy()` 开始）一致。
- **测试**：Python `tests/test_duplicate_hand_ids.py`（5 个）——**在旧代码上 3 个失败**（第二张同 id 卡被放置和移除、两张 None 按身份区分、300 个固定种子的随机 make/undo 往返不丢卡不重复），另外 2 个守住不变的行为（`copy()` 回退、不在手牌且 id 不匹配时返回 `None`），新代码上 5 个全过；`pytest` 现为 **34 passed**（29 + 5）。C#：`D004_duplicate_hand_ids_are_rejected` 被 `Duplicate_hand_ids_are_supported` 取代（**理由**：旧测试断言的正是 Q-2 批准要改变的行为；CLAUDE.md §7 允许断言本身被证明过时时替换并记录原因）——新测试覆盖状态往返、按实例 make/undo（第二张同 id 卡落到棋盘、剩下的是 1-2-3-4 那张、undo 追加到末尾 D-001）、200 条随机游走中增量 Zobrist 哈希 = 重算值且完全撤销后哈希复原、depth 1–4 两种模式下 Fast 根分数与 oracle 位级相等、ReferenceCompat 在克隆上节点数与分数确定。其他测试没有跳过、删除或放宽。
- **golden**：生成器版本升级（`py_rules_trace/4`、`py_search_trace/3`、`py_endgame_trace/2`），源码哈希绑定（M4-2）使旧 golden 自动作废并重新生成。重复 id 用独立的随机流加入（不改变其余记录的内容）：规则轨迹每 8 条中 1 条，搜索 eval/search/id/tt 同样，endgame 的 robust 每 4 条 1 条（交替放在走子方手牌和对手已知牌里）、reply 与 corner 每 8 条 1 条；重复形式 = 同 id 同边的复制品，或两张 id 为 None 的占位牌（边不同）。C# 差异测试统计并断言覆盖：规则轨迹 **2,500 条序列（12.5%）、23,100 步、3,314 次对"手牌里 id 重复的卡"的 make**；eval **12,500 / 100,000** 个局面；固定深度搜索 **112 个局面、557 次运行**；ID 25 个局面；robust **71 条记录**（走子方手牌 50、场景手牌 71，各 2 种模式 = 142 次选择，Python 场景与 C# 场景各跑一遍）；reply 与 corner 各 187 / 1,500 条。全部 78 个 C# 测试在重新生成的 golden 上通过（0 失败、0 跳过），0 个不一致。
- **变异检查**（改一处，对应测试必须失败，之后已还原）：去掉 `BestImmediateReplyScore` 的入口映射 → reply 测试失败；`Select` 改为每次迭代重新解析"第一张同 id" → 两个 robust 测试失败（记录 #17，返回顺序不同）。
- **D-004 → FIXED-BOTH，新增 D-019（OPEN-EMULATE）**。已知限制：exact 记录和 scenario 记录没有加入重复 id（Python 的 exact 生成器本来就按 id 取第一张，C# exact 测试按槽位走，两者对"边不同的同 id 卡"并不等价；真实数据里同 id 的卡定义相同，不受影响）。

### Phase 6 — Profiler 驱动优化

范围：用 dotnet-trace / BenchmarkDotNet 找热点，每次改动都走 CLAUDE.md §6 的流程；可能的方向（必须先测量）：增量评估、ordering 计算缓存、TT 大小与布局、残局 α-β（值必须保持精确）。单线程完成后，经用户批准才做 root parallelism。
验收：每次改动都有 Baseline/After 数据，parity 全部保持。

#### Phase 6 交付记录（2026-10-10，`.claude/phase6_goal.md` 步骤 C；详细数据 `docs/reports/PHASE6_OPTIMIZATION.md`）

**状态：DONE，等待独立审查**（审查门禁由审查者设置，整改者不设）。**审查门禁（2026-10-10）：`PASS_WITH_NONBLOCKING_FINDINGS`**，见 `docs/reviews/OPUS_PHASE6_REVIEW.md`。单线程；没有 `unsafe`、SIMD、多线程；没有 root parallelism；没有改变任何规则/搜索语义。所有计时固定在 P 核 `0xFFF`（`run_cs_pinned.py` / `run_interleaved.py --affinity auto`），B、C 的提交只在本地，未推送。
- **工具**：`tools/bench/run_cs_pinned.py`（C# 单独、固定亲和性、多次进程运行 min/median/max、检查确定性输出的 SIG 签名在每次运行与基线之间完全相同、`--exe` 做同窗口 A/B）、`tools/bench/profile_summary.py`（`dotnet-trace` 固定核采集并汇总 Speedscope）、`Bench` 的 `SIG` 行、make/undo 的 `RESULT` 行与 `loop fast|compat|exact`。
- **profile（优化前）**：Fast——`MakeMove` 44%（`RecalculateTypeModifiers` 29%）、`EvaluateState` 31%；compat——`OrderMoves` 57%；exact——`RecalculateTypeModifiers` 59%（复现评审 §8）。
- **保留的改动（各为一个本地提交）**：C1 `3a77dbe` 一次遍历实例表重算同类修正（同类强化 exact −33%、Fast −10%、含同类规则的 make/undo −43..−48%）；C3 `6b61630`（回退后以 `290b15e` 重新应用）查表式 `PlacementEdgeScore`（compat −20%、Fast −19%，位级相同，因为所有项都是 0.5 的倍数）；C4a `2c226bb` `EvaluateMove` 走邻接表（compat −6%）；C7 `782a3b5` 每线程复用 exact memo（robust 流水线 −21%，评审 L5-2）。
- **回退并记录的候选**：C2 Span 拷贝快照、C5a `EvaluateState` 单元格权重表、C6 缩小 `stackalloc`、C8a 修正值未变时跳过哈希、C9 切片刷新显示边（中位数 −2.8%，未到 3%）。
- **同窗口 A/B（原始 vs 最终，warm，7 次）**：compat −24.6%、Fast −28.9%（TT 开/关 −22.8/−23.9%）、exact mid3 −38.0%、robust −28.0%；不含同类规则的 make/undo 不变。**与 Python 交替（`run_interleaved.py --affinity auto`，两边 `0xFFF`，3 轮，最终提交 `427d67f` 上重新测量，warm）**：fixed **156.0–165.7×**（基线 114–119×）、exact **249–263×**（167–171×，其中 1.235× 是 memo 键）、robust **236–255×**（181–186×，其中 1.074× 算法性）；cold：fixed 62.9–66.6×（基线 61.8–80.2×，cold 双峰，范围重叠）、exact 242–259×、robust 141–150×（基线 115–121×）。
- **不变量**：78/78 测试（无断言被放宽），pytest 34 passed；ReferenceCompat 的 SIG（节点数、分数位、走法）、Fast 的 41 行（节点数、分数位、走法，40/40 `same`）、exact 值与 `make_move` 数（1/−2/−1；217/4,707/241,264）、72/72 选择、make/undo 校验和与原始构建**逐字相同**；热路径分配仍为 0。
- **C3 的 cold 保留判断（整改记录）**：第一次块顺序比较显示 C3 使 cold 22 项 compat 约 +12%，我曾错误地保留并标为偏离规则；之后回退（`17c07c7`）并按**目标文件的协议**（`run_cs_pinned.py --cold --runs 7`，中位数，交替先后顺序，4 次重复，`docs/reports/phase6/c3_cold_protocol.txt`）重测：候选相对前一提交为 −7.6%、+4.4%、+7.7%、+1.8%（四个中位数的中位数 +2.2%），符号随重复而变，前一提交自身的中位数波动 16%，所以 >3% 的回退**不可重复**，满足保留规则；补充证据（非判断依据）：交替构建逐样本测量（`ab_interleaved.py`）原始 0.0431、含 C3 0.0423、不含 C3 0.0500 s。C3 已重新应用为 `290b15e`。详见 `PHASE6_OPTIMIZATION.md` §5。
- **仍存在的热点**：同类修正维护（exact 42%、Fast 18%，需要能精确复现 D-017/D-003 的结构性方案）、`EvaluateState` 的 9 次落点评分（Fast 29%）、compat 的 `EvaluateMove`（39%）、`ResolveFlips`（11–16%）。

### Phase 7 — 集成策略

建议方案：C# 引擎做成常驻子进程（stdin/stdout JSON 行协议）或 localhost HTTP；Python `ai_server` 仍然负责请求解析和未知牌采样，然后把完整确定的状态交给 C#（minimax + robust 残局），出错或超时时回退到 Python 求解器（受开关控制）。
验收：录制的真实 `/ai_move` 请求（固定 Python 随机种子）端到端结果一致；延迟报告；回退路径有测试。
**用户决定（2026-10-10，A-8、Q-7、D-015）**：本机 HTTP 服务（只监听 127.0.0.1，ASP.NET Core，依赖系统 .NET 10 运行时，引擎内工作线程池做请求级并行，不做 root parallelism）；默认开启、出错/超时/忙/协议或卡库不匹配/非法走法时自动回退 Python；引擎用 Fast 搜索 + 请求原始手牌顺序（D-015）；搜索参数沿用 Python（10 层 / 10 s）；部署 Windows Server。

#### Phase 7 交付记录（2026-10-10，`.claude/phase7_goal.md`；详细数据 `docs/reports/PHASE7_INTEGRATION.md`）

**状态：DONE，等待独立审查**（审查门禁由审查者设置，实现者不设）。**审查门禁（2026-10-10）：`PASS_WITH_NONBLOCKING_FINDINGS`**，见 `docs/reviews/OPUS_PHASE7_REVIEW.md`（M7-1 过载级联回退、M7-2 每请求新建连接、M7-3 e2e golden 经测试框架重新生成会死锁；上线检查清单见该文 §9）。Step A（Phase 6 收尾 + 推送）：`d719a59`（审查文件）、`03b4704`（L6-1/L6-2 文档更正），`b6e4ea4..03b4704` 快进推送到 `origin/csharp-solver-migration`，`main` 仍是 `eb60a02`，没有 PR。Step B 的提交只在本地。
- **引擎**（`csharp/src/TTC.Siren.Engine`，`725b917`）：协议版本 1，`POST /v1/solve`、`GET /v1/health`（协议、构建/commit、卡库 SHA-256、worker/队列状态）；`EngineOptions` 拒绝非回环地址（除非 `--allow-remote`）；`WorkerPool`：固定数量的 worker 上下文（默认 = 逻辑 CPU 数），每个 worker 独占一张 Fast 置换表（默认 96 MiB）和一张精确残局 memo（显式传给 `EndgameRobust.Select`，不依赖 `[ThreadStatic]`，对应 L6-4），首次使用时才创建；有界等待队列，满或超时返回 HTTP 503 `busy`；请求之间不共享可变状态（每个请求自己 `FromSnapshot`、自己的 `SearchContext`，H4-1）。robust 层在请求状态的**第二份副本**上运行。严格校验（未知 owner/规则/类型、边值、手牌超容量、行动方无牌 → `invalid_request`），`maxDepth`/`maxTime` 按 Python 语义（见报告 §2）。
- **共享 C# 代码的改动**（`3e232ec`，仅限允许的范围）：L-9（未知 owner 不再当作蓝方）、L-1（`maxDepth` 钳到 Python 的 100）、L-2（`SearchLimits`：NaN = 无限制，≤ 0 = 已超时）、`SearchStats` 的逐层时间/TT 命中/剪枝累计（只为引擎的逐层报告；搜索本身不变）。现有 78 个测试原样通过。
- **Python 改动**（`7fa5131`，仅限允许的范围）：新模块 `ai/csharp_engine.py`（只用标准库）；`ai_server.py`：求解块**原样**移入 `_solve_minimax_python`（与旧内联块的 AST 完全相同），`ai_move` 的 minimax 分支先试引擎（`TTC_CS_ENGINE`，默认开）、任何问题回退；`performance_stats`/`search_progress` 用引擎统计；新增 `engine`/`fallback_reason` 字段（引擎关闭时不出现）；默认关闭的请求录制 `TTC_RECORD_REQUESTS`。`solver=monte_carlo` 不变。
- **回退矩阵**（报告 §4）：connection_failed/timeout（熔断）、circuit_open、busy、invalid_request、engine_error、malformed_response、http_error、protocol_mismatch、card_db_mismatch、health_failed、illegal_move（槽位/卡 id/占用格/秩序·混乱）、unsupported_parameter、unexpected_status、client_error——每项有 pytest；`needs_fallback_candidates` 往返对随机数的消耗与 Python 路径一致，失败时随机状态恢复。
- **引擎关闭等价**：360 条请求（固定种子、`max_depth` 4、`max_time` 10^6）与 Phase 7 之前的提交 `03b4704` 逐一比较：**360/360 相同**（只排除测时间的字段和 minimax 自己按墙钟发出的中间进度项）；pytest 回归样本 43 条。
- **端到端比较**（Q-5 (b)，404 条记录 / 1,392 个搜索层）：引擎根分数与 C# oracle 位级相等 1,392/1,392；引擎走法在精确并列集合内 1,392/1,392；Python 走法与引擎相同 1,381，不同但在并列集合内 6，集合外但可由 D-006 解释 5，**无法解释 0**；robust 候选表（走法、final、safety、robust、corner、顺序）与 Python 逐位相同 209/209（1,274 个候选），场景数相同 209/209，最终走法 = 由 Python 的表与引擎标准走法算出的覆盖决策（覆盖 96 / 保持 113）。
- **并发确定性**：230 个不同请求、16 路并发打到 4 个 worker，两种顺序，逐字段（除时间）与新引擎上顺序执行的结果相同；跨并发度（1/4/16）与 worker 数（1/8/20）的 36 个格子 0 差异；请求之间无泄漏、`busy`、worker 池有测试。
- **延迟与压测**（仅限这台机器，i7-13650HX；**不外推到服务器**）：固定 P 核 `0xFFF`、3 次独立运行：开局 Python 10.0 s（到限时，深度 5）→ 引擎路径 0.56–2.75 s（深度 9）；中局 Python 3.76–4.02 s → 4.3–9.4 ms；残局 Python 0.013–0.54 s → 2.0–6.6 ms。引擎直接压测（P 核，20 worker，并发 16）：开局 5.5 请求/s、中局 160 请求/s、robust 约 5,500 请求/s；经 Flask 时受 GIL 限制（robust 约 450 请求/s，Python 侧每请求中位数 9–36 ms）；全部 20 个逻辑 CPU 时开局只有 3.4 请求/s（混合架构，E 核更慢）。内存：1 worker 164 MiB，8 worker 845 MiB，20 worker 1.6 GiB（工作集）。204 个压测格子没有错误、`busy` 或回退。
- **新增文件**：`docs/DEPLOY_WINDOWS.md`（部署：.NET 10 运行时、publish、配置、`sc.exe` 服务、防火墙、环境变量、健康检查、启动顺序、关闭与回滚、在 Windows Server 上选 worker 数的步骤与内存上界）、`docs/reports/PHASE7_INTEGRATION.md`、`docs/reports/phase7/`（原始测量）、`tools/phase7/`（语料生成、等价比较、e2e 参考、延迟、压测矩阵、确定性检查）、`tools/bench/load_test.py`、`tests/data/engine_off_expected.json`。
- **测试**：`dotnet test` 149 个全部通过（原 78 + 新增 71；原有测试没有改动断言）；`pytest` 126 个（原 34 + 92）；`dotnet build --no-incremental` 0 warning / 0 error。
- **已知限制**：未知牌采样、请求解析、文案、Flask 仍是 Python（Phase 8）；引擎路径不运行 Python 搜索，所以走法描述文本里的显示边可能与 Python 路径不同；墙钟限时的搜索结果取决于机器速度；Flask 内置开发服务器仍是前端（部署文档给出 WSGI 选项，未改启动方式）；未做 root parallelism、搜索优化、Phase 8。

#### Phase 7 加固交付记录（2026-10-10，`.claude/phase7_hardening_goal.md`；详细数据 `docs/reports/PHASE7_HARDENING.md`）

**状态：DONE，等待独立审查**（审查门禁由审查者设置，实现者不设）。**审查门禁（2026-10-10）：`PASS_WITH_NONBLOCKING_FINDINGS`**，见 `docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md`（L8-1 GC 结论措辞偏强、L8-2 排队中的短请求排在开局之后、L8-3 游戏端超时需 ≥ 75 s；上线检查清单见该文 §8）。Step A：把审查者写好的文件提交为 `45a9b28`，并把所有本地提交原样快进推送到 `origin/csharp-solver-migration`（`03b4704..45a9b28`，`main` 仍是 `eb60a02`，无 PR，无强推）。Step B（提交只在本地，未推送）：
- **M7-3**（`e35e8c3`）：`GoldenFiles.RunPythonProcess` 同时异步读取 stdout 与 stderr；回归测试（子进程各写 1 MiB 到两个流）；变异检查：改回顺序读取则测试在 60 s 限时后失败。`tools/phase7/e2e_reference.py` 适配新分发（`76eafcc`）。**证明**：删除缓存的 e2e golden 与语料后，`dotnet test` 通过测试框架重新生成并 **155/155 通过，总耗时 88 s**；重新生成的 404 条记录与改动前的缓存逐条相同（语料相同，只有三个绑定源文件的哈希不同），0 无法解释。
- **A-20 + M7-1**（引擎 `262ba77`，Python `76eafcc`）：引擎开启时 minimax 路径不再自动回退 Python；任何引擎问题（连接失败/超时/`busy`/`error`/`invalid_request`/响应异常/协议或卡库不匹配/非法走法）返回 **HTTP 503 + `{"error", "engine": "csharp", "engine_error": <原因>}`**（`max_depth`/`max_time` 不是数字 → 400 `unsupported_parameter`；Python 客户端自身的缺陷 → 500 `client_error`），Python 求解器不被调用（测试里用 spy 断言 `find_best_move_parallel`、`select_endgame_robust_move`、`_solve_minimax_python` 一次都没被调用）；失败以 error 级别记日志，`TTC_RECORD_REQUESTS` 照常记录；熔断器删除，健康检查（协议、卡库 SHA-256）在首次使用和任何连接级失败后重做，失败不缓存。客户端读超时 = `max_time` + 引擎排队等待（`/v1/health` 的 `queueWaitMs`）+ 余量；引擎丢弃连接已断开的排队请求（不搜索，不打断正在进行的搜索，搜索热路径无任何检查）；默认 `--queue 256`、`--queue-wait-ms 60000`。人工开关 `TTC_CS_ENGINE=off` 与 Python 求解代码（`_solve_minimax_python`、`ai/ai.py`、`core/`、残局函数）不变；引擎关闭的等价测试原样通过。`fallback` 测试被逐条替换为新行为的测试（原因与对应关系见报告 §2，提交信息里也有）。
- **M7-2**（`76eafcc` 实现、`9e60222` 池测试、`93b0da1` waitress）：进程级、线程安全的 LIFO keep-alive 连接池（`TTC_CS_ENGINE_POOL_SIZE`，默认 32）；失败的连接丢弃，空闲连接被服务端关闭时重试一次，多余连接显式关闭（不依赖 GC，PyPy）；可选启动脚本 `serve_waitress.py`（固定线程池，与 `app.py` 相同的初始化，`app.py` 不变；本机未安装 waitress，用替身模块测试了接线）。
- **过载复现**（`tests/test_engine_overload.py`：真实引擎进程 + 真实线程化 Flask；1 个 worker、12 个并发开局）：**12/12 由引擎回答，0 次客户端超时，0 次 Python 求解**；修复前同一场景 4/12 由引擎回答、8 个超时后回退 Python、随后 5 个廉价请求全部 `circuit_open`。队列满（`--queue 2`，8 个并发）→ 多余请求 503 `busy`。排队后被中止的请求不搜索（`abandonedSkipped`）。
- **连接数**（netstat，每种状态）：60 个顺序请求后到引擎端口的客户端连接，修复前 60 个，修复后 1 个（池大小 4 时的上限 4）。
- **L7-1**（`db8bab8`）：`ServerGarbageCollector` 是 MSBuild 不认的拼写（Server GC 其实是默认生效）；按 CLAUDE.md §6 实测 Server 与 Workstation GC（同一构建，`DOTNET_gcServer`；workers 12/16；并发 1/4/8/16；开局/中局/robust；P 核 `0xFFF` 与全部 20 个逻辑 CPU；各 3 次交替运行；`tools/phase7/gc_compare.py`，原始数据 `docs/reports/phase7/gc/`）：开局与中局无差别，robust 在 P 核上 Server GC 更快（8 格中 7 格中位数比值 < 1.0），内存相同；**选择 Server GC（非并发）并显式写入**，发布的 `runtimeconfig.json` 为 `"System.GC.Server": true`。
- **L7-2 / L7-3 与文档**：`PHASE7_INTEGRATION.md` 追加日期注记（绝对延迟随会话变化，只有同一会话内的比值可比；自动回退已撤销）；`DEPLOY_WINDOWS.md` 改写（引擎必需、无引擎的机器设 `TTC_CS_ENGINE=off`、503 错误矩阵、监控提示、新默认值、waitress、GC、§8 第 5 步更正）；新增 `docs/reports/PHASE7_HARDENING.md`。
- **测试结果**：`dotnet build --no-incremental` 0 warning / 0 error；`dotnet test` **155 个全部通过**（原 149 + 2 个 harness + 2 个中止请求 + 1 个默认设置 + 1 个 GC 配置）；`pytest` **152 个通过**（原 126；被替换的 fallback 测试一一对应，另加客户端/连接池/过载/launcher 测试）。
- **已知限制**：PyPy 与真实 waitress 未运行；Windows 服务（`sc.exe`）安装未执行；Server GC 未在 32 线程的目标服务器上测；排队请求可能等待至 `queue-wait-ms`（60 s）+ `max_time`，比这更短的游戏客户端超时会先放弃（排队中的请求随后被丢弃，已开始的搜索会完成）。Phase 8、root parallelism、搜索优化均未做；`TTC.Siren.Core`/`TTC.Siren.Search` 未改。

### Phase 8A — 交付记录（2026-10-11；审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`，见 `docs/reviews/OPUS_PHASE8A_REVIEW.md`）

提交（本地，未推送）：A `8f78212`；B `5afa01c`（RNG）、`9b9fcba`（解析+采样）、`7fc3273`（原生 `/ai_move`）；C `4d6f2e3`（搜索逐层回调）、`968118b`（解释事实 v2+模板）、`833c8f1`（对局日志+实时流）；D `8c97537`（O8-1 调度）、`489edb6`（密钥/限流/HTTPS/代理/占位页），其后为部署文档与报告提交。

- **新项目**：`TTC.Siren.Compat`（CPython 兼容 RNG、请求期 Python 对象模型 `Legacy*`、解析器、`UnknownCardHandler`、旧分析文案、Python `round`/`repr`/Flask JSON）、`TTC.Siren.Explain`（`FlipAnalyzer`、事实、模板渲染）、`TTC.Siren.Server`（Kestrel；对局日志、实时流、密钥、代理）。`Core` 不变；`Search` 仅 `FastSearcher` 增加每完成一层调用一次的 `onLevel` 回调（`LevelCallbackTests` 证明结果/统计/零分配不变）。Engine 仍是库与 loopback 宿主，服务器进程内直接复用 `SolveService`/`WorkerPool`。
- **一致性（全部精确，浮点位级）**：RNG 600 个种子 >10,000 次操作；解析 2,684 个请求 + 76 个拒绝（状态码与错误文本一致，39,288 个牌字段）；采样 5,400 个情形 116,078 张牌（378 种规则组合）；端到端 960 个请求与 Python 服务器逐值相同（44,569 个值）。端到端发现并复现 D-023。无 BLOCKED。
- **非位级/不一致项**（均在 §4）：D-020（畸形 JSON：400 对 Python 500；任意 Content-Type）、D-021（请求字段的异常 JSON 类型与 >2^63 的 id）、D-022（生产每请求新种子，按种子精确）；复现的 Python 怪异：D-023、D-024、D-025。无需统计检验（无依赖字符串哈希/集合顺序的构造）。
- **解释事实 v2**：8 个事实族，全部用引擎 `MakeMove`/精确求解/定深搜索重算验证（583 个决策，30,702 个文本数字全部能在事实中找到）；`docs/design/EXPLANATION_FACTS.md`。**对局日志** schema v1（`docs/design/GAME_LOG_SCHEMA.md`）：36 局完整对局重建、162/162 决策重放一致。**实时流**：request.received、search.depth、robust.progress（仅起止，Search 未授权改动）、decision、game.updated、game.ended、lagged。
- **安全**：密钥只存 SHA-256、CLI 管理、令牌桶+并发+流数限制、HTTPS/HSTS/CORS/安全头、仅回环的 `/admin/*`、非回环无 HTTPS 拒绝启动、代理 `/api/*`。撰写报告时发现并修复：被拒绝的密钥曾记录其前 8 个字符，现在不记录任何部分（测试改为断言不出现任何前缀）。
- **O8-1 调度**：按估计代价分三档+老化（默认 2 s 升一档）；审查探针：廉价请求 303–626 ms 内应答（FIFO 为 3,664–3,734 ms），所有开局均被应答，最长开局等待不增加。
- **测量（本机 i7-13650HX，P 核 `0xFFF`）**：robust 残局吞吐 825 req/s（8 并发）/930（16）对 Python+引擎 424/402；开局/中局持平；详见报告 §9。
- **测试**：`dotnet build -c Release --no-incremental` 0 警告 0 错误；`dotnet test` 228/228（此前 155）；`python -m pytest -q` 152 passed（`tests/test_csharp_engine_pool.py` 的并发池测试对时序敏感，在同机 `dotnet test` 之后偶发失败，单独运行通过，与本阶段无关）。
- **已知限制**：8B/8C/8D 未开始；对手落下第 9 张牌后若插件不再请求，日志看不到该牌；Windows 服务/win-acme/NSSM 步骤已写未执行；负载数字仅限本机。`assets/mascot/raw/*.png`（9 张，约 21 MB）未提交，等用户决定。
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
| A-7 | 随机采样类逻辑不追求位级 parity | Python RNG 与 `random.sample` 的实现细节不应复刻 | 已定 → **2026-10-11 由 A-23 细化**：Phase 8A 的未知牌采样移植用与 CPython 兼容的随机数生成器做到位级一致（便于证明"按原逻辑移植"并作为日后优化的基线） |
| A-8 | 集成先用子进程/本地 HTTP 桥接，再做全量替换 | 降低风险，可回退 | **已定（用户，2026-10-10）：本机 HTTP 服务**（只监听 127.0.0.1，所有 Python 进程共用一个引擎进程，引擎内工作线程池；也是 Phase 8 的基础）；默认开启，出错/超时/协议或卡库不匹配时自动回退 Python |
| A-9 | D-002 在 Python 和 C# 两边同时修复（按卡身份恢复 `type_modifier`） | 用户批准（Q-1）；否则同类规则下的搜索结果依赖访问顺序，Fast 模式的 parity 无法定义 | 已定（FIXED-BOTH） |
| A-10 | `Move` 的身份是卡实例 `(Inst, Cell)`，不是手牌槽位 | D-001 会在每次 make/undo 后重排手牌；Phase 3 真实出现过由此导致的 parity 错误；Phase 4 的 TT 走法/history/killer 会跨节点保存走法 | 已定（审查 H-3 整改） |
| A-11 | ReferenceCompat TT：128 位摘要代替字符串 key，按需翻倍扩容 + 硬上限（默认 2^25 槽），代际号清空 | Python 的 dict 无界且 key 是字符串；等价类已用 SHA-1 双射验证；扩容不改变结果 | 已定（审查 H-2 整改） |
| A-12 | 搜索入口用 `try/finally` 回退 `UndoDepth`，热路径不加 `try/finally` | 异常后共享的 `GameState` 必须恢复；保持热路径零开销 | 已定（审查 H-1 整改） |
| A-13 | golden 数据不入库：由测试按需调用 Python 生成，文件名与文件头带生成器版本 | 体积大（≈ 26 MB）且可重新生成；缺点是未绑定参考源码内容（审查 M-1） | 已定（M-1 待改进） |
| A-14 | 每个阶段由用户用 `/goal @.claude/*.md` 单独授权；独立审查产生"门禁"（`BLOCK_PHASE_4` / `PASS_TO_PHASE_4`），门禁只能由复审者改，整改者不能自己改 | 控制范围，防止越权进入下一阶段 | 已定 |
| A-15 | **Q-5 = 选项 (b)**：Fast 的根分数必须与精确的深度限制 minimax 值位级相等；根走法只允许在精确并列集合内与 compat/Python 不同，每个差异都要验证并报告；Fast 必须确定性 | 值级结果可精确验证，同时允许 TT 走法优先等排序改进；Python 的 TT（D-006）本身已不等于精确 minimax，所以"与 Python 一致"早已缩小为"与精确值一致" | 已定（用户，2026-10-07） |
| A-16 | Fast 的 oracle = ReferenceCompat 关闭 TT（`UseTranspositionTable=false`），并用独立的全宽 minimax 交叉验证；Fast 的 TT 只在同深度复用分数，不存被中断节点，bound 按原始窗口分类 | 避免 oracle 与被测对象共享同一个缺陷；D-006 使 compat（TT 开）不能当精确值 | 已定（Phase 4） |
| A-17 | Zobrist 哈希在 make 中增量更新，undo 恢复保存的哈希；哈希只覆盖决定局面价值的信息（格子→实例、owner、修正值、行动方），不含手牌顺序与显示边 | 排序相关状态（D-001/D-003）不影响分数；O(1) 的 undo 不会漂移 | 已定（Phase 4） |
| A-18 | 精确残局求解器的 memo 是 Zobrist 键的 `FastTranspositionTable`，上下文 (GameState, AI 方, `EvalMode.ExactCardDifference`)，每个场景 `GameState` 一个 memo 代，不与任何启发式搜索共用；终局值是整数牌数差 | Python 的结构化 memo 键只是实现细节（值是局面的纯函数，已验证 Python 共享 cache 与新 cache 无差异）；H4-1 的上下文规则必须适用于新的评分类型 | 已定（Phase 5） |
| A-19 | golden 数据绑定到参考源码：header 写入 Python major.minor 与每个依赖文件的 SHA-256，C# 不匹配就重新生成；覆盖逻辑通过 AST 从 `ai_server.py` 提取原块执行，而不是复制公式 | 避免 golden 比 Python 源码活得更久（审查 M-1/M4-2）；内联在 Flask 处理函数里的逻辑无法 import | 已定（Phase 5） |
| A-20 | **C# 引擎是 minimax 路径唯一的求解器，取消自动回退到 Python**（取代 A-8 中"出错/超时自动回退"的部分）：引擎出错、忙、超时、不可用、协议或卡库不匹配、返回非法走法时，`/ai_move` 返回明确的错误（HTTP 503 + 原因），不再悄悄改用 Python 求解；引擎繁忙时请求在引擎队列里等待（客户端等待时间必须长于引擎排队上限 + `max_time`）。保留一个**人工**应急开关 `TTC_CS_ENGINE=off`（运维手动设置才生效，从不自动触发）。Python 求解代码不删除：它仍是 parity 的参考实现（CLAUDE.md §1.2），golden 生成器依赖它 | 用户（2026-10-10）："现在完全可以替代 Python 了，没必要把 Python 作为回退源，出现 bug 修 bug"；Phase 7 审查 M7-1 证明过载时自动回退会把压力转给慢约 100 倍的 Python 并触发熔断 | 已定（用户，2026-10-10）；**已在 Phase 7 加固中实施**（`76eafcc`，测试逐条替换，见 `docs/reports/PHASE7_HARDENING.md` §2） |
| A-21 | **交付形态：云端服务 + Web 前端，求解器只在用户自己的服务器上**；游戏插件与网页通过 HTTPS 调用；**每用户一个 API 密钥**（运维发放/吊销，只存哈希，按密钥限流）；**对局日志存服务器**（按密钥归档、可回放/下载/删除，保留天数可配置，默认 30 天）；Web 前端用 React + TypeScript + three.js（react-three-fiber，Vite），由 C# 服务器托管 | 用户（2026-10-11）：项目不开源；云端交付让算法不离开服务器 | 已定（用户，2026-10-11；8A 实施后端部分） |
| A-22 | **蒙特卡洛求解器下线，不移植**：`solver=monte_carlo` 的请求由引擎回答并在响应中说明；`ai/monte_carlo.py` 保留在仓库中不再提供服务 | 比 minimax + 精确/robust 残局弱；源自第三方算法（LinkRoss），闭源产品中需另行考虑授权 | 已定（用户，2026-10-11；8A 实施） |
| A-23 | 未知牌采样**先按原逻辑移植**，用 C# 实现的 CPython 兼容随机数生成器（MT19937 与 `random.*` 的具体算法）做到给定种子时与 Python 位级一致；生产中每个请求使用新的密码学随机种子并写入对局日志，任何决策都可完整重放；采样的改进以后作为单独阶段，以移植版为基线 | 用户（2026-10-11）："先按照源逻辑做，后面肯定要优化"；位级一致是证明"原逻辑"的最强方式；可重放便于排查 | 已定（审查者设计，2026-10-11；8A 实施） |
| A-24 | **分析文案 v2 = 引擎给出可复算的结构化"解释事实" + 中文模板成文**（简要/详细两级，模板为数据文件，以后可加看板娘口吻）；文案只能陈述事实中有的内容；旧的分析字段为兼容现有插件按原逻辑移植保留，新内容放在追加字段 `explanation` | 用户（2026-10-11）：现有文案太死板，要更详细、维度更多；模板方案快、免费、可测试、不会编造 | 已定（用户，2026-10-11；8A 实施） |
| A-25 | 服务器进程内直接承载引擎（`SolveService`/`WorkerPool`），Engine 项目保留为库与 loopback 宿主 | 避免第二次序列化与进程跳转；Phase 7 的测试与工具继续可用 | 已实施（Phase 8A） |
| A-26 | 请求期的 Python 对象模型（`Legacy*`）只存在于兼容层，绝不进入搜索热路径；日志中不记录任何密钥内容（含被拒绝密钥的前缀） | 兼容层要复现 D-001/D-003/D-010/D-023；密钥卫生 | 已实施（Phase 8A） |
| A-27 | **访问分级与算力分级**：游戏插件只用 API 密钥（不建账号），请求优先级最高并预留可配置比例的工作线程；网页用户用同一个密钥登录一次，服务器换发 HttpOnly + Secure + SameSite=Strict 的会话 Cookie（密钥不存浏览器，可登出、过期、随密钥吊销失效）；匿名访客只读（落地页、演示回放、以后的 NPC 通杀库），并保留 8D 的"对局指导"，但匿名占用工作线程有上限（例如 25%）、搜索深度/时间封顶、按真实 IP 限流、全局匿名队列满时提示稍后再试；部署在 **Cloudflare** 之后：真实 IP 只从受信任代理的 `CF-Connecting-IP` 取，SSE 心跳短于 Cloudflare 空闲超时，动态接口不缓存，响应时间低于 Cloudflare 源站超时，运维接口拒绝任何带代理/Cloudflare 头的请求 | 用户（2026-10-11）：网页面向大众，重点是防滥用与排队节流；真正需要鉴权和高优先级的是插件实战求解 | 已定（用户，2026-10-11；8B 实施后端部分） |
| A-28 | **8E NPC 通杀库**：NPC 数据从游戏数据表导出（`TripleTriad`：固定牌×5、可变牌×5、规则×2、是否用流行规则；144 个 NPC）；流行规则（每地区每日变化、最多两条、客户端无数据，46 个 NPC）由插件上报当天规则、服务器当天按需计算并缓存；用到引擎未支持规则（交换/天选/不胜不休/随机，43 个 NPC）的标记为"不支持"；"通杀"= 固定我方引擎策略（不偷看 NPC 暗牌），穷举 NPC 全部可能手牌与每一步全部应对、双方先手，全部获胜才算（全明牌用精确值）；做不到时给必胜比例/不输比例/风险最低卡组；离线预算全卡库最优/通杀卡组与每种开局的第一手，用户自己的卡组按需计算并缓存，对局后续每步由引擎实时计算 | 用户（2026-10-11）；数据调研见 `docs/design/PHASE8_PLAN.md` §6.3 | 已定（用户，2026-10-11；8E 实施，未授权） |
| A-29 | **Web UI**：React + TypeScript + three.js（react-three-fiber）+ Vite，由 C# 服务器同源托管；深海暗色主题、桌面优先、手机自动简化为 2D；**中英双语**（界面文字 + 解释文案英文模板，事实与语言无关）；卡面为原创程序化设计（不使用游戏美术）；看板娘原图提交进私有仓库，构建时生成 WebP | 用户（2026-10-11） | 已定（用户，2026-10-11；8B 实施） |

---

## 9. BLOCKED / 待用户决定

当前 **没有 BLOCKED 项**（Phase 2、3、审查整改与 Phase 4 期间都没有出现无法解释的 Python/C# 差异；Phase 4 发现的 3 次 Python 与精确值不同，已归因于 D-006，见 §4）。以下问题需要用户在对应阶段之前做出决定（Q-5 已关闭）：

| ID | 问题 | 默认做法（未回复时） | 必须在何时之前决定 |
|---|---|---|---|
| Q-1 | ~~D-002 是否同时修复 Python 和 C#？~~ **已决定：修复（用户，2026-10-07），已完成，D-002 = FIXED-BOTH。** | — | 已关闭 |
| Q-2 | ~~D-004（手牌内 card_id 重复导致状态损坏）：在 Python 侧修复（按卡身份移除），还是只在输入层拒绝/报告？~~ **已决定：Python 与 C# 都修复（用户，2026-10-09）**；在 `.claude/phase6_goal.md` 步骤 B 中实施（独立提交，先于 Phase 6 的性能基线）。另外用户决定：**D-015 → Phase 7 用 Fast + 规范手牌顺序**；**授权 Phase 6**；**授权把 Phase 5 的提交推送到 `origin/csharp-solver-migration`**（不动 main、不提 PR；之后的提交只在本地，推送需另行授权） | — | 已关闭 |
| Q-3 | D-013 连携只有一层：确认这是有意设计吗？ | 保持 Python 行为 | Phase 8 前（不影响 parity） |
| Q-4 | D-012 卡库 474/475 全 0 空行是否应从候选池中排除？ | 保持 | Phase 5 前（**Phase 5 按默认保持；golden 的合法候选 id 列表含 474/475，与 Python 一致**；若用户之后要排除，Python 与 C# 要同时改） |
| Q-5 | ~~Fast 模式中，同分并列时是否允许选择与 Python 不同的走法？~~ **已决定：选项 (b)（用户，2026-10-07）——只允许在分数位级完全相等的走法之间不同，并逐条验证、报告；已在 Phase 4 完成，见 §7。** | — | 已关闭 |
| Q-6 | 是否需要 PyPy 的 baseline（生产构建使用 PyPy，本机未安装）？ | 只用 CPython baseline | Phase 5 前（**Phase 5 按默认只用 CPython 3.11.4**；`docs/reports/PHASE5_BENCHMARK.md` 注明了这一点） |
| Q-7 | ~~Phase 6 之后做什么、怎么上线？~~ **已决定（用户，2026-10-10）**：授权 Phase 7 + 请求级并行（`.claude/phase7_goal.md`）；**不做** root parallelism（只降低单个请求的延迟，对多用户吞吐无益）；把现有本地提交（`a0eb193..`，含 Phase 6 收尾文档）**原样快进推送**到 `origin/csharp-solver-migration`（不压缩、不动 main、不开 PR；Phase 7 的提交只在本地）；引擎为本机 HTTP 服务、依赖系统 .NET 10 运行时（ASP.NET Core 共享框架）、默认开启 + 自动回退、搜索参数沿用 Python（10 层 / 10 s）、部署在 Windows Server | — | 已关闭 |
| Q-8 | ~~Phase 7 审查之后做什么？~~ **已决定（用户，2026-10-10）**：先做 **Phase 7 加固**（`.claude/phase7_hardening_goal.md`：M7-1、M7-2、M7-3、L7-1、L7-2、L7-3），Phase 8 之后单独授权；**取消自动回退到 Python**（A-20，只保留人工 `TTC_CS_ENGINE=off`）；M7-2 用客户端进程级连接池修复，另提供可选的 waitress 启动脚本（`app.py` 原启动方式不变）；**授权把当前所有本地提交（Phase 7 的 11 个提交 + Phase 7 审查文件提交）快进推送**到 `origin/csharp-solver-migration`（不压缩、不动 main、不开 PR）；加固的提交只在本地，审查后再定 | — | 已关闭 |
| Q-9 | ~~Phase 7 加固之后做什么？Phase 8 怎么做？~~ **已决定（用户，2026-10-11）**：GitHub 仓库**改为私有**（审查者已于 2026-10-11 执行，`gh repo edit --visibility private`；公开期间的内容无法收回）；交付形态云端 + Web（A-21）；MC 下线（A-22）；Phase 8 拆成 **8A→8B→8C→8D**，**先做 8A**（`.claude/phase8a_goal.md`）；解析与未知牌采样先按原逻辑移植（A-23）；分析文案用事实 + 模板（A-24）；看板娘为原创角色"瑟琳"（清冷优雅的塞壬荷官，`docs/design/MASCOT.md`，图片放 `assets/mascot/raw/`）；牌组优化两种模式都要（指定 NPC/规则 + 通用强度，8C）；对局指正由用户摆双方、可切换"引擎代下对手"（8D）；采纳审查者建议的实战跟随、自动复盘报告、残局谜题、看板娘台词库；O8-1 在 8A 修复；8A 的提交只在本地，推送另行授权 | — | 已关闭 |
| Q-10 | ~~8A 审查之后做什么？~~ **已决定（用户，2026-10-11）**：授权 **8B**（`.claude/phase8b_goal.md`，先修 8A 遗留问题）；**授权把当前全部本地提交（含 8A 与本次审查文件）快进推送**到 `origin/csharp-solver-migration`（私有仓库；不压缩、不动 main、不开 PR；8B 的新提交只在本地）；看板娘 9 张原图提交进仓库；访问与算力分级见 A-27（插件只用密钥、网页用密钥登录、匿名只读并保留对局指导、插件优先、Cloudflare 防护）；NPC 通杀库见 A-28（新增 8E，顺序 8B→8C→8D→8E，对局指导仍在 8D）；界面见 A-29（加英文） | — | 已关闭 |

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
| 2026-10-08 | 5 | Python（CPython 3.11.4）vs C# Release，robust 残局全流程（`_build_endgame_scenarios` + `select_endgame_robust_move` 普通 + 激进），72 个确定性局面（种子 20261010；占用 5/6/7/8 × 未知 0/1/2 × 每格 6 个） | Python 8.7144 s（每次调用 3 次中位数之和）；C# 0.07786 / 0.07989 / 0.07969 s（每次调用 5 次中位数，3 次独立运行）→ 中位数 **0.07969 s** | **≈ 109×**（单次运行 111.9×）；72/72 个选择与 Python 相同；分格数据见 `docs/reports/PHASE5_BENCHMARK.md`（占用 5、未知 2：209×；占用 8：39–44×） |
| 2026-10-08 | 5 | 同上，精确求解器 `_solve_endgame_exact`（§3.6 的 end5 / mid4 + `mid3_ascension`），根值 | Python 中位数（3 次）0.00381 / 0.05012 / 15.58 s（217 / 4,707 / 297,935 次 make_move）；C#（3 次独立运行的中位数）0.00004 / 0.00082 / 0.05918 s（217 / 4,707 / 241,264 次 make_move），0 B 分配 | ≈ 95× / 61× / 263×；值相同（1 / −2 / −1）；C# 的 memo 与手牌顺序无关，所以 mid3 的 make 次数更少 |
| 2026-10-08 | 5 | §3.4 的 22 个固定深度项合计（43,003 节点），Python 与 C# ReferenceCompat，各 3 次 | Python 2.592 / 2.603 / 2.732 s → **2.603 s**；C# 0.0528 / 0.0516 / 0.0524 s → **0.0524 s** | ≈ 49.7×；节点数、分数、走法与 Python 相同（Phase 3 parity）；Phase 3 记录的 2.965 s 是另一次测量，两次相差在机器噪声范围内 |
| 2026-10-08 | 5 | 服务端等价调用 `max_depth=10`、`max_time=10`（§3.5），3 次 | Python：open_none 深度 5（187,537 / 187,078 / 187,870 节点）、open_same_plus 深度 5（178,099 / 176,681 / 178,213）、mid3_ascension 深度 10（30,500 节点，2.6 s）；C# ReferenceCompat：10 层 / 15.9 M 节点（8.8 s）、9 层 / 16.5–16.7 M、10 层 / 30,500（0.03 s）；C# Fast：9 层 / 2,566,250 节点 / 0.93 s、9 层 / 3,915,260 / 1.54 s、6 层 / 8,651 / 0.01 s | 走法在所有实现与所有次数中相同（27@(2,2)、27@(2,2)、53@(1,2)）；Fast 因整棵树搜完（深度 = 空格数，D-014）而停止；墙钟相关，深度随机器速度而变 |
| 2026-10-08 | 5 | Phase 4 的 `Bench fast`（d ≥ 4，3 次运行，Fast / compat TT 关闭 / compat TT 开） | Fast 1.3272 / 1.3304 / 1.3388 s，TT 关闭 2.0227 / 2.0234 / 2.0222 s，TT 开 2.1175 / 2.1230 / 2.1352 s（节点 3,625,620 / 3,771,758 / 3,431,185） | Fast ≈ 1.52× 于 TT 关闭、≈ 1.60× 于 Python 语义的 compat；与 Phase 4 同一代码（只是这一批机器负载更低，所以绝对值比 Phase 4 的记录小） |
| 2026-10-08 | 5 更正 | **更正说明**（审查 `docs/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md`；详见 `docs/reports/PHASE5_BENCHMARK.md` 的 "Corrections" 一节） | 上面 2026-10-08 的 Phase 5 行里：**"≈ 263×"（精确求解器）撤回**（那次 Python 计时 15.58 s 与后台生成 golden 的进程重叠，被污染）；**"≈ 49.7×"（22 项）是冷 JIT 数字**；robust 流水线的 109.4×/111.9× 是冷 JIT 数字；分格比值（31.8×…209.4×）是运行顺序的假象（占用 5 的格子最先跑，在 tier-0 代码上），不要再引用 | 之前的行保留作记录；以下面的新行为准 |
| 2026-10-08 | 5 更正 | `Bench` 改造（B-1/B-2）：默认 **warm**（先把整组项目反复预热 1500 ms 再计时），`--cold` 保留旧行为；每张表标 warm/cold，输出 min/median/max；逐项以 µs 打印，< 50 µs 标 `*`（受计时精度限制）；`fast` 三个变体轮换测量顺序；计时区域只含搜索/求解调用。`Bench search` 22 项合计，3 次独立进程运行 | warm：0.02333 / 0.02289 / 0.02306 s（评审的稳态 0.024–0.025 s）；cold：0.05104 / 0.05177 / 0.05287 s（旧配置约 0.05 s）。`open_none` d4：cold 25.6 ms（0.62 M NPS，min 8.6 / max 36.2 ms），warm 8.8 ms（1.80 M NPS）；`open_same_plus` d4：1.67–1.70 M NPS（两种模式） | 节点数（43,003）与 22 项分数位级与改造前逐项相同；`fast` 40 行节点数相同且全部 `same`；`exact` 的值、make_move、节点、memo 命中相同；`endgame` 72/72；`search`/`fast`/`exact` 分配列 0 |
| 2026-10-08 | 5 更正 | **Python/C# 交替运行**（B-3，`tools/bench/run_interleaved.py --rounds 3`；Python 与 C# 一轮一轮交替，同一负载）：22 项固定深度，C# warm | Python 2.6816 / 2.7109 / 2.7451 s；C# 0.02252 / 0.02280 / 0.02300 s（min / median / max）；比值 **118.9 / 119.1 / 119.3×** | 节点数 43,003 两边相同 |
| 2026-10-08 | 5 更正 | 同上，C# cold（`--cold`） | Python 2.7300 / 2.7554 / 2.7984 s；C# 0.04404 / 0.04684 / 0.05237 s；比值 **53.4 / 58.3 / 62.6×** | 旧 "49.7×" 对应此配置（分批测量） |
| 2026-10-08 | 5 更正 | 同上，精确求解器 `mid3_ascension`（warm，3 轮） | Python 10.239 / 10.325 / 10.393 s（297,935 次 make_move）；C# 0.0636 / 0.0639 / 0.0646 s（241,264 次）；比值 **161 / 161 / 162×**；每次 make_move：Python 34.7 µs，C# 0.265 µs → **≈ 131×**，工作量比 1.235 | 单独测 Python：10.35 / 10.47 / 10.49 s；评审测得 9.95–10.21 s，相差 1–5%（< 20%，无需人工取舍）。cold：Python 10.36 / 10.61 / 10.67 s，C# 0.0622 / 0.0649 / 0.0651 s，**163 / 164 / 167×**。小局面：`mid4_rev_ace` 245–308×（warm）/ 65–73×（cold），`end5_same_plus` 205–251×（warm，C# 14–17 µs，受计时精度限制）/ 86–99×（cold）；两者工作量比 1.000 |
| 2026-10-08 | 5 更正 | 同上，robust 流水线 72 个局面（warm / cold，各 3 轮） | warm：Python 9.1822 / 9.2005 / 9.5797 s，C# 0.04987 / 0.05081 / 0.05161 s，**177.9 / 184.5 / 188.5×**；cold：Python 9.0262 / 9.1422 / 9.2154 s，C# 0.08035 / 0.08243 / 0.08264 s，**110.9 / 111.5 / 112.3×** | 72/72 选择相同（每轮）；单次 warm 的分格比值 44–198×（占用 5：55 / 183 / 198×），见报告 C5 |
| 2026-10-08 | 5 更正 | robust 流水线的工作量对比（B-4，不计时；`tools/bench/count_robust_work.py` 与 `Bench endgame-work`；72 个局面 × 2 种模式，含每个场景的第一步） | Python **418,990** 次 make_move（每种模式 209,495），C# **389,990**（194,995），**1.074×**；两边评估的 (走法, 场景) 对相同：10,834 | 在相等工作量下 warm ≈ 172×，cold ≈ 104×。只有占用 5 的格子工作量不同（C# 每个场景一个 memo、Python 每个走法一个 cache） |
| 2026-10-08 | 5 更正 | `Bench fast`（Phase 4，d ≥ 4，5 次，轮换顺序），warm / cold | warm：TT 开 2.2288 s（2.1865–2.3197），TT 关 2.0791 s（2.0504–2.1300），Fast 1.3911 s（1.3521–1.4625）；cold：2.1897 / 2.0652 / 1.3830 s | Fast / TT 关 **1.49×**，Fast / TT 开 **1.60×**（warm）、1.58×（cold）。Fast 的节点数与 NPS **不能** 与 Python 或 ReferenceCompat 比较（不同的树，且把未排序的叶子计为节点） |
| 2026-10-08 | 5 更正 | 可信范围（对 CPython 3.11.4，单线程，本机，观察到的 3 轮 min–max，不是统计置信区间） | Python 语义搜索（相同节点）：cold 53–63×，warm 119×；robust 流水线：cold 111–112×，warm 178–189×（其中 1.074× 是工作量）；大残局精确求解：161–167×（其中 1.235× 是 memo 键）；Fast 对 TT 关闭：1.49× | **PyPy 未测量**（Q-6），所有比值只对 CPython 3.11.4；单一口径："≈ 55–190×（取决于工作负载与 JIT 预热），对 CPython 3.11.4，PyPy 未测"。不要引用 263× 或 49.7× |
| 2026-10-08 | 5 A-7 | **CPU 核心放置（A-7）**：本机 i7-13650HX 是混合 CPU（P 核 6 个 = 逻辑处理器 0–11，掩码 `0xFFF`；E 核 8 个 = 12–19，`0xFF000`），Windows 把长时间单线程运行非确定性地放在 P 或 E 核上。`tools/bench/pin.py` + `run_interleaved.py --affinity`：子进程以挂起状态创建、先设亲和性掩码再恢复（.NET 运行时因此按受限的 CPU 集启动），C# 直接启动已构建的 `TTC.Siren.Bench.exe`；掩码由 OS 的效率类检测（`--affinity auto`）或显式给出（`FFF`）；`Bench`/`py_bench.py`/`py_endgame_bench.py` 的表头打印它们实际运行的掩码，runner 校验并在每个结果行打印 | 同一份 Python 固定深度套件：未固定 2.68–2.75 s（一批）/ 3.40–3.47 s（另一批），固定到 P 核 3.04–3.10 s，固定到 E 核 4.72–4.74 s | **以后所有可信比值都是"两边都固定在 P 核集 `0xFFF`"**；未固定的运行在混合 CPU 上不可复现（runner 会警告）；C#-对-C# 的 Baseline→Compare 也必须固定到同一个集合（E 核让 C# 慢 1.3–1.4×） |
| 2026-10-08 | 5 A-7 | 两边固定在 P 核 `0xFFF`，3 轮交替，warm C#（min / median / max） | fixed：Python 3.0420 / 3.0725 / 3.1000 s，C# 0.02625 / 0.02625 / 0.02635 s，**115.9 / 116.6 / 118.1×**；exact（3 个局面）：Python 11.3794 / 11.4229 / 11.5766 s，C# 0.07024 / 0.07052 / 0.07186 s，**161.1 / 162.0 / 162.0×**（`mid3_ascension` Python 11.312 / 11.358 / 11.509 s，C# 0.06999 / 0.07018 / 0.07163 s）；robust 流水线：Python 9.7953 / 10.0004 / 10.0500 s，C# 0.05730 / 0.05788 / 0.05823 s，**169.2 / 172.6 / 174.5×** | 节点数 43,003、exact 值 1/−2/−1、72/72 选择每轮相同；与评审的 P 核固定值一致（fixed 118–121×、robust 178–189×、exact 150–180×；差在 15% 内） |
| 2026-10-08 | 5 A-7 | 同上，cold C#（`--cold`） | fixed：Python 2.8286 / 2.9547 / 3.0894 s，C# 0.03396 / 0.03591 / 0.04685 s，**60.4 / 82.3 / 91.0×**；exact：**159.5 / 170.2 / 174.5×**（`mid3_ascension` Python 10.707 / 10.870 / 11.112 s，C# 0.06176 / 0.06482 / 0.06664 s）；robust：Python 9.1942 / 9.4752 / 9.5828 s，C# 0.07799 / 0.08463 / 0.08564 s，**108.6 / 110.6 / 122.9×** | **暂定，待用户确认**：fixed cold 的第 1 轮 91.0× 比评审的 55–73× 上界高 25%（超过目标文件的 15% 带），中位数 82.3× 在带内。再跑 8 个固定到 `0xFFF` 的 cold 轮（仅 fixed）：83.6 / 57.5 / 59.4 / 81.9 / 81.0 / 70.1 / 63.0 / 63.8×；11 轮合计 **57.5–91.0×，中位数 70.1×**；C# cold 总时间呈双峰（约 0.035 s 或约 0.048 s，取决于 tier-1 代码何时到位），Python 稳定。用户只回复了"继续"，助手按最保守的方式暂定口径（不等于用户批准）：**同时写出完整范围与中位数（"57–91×，中位数 70×"），不写单个数字，也不收窄成"可信区间"**；可一句话改动 |
| 2026-10-08 | 5 A-7 | 两边固定在 E 核 `0xFF000`，warm，3 轮（记录核心类别的影响） | fixed：Python 4.7197 / 4.7386 / 4.7398 s，C# 0.03666 / 0.03679 / 0.03726 s，126.7 / 128.8 / 129.2×；exact：Python 16.4582 / 16.6698 / 17.4180 s，C# 0.09320 / 0.09331 / 0.09345 s，176.4 / 178.4 / 186.9×；robust：Python 15.4046 / 15.4759 / 15.7849 s，C# 0.08032 / 0.08238 / 0.08300 s，186.4 / 187.0 / 196.5× | E 核让 Python 慢 1.46–1.55×、C# 慢 1.32–1.42×；两边同类时比值只高 8–10%，**混合放置**才会破坏比值（评审曾见 271× / 301×，P 核固定的真实值约 162× / 173×） |
| 2026-10-10 | 6 | C# 基线（`a0eb193`，未优化），固定 P 核 `0xFFF`，warm，3 次进程运行，`run_cs_pinned.py` | search / fast / exact / endgame / make-undo | 22 项 0.02338 s；Fast d≥4（TT 开/关/Fast）2.306/2.193/1.438 s；exact mid3 0.06447 s；robust 流水线 0.05238 s；make/undo mid3 0.7239 s、open_all_rules 1.0257 s | cold（5 次）：search 0.0409、exact mid3 0.0652、endgame 0.0777 s；交替 Python/C#：fixed 114–119×、exact 167–171×、robust 181–186×（warm） |
| 2026-10-10 | 6 | profile（`dotnet-trace`，采样线程时间，12 s，固定 `0xFFF`，`Bench loop`） | 优化前 | Fast：`MakeMove` 44%（`RecalculateTypeModifiers` 29%）、`EvaluateState` 31%；compat：`OrderMoves` 57%（`EvaluateMove` 45%）；exact：`MakeMove` 72%（`RecalculateTypeModifiers` 59%） | 复现评审 §8 的证据；原始摘要在 `docs/reports/phase6/profile_*_before.txt` |
| 2026-10-10 | 6 | 同窗口 A/B（块顺序）：原始提交 `48e38cd` vs 最终 `782a3b5`，`0xFFF`，warm，7 次进程运行 | 逐项 | compat 22 项 0.02439 → 0.01839 s（−24.6%）；Fast（TT 开/关/Fast）2.300/2.180/1.443 → 1.777/1.658/1.026 s（−22.8/−23.9/−28.9%）；exact mid3 0.06755 → 0.04191 s（−38.0%）；robust 0.05356 → 0.03859 s（−28.0%）；make/undo 含同类规则 −49.9%/−44.3%，不含同类规则 ±1.4%（不变） | 节点数、分数位、走法、exact 值、make/undo 校验和全部与原始构建逐字相同（SIG 哈希相同）；cold：Fast −21..−27%、exact mid3 −34%、robust −1%（另一次 −20%）、块顺序的 compat 22 项 +24.6% 是顺序假象；**交替构建**（`ab_interleaved.py`）：compat 22 项 0.04450 → 0.04116 s（−7.5%）、Fast −28%、exact mid3 −34%、robust −23%（见 `PHASE6_OPTIMIZATION.md` §5） |
| 2026-10-10 | 6 | 交替 Python/C#，两边固定 `0xFFF`，3 轮，**最终提交 `427d67f`** | fixed / exact / robust | warm：**156.0/162.4/165.7×**、**249.4/255.2/263.4×**、**235.9/239.7/254.8×**；cold：62.9/63.4/66.6×、241.8/256.6/259.1×、140.9/142.7/150.3× | 对 CPython 3.11.4；每个子进程报告 `0xFFF`，72/72 选择每轮相同；exact 中 1.235× 来自 memo 键、robust 中 1.074× 来自 memo 共享方式（算法性）；原始文件 `docs/reports/phase6/final_427d67f_interleaved_*.txt` |
| 2026-10-10 | 6 | 候选记录 | C1–C9 | 保留：C1 一次遍历实例表重算同类修正、C3 查表式 `PlacementEdgeScore`、C4a `EvaluateMove` 用邻接表、C7 复用 exact memo；回退：C2（Span 拷贝快照）、C5a、C6、C8a、C9（−2.8% 中位数） | 每个候选的数据见 `docs/reports/PHASE6_OPTIMIZATION.md` §3 与 `docs/reports/phase6/candidates_log.txt` |
| 2026-10-10 | 7 | 引擎关闭等价（`equivalence.py`，360 请求，对 `03b4704`） | `/ai_move` 响应 | **360/360 相同**（347 有走法，13 无走法）| 只排除测时间的字段与 minimax 按墙钟发出的中间进度项；`PYTHONHASHSEED=0` 与不固定各跑一次 |
| 2026-10-10 | 7 | 端到端（`EngineEndToEndTests`，404 记录 / 1,392 层） | 引擎 vs C# oracle vs Python | 引擎分数 = oracle 位级 1,392/1,392；走法在并列集合内 1,392/1,392；Python 走法：相同 1,381、并列集合内另一手 6、D-006 解释 5、无法解释 0；robust 表逐位相同 209/209，覆盖 96 / 保持 113 | 并列集合大小 1/2/3/4/6：1294/86/8/3/1 |
| 2026-10-10 | 7 | 单请求延迟（固定 P 核 `0xFFF`，3 次独立运行，min/median/max，warm/cold，生产参数 10 层 / 10 s） | Python 求解器 vs 引擎路径（Flask + 引擎） | 开局：Python 10.02–10.03 s（深度 5，到限时）→ 引擎路径 0.56–0.63 / 2.46–2.75 / 0.79–0.92 s（深度 9）；中局 3.76–4.02 s → 4.3–9.4 ms；k:4–7 的 robust/残局请求 0.013–0.54 s → 2.0–6.6 ms（min–max over 3 runs，逐请求见报告 §8.1） | 表见 `PHASE7_INTEGRATION.md` §8.1；cold 与 warm 接近 |
| 2026-10-10 | 7 | 引擎压测（`run_load_matrix.py`，worker 1/4/8/12/20 × 并发 1/2/4/8/12/16，全部 CPU 与 P 核 `0xFFF`） | 开局 / 中局 / robust，请求/s | P 核 20 worker 并发 16：开局 5.5、中局 160、robust 5,528；全部 CPU：3.4 / 113 / 3,316；经 Flask（P 核）：5.5 / 112 / 452（Python 侧每请求中位数 9–36 ms，GIL 受限） | 204 格无错误、无 `busy`、无回退；表见报告 §8.2；**只代表这台机器** |
| 2026-10-10 | 7 加固 | 过载（`tests/test_engine_overload.py`，1 个 worker，12 个并发开局，`max_time` 1 s，读余量 1 s；修复前用 `262ba77` 的探针） | 12 个并发开局 | 修复前：4/12 由引擎回答，8 个客户端超时后回退 Python，随后 5 个廉价请求全部 `circuit_open`；修复后：**12/12 由引擎回答，0 超时，0 次 Python 求解**，最慢 6.6 s（旧读超时 3.0 s） | 同一会话、同一台机器；队列满 → 503 `busy` |
| 2026-10-10 | 7 加固 | 连接数（netstat，所有状态） | 60 个顺序请求经线程化 Flask | 修复前 60 个到引擎端口的客户端连接，修复后 1 个（池大小 4） | 同上 |
| 2026-10-10 | 7 加固 | Server GC 对 Workstation GC（`gc_compare.py`，同一构建 `DOTNET_gcServer=1/0`，workers 12/16，并发 1/4/8/16，P 核 `0xFFF` 与全部 CPU，各 3 次交替运行，median [min–max]） | 吞吐（请求/s）、p95、工作集 | workstation/server 吞吐比：开局 0.97–1.04，中局 0.94–1.05，robust（P 核）0.80–1.01，robust（全部 CPU）0.94–1.24（区间重叠）；工作集 12 workers 1,235/1,234 MiB，16 workers 1,623/1,621 MiB | 选择 Server GC（非并发）；完整表 `docs/reports/phase7/gc/gc_table.txt`；3 次的 min–max 不是置信区间 |
| 2026-10-10 | 7 加固 | e2e golden 重新生成（删除缓存后 `dotnet test`） | 总耗时 | 155/155 通过，88 s（语料与 e2e 参考重新生成，其余 golden 已是最新） | 与改动前的 404 条记录逐条相同 |
| 2026-10-10 | 7 | 内存（引擎进程，压测后工作集） | worker 1 / 4 / 8 / 12 / 20 | 164 / 461 / 845 / 1,233 / 1,622 MiB | 空闲 50 MiB；每个用过的 worker 约 77 MiB（96 MiB 的表按需触碰）|
| 2026-10-11 | 8A | `/ai_move` 吞吐（`tools/phase8a/load_server.py`，P 核 `0xFFF`，12 worker，8/16 并发） | robust 残局 req/s | C# 服务器 825/930；Python+引擎 424/402 | 含密钥检查、限流、解释、日志；原始数据 `docs/reports/phase8a/` |
| 2026-10-11 | 8A | O8-1 探针（1 worker，12 开局+5 廉价） | 廉价请求应答时间 | FIFO 3,664–3,734 ms → 优先级+老化 303–626 ms | `SchedulerTests` |
| 2026-10-07 | 2 | Python（参考） | `_solve_endgame_exact` 复测 | end5: 217 make / 0.004 s；mid4_rev_ace: 4,707 make / 0.055 s（≈85k make/s） | 与 Phase 1 一致；Python 的 make/s 含 minimax 与 memo 开销，不是纯 make/undo，不可直接比 |

---

## 11. 下一步行动

00. **Phase 8A 已完成（2026-10-11），审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`（`docs/reviews/OPUS_PHASE8A_REVIEW.md` §7/§8/§10；建议 8B 第一步先修 M8-1、M8-2、L8-1…L8-4）**，见 §7 Phase 8A、`docs/reports/PHASE8A_REPORT.md`。下一步由用户决定：审查 8A，然后单独授权 8B（Web UI）/8C/8D；9 个看板娘 PNG 是否提交；是否推送本地提交。审查建议：日志中无密钥（含前缀）、O8-1 老化上界、`/v1/games` 的密钥隔离、D-020…D-025 的措辞、事实与文本一致性。
0a. **Phase 7 加固已完成（2026-10-10），审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`（`docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md` §7/§8/§10）**（见 §7 Phase 7 加固交付记录）：审查重点建议为——各条引擎错误路径都不调用 Python 求解器、被替换的测试与新测试的对应关系、过载复现与连接数测试的可信度、连接池在 `threaded=True` 与（未测的）PyPy 下的行为、引擎对已中止排队请求的处理、GC 的实测表、e2e golden 的重新生成。Step B 的提交**未推送**（需要用户另行授权）；Phase 8 未开始。
0. **Phase 1–7 完成（Phase 7 审查门禁 `PASS_WITH_NONBLOCKING_FINDINGS`，2026-10-10，`docs/reviews/OPUS_PHASE7_REVIEW.md` §7/§9/§10；Phase 6 门禁 `PASS_WITH_NONBLOCKING_FINDINGS`）。** 下一步由用户决定：(a) 让审查者复核 Phase 7（重点：引擎关闭等价的方法与排除项、端到端比较的分类（Q-5 (b)、D-006）、回退矩阵、并发确定性、对共享 C# 代码的小改动 L-1/L-2/L-9 与逐层统计、`ai_server.py` 求解块的原样移动、请求录制的隐私说明）；(b) 是否推送 Phase 7 及之前的本地提交（`a0eb193` 已随 Step A 推送；本地未推送的是 Phase 7 的提交，`3e232ec`…，CLAUDE.md §8：未经授权不推送）；(c) 在 Windows Server 上按 `docs/DEPLOY_WINDOWS.md` 部署并测 worker 数；(d) Phase 8（全量服务移植）与 root parallelism 均**未开始、需要用户明确授权**。
1. 工作区状态（2026-10-10）：分支 `csharp-solver-migration`，Phase 5 的 5 个提交已推送（`710a70e..b6e4ea4`）；本地未推送：`a0eb193`（D-004 双边修复 + D-019）、`48e38cd`（Phase 6 工具）、`3a77dbe`（C1）、`6b61630`（C3）、`2c226bb`（C4a）、`782a3b5`（C7）、`c6d3115`（文档）、`17c07c7`（回退 C3）、`290b15e`（重新应用 C3）及随后的文档提交。`csharp/.vs/`、`bin/`、`obj/`、golden 数据不入库。PR #1 已关闭，`main` 仍是 `eb60a02`。
2. Phase 6 开始前不需要用户回答问题。已有默认的待决问题：~~Q-2（D-004，Phase 7 前）~~（**已决定并完成，2026-10-09：两边修复，FIXED-BOTH**）、Q-3（连携一层，Phase 8 前）、Q-6（PyPy baseline，按默认 CPython）。
3. 仍未处理的审查遗留项：L4-1 到 L4-7、O4-1 到 O4-3、Phase 3 的 L-3…L-8、L-10…L-12 与 O-1 到 O-5（见 §12.7）；**M4-3 已由 D-015 的决定和 Phase 7 关闭**；L-1/L-2/L-9 已在 Phase 7 处理；L6-4 已处理（引擎的 worker 显式持有 memo）；L6-5（`exact`/`scenario` golden 没有重复 id）仍开放。

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
- **（Phase 4 的改动已提交并推送：`1f2d0a9` Zobrist、`9faf189` Fast 搜索与 H4-1、`710a70e` 文档，用户于 2026-10-08 要求推送；HEAD `710a70e`。）下面一条是 Phase 4 完成当时的状态，保留作记录：**
- **（Phase 4 之后，当时）未提交的改动**：`CSHARP_MIGRATION.md`、`csharp/src/TTC.Siren.Core/GameState.cs`、`csharp/src/TTC.Siren.Search/CompatSearcher.cs`、`csharp/src/TTC.Siren.Bench/Program.cs`（已修改）；`ZobristKeys.cs`、`FastSearcher.cs`、`FastTranspositionTable.cs`、`ZobristTests.cs`、`FastSearchTests.cs`（新文件）。CLAUDE.md §8：用户没有要求就不要 commit/push。
- 本机 git 会提示 "LF will be replaced by CRLF"（`core.autocrlf`），无害。
- **（2026-10-10，Phase 7）** `origin/csharp-solver-migration` = `03b4704`（Phase 1–6 与 Phase 6 收尾文档，原样快进推送）；本地领先的 Phase 7 提交：`3e232ec`（L-1/L-2/L-9 + 逐层统计）、`725b917`（引擎项目 + 测试）、`285efde`（语料与等价工具）、`7fa5131`（Python 客户端与分发）、`82237f4`（端到端比较）、`167163b`、`55cf4fc`（压测工具）以及随后的测量与文档提交；未推送。
- **（2026-10-10，Phase 7 加固后）** `origin/csharp-solver-migration` = `45a9b28`（Step A：Phase 7 全部提交 + 审查文件，用户授权的推送，原样快进）；本地领先（未推送）的加固提交：`e35e8c3`（M7-3）、`262ba77`（引擎：丢弃中止的排队请求等）、`76eafcc`（A-20 + 连接池）、`9e60222`（池测试）、`93b0da1`（waitress 启动脚本）、`db8bab8`（Server GC）及随后的文档提交。`main` 仍是 `eb60a02`。

### 12.2 已完成的阶段与用户决定（Phase 4、5 的行见表末）

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
| **Phase 4** Zobrist / bound-aware TT / Fast | **DONE**（2026-10-07，`/goal @.claude/phase_goal.md` + Q-5 = (b)） | 验收见 §7 Phase 4；改动未提交**（已更正：Phase 4 的提交 `1f2d0a9`/`9faf189`/`710a70e` 早已推送）** |
| 独立审查 Phase 4 | 初审 `BLOCK_PHASE_5`（H4-1），整改后 **`PASS_TO_PHASE_5`** | `docs/reviews/OPUS_PHASE4_REVIEW.md`（§13 复审） |
| 推送分支 | DONE | 用户于 2026-10-08 明确要求：`d0ef721..710a70e` 推送到 `origin/csharp-solver-migration`（`main` 未动） |
| M4-1、M4-2 | DONE（2026-10-08） | 用户要求在生成任何 Phase 5 golden 之前先解决，不扩大为 Phase 4 清理 |
| **Phase 5** 精确残局 / robust / 场景 / benchmark | **DONE**（2026-10-08，`/goal @.claude/phase_goal.md`） | 验收见 §7 Phase 5；Q-4、Q-6 保持默认；改动未提交**（已更正：Phase 5 的 5 个提交 `e10574a`…`b6e4ea4` 已于 2026-10-09 推送）** |
| **Phase 7** Python 服务调用 C# 引擎 | **DONE**（2026-10-10，`.claude/phase7_goal.md`）；审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（`docs/reviews/OPUS_PHASE7_REVIEW.md`） | Step A：审查文件 + L6-1/L6-2 + 推送 `b6e4ea4..03b4704`；Step B：本机 HTTP 引擎 + Python 客户端与回退 + 验证 + 压测 + 部署指南；提交只在本地 |
| **Phase 7 加固** | **DONE**（2026-10-10，`.claude/phase7_hardening_goal.md`）；审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（`docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md`） | 用户决定 A-20/Q-8：取消自动回退、只留人工 `TTC_CS_ENGINE=off`；Step A 推送 `45a9b28`；Step B 本地提交；详见 §7 Phase 7 加固交付记录 |
| **Phase 6** | **DONE**（2026-10-10）；审查门禁 **`PASS_WITH_NONBLOCKING_FINDINGS`**（`docs/reviews/OPUS_PHASE6_REVIEW.md`） | 用户 2026-10-09 授权 `.claude/phase6_goal.md`：A 推送 Phase 5（`b6e4ea4`）、B Q-2/D-004 双边修复（`a0eb193`，本地）、C 性能优化（4 个保留提交，本地）；详见 §7 Phase 6 交付记录 |

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
  - **Phase 5**：`ExactEndgameSolver.cs`（精确求解 + 绑定上下文的 memo）、`EndgameScenarios.cs`（合法候选/去重/威胁分数/排序/排列截断/场景状态）、`EndgameRobust.cs`（corner 风险、一步回应、`Select`、`ShouldUse`、`DecideOverride`）；`EvalMode.ExactCardDifference`
- `src/TTC.Siren.Bench`：`makeundo`、`search`、`fast`、`exact`、`endgame`（需要 `tools/bench/endgame_bench_positions.json`，由 `py_endgame_bench.py` 生成）、`server` 模式。
- `tests/TTC.Siren.Tests`（xUnit，**78 个测试**；Phase 5 新增 `EndgameDifferentialTests`(8)、`ExactEndgameSolverTests`(3)，M4-1 `ZobristSufficiencyTests`(2)，M4-2 `GoldenBindingTests`(3)，辅助 `GoldenSources`/`EndgameGolden`；H4-1 整改新增 `FastTableContextTests`(4)；Phase 4 新增 `ZobristTests`(4)、`FastSearchTests`(8，含 `Oracle` 辅助)、`FastTranspositionTableTests`(3)）：`RulesTraceDifferentialTests`、`PortedRuleTests`、`GameStateBehaviourTests`（D-001/3/4、CardDb 对照 Python 等）、`SearchDifferentialTests`（评估 / 固定深度 / ID+超时 / TT key 四个）、`SearchBehaviourTests`、`SearchExceptionSafetyTests`（H-1）、`CompatTtGrowthTests`（H-2，含 30 s 的服务端等价测试）、`MoveIdentityTests`（H-3）、`AllocationTests`/`SearchAllocationTests`（0 字节）；辅助 `GoldenFiles`、`SearchGolden`。
- Python 侧工具：`tools/parity/py_rules_trace.py`、`py_search_trace.py`、`py_endgame_trace.py`（golden 生成器，只 import 参考实现）、`tools/parity/golden_sources.py`（把 golden 绑定到源码哈希）、`tools/bench/py_endgame_bench.py`（Phase 5 benchmark）、`tools/bench/py_bench.py`（基准脚本，用法 `python tools/bench/py_bench.py <repo_root> fixed|exact|server|all`）。
- **Phase 7**：`src/TTC.Siren.Engine`（`Program.cs`、`EngineHost.cs`、`EngineOptions.cs`、`Protocol.cs`（请求读取与校验）、`SolveService.cs`、`WorkerPool.cs`、`ResponseWriter.cs`、`appsettings.json`）；Python：`ai/csharp_engine.py`（客户端）、`ai_server.py` 的 `_solve_minimax_python` / `_solve_minimax_with_engine` / `_engine_outcome` / `_engine_move` / `_record_request`；`tools/phase7/`（`gen_corpus.py`、`equivalence.py`、`e2e_reference.py`、`latency.py`、`run_load_matrix.py`、`determinism_check.py`、`serve_flask.py`、`render_tables.py`）、`tools/bench/load_test.py`；测试：C# `EngineTests`、`EngineConcurrencyTests`、`EngineEndToEndTests`、`ReviewFindingsTests`，Python `tests/test_csharp_engine_client.py`、`test_engine_integration.py`、`test_engine_off_equivalence.py`。
- 审查与目标文件：`docs/reviews/OPUS_PHASE3_REVIEW.md`、`.claude/phase_goal.md`（通用阶段目标）、`.claude/sonnet_review_fix_goal.md`（整改目标，已完成）。

### 12.4 构建、测试、基准、golden 数据（命令与注意事项）

```
dotnet build csharp/TTC.Siren.sln -c Release --no-incremental        # 期望 0 warning / 0 error
dotnet test  csharp/TTC.Siren.sln -c Release                           # 期望 78 通过 / 0 跳过，约 30 s（缓存的 golden 存在时）；golden 首次生成或源码哈希变化后重新生成约 10 分钟
python -m pytest -q                                                    # 期望 34 passed（26 原有 + 3 个 D-002 回归 + 5 个 D-004 回归）
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search [runs=5]
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- makeundo [pairs=2000000] [runs=5]
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- fast [runs=5]     # Phase 4：compat TT 开 / compat TT 关（oracle）/ Fast
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- exact [runs=5]    # Phase 5：精确求解器
python tools/bench/py_endgame_bench.py --out tools/bench/endgame_bench_positions.json   # 然后：Bench endgame <该文件> [runs]
```

- **golden 数据**在 `tools/parity/golden/`（被 git 忽略）。测试发现文件缺失时会自动调用 `python tools/parity/py_*_trace.py` 生成（失败则测试失败，不跳过）。文件名：`rules_trace_g2_s20261007_n20000.jsonl.gz`（≈6 MB，≈20 s）、`search_eval_g1_s20261008_n100000.jsonl.gz`（≈20 MB，≈3 min）、`search_search_g1_s20261008_n900.jsonl.gz`、`search_id_g1_s20261008_n200.jsonl.gz`、`search_tt_g1_s20261008_n3000.jsonl.gz`、`carddb.json`；全部生成约 10 分钟，需要 `PATH` 上有 Python 3.11。目录里可能还有一个不再使用的 `search_search_g1_s20261008_n600.jsonl.gz`（旧计数，可删）。
- **（M4-2，2026-10-08）golden 现在绑定到参考源码**：header 里的 `source_sha256` / `python_version` 与工作区不一致时，测试自动删除并重新生成该 golden（下面这条"手动提升版本"的旧规则只在 header 格式改变时才需要）。endgame golden 文件名 `endgame_<kind>_g1_s20261009_n<count>.jsonl.gz`（kind：scenario、unique、corner、reply、exact、robust、override）。
- **golden 失效规则（旧，Phase 3/4）**：Python 参考实现（`core/*`、`ai/ai.py`）的语义改变时，必须同时提升生成器版本（`GENERATOR_VERSION`，规则 golden 为 `py_rules_trace/2`，搜索 golden 为 `py_search_trace/1`）和 `GoldenFiles.cs` 里对应的常量/文件名里的 `g2`/`g1`；测试会校验文件头里的版本号。**但 golden 没有和参考源码内容绑定（审查 M-1）**：改了 Python 却忘了提升版本号，测试会继续对着旧数据通过。
- 基准机器噪声大：同一代码连续运行就能差 ±10%。比较前后性能必须在同一时段、同一机器、至少 3 次独立运行取中位数；单次差异不能归因于改动。
- 开发环境陷阱（本会话踩过）：
  - PowerShell 里 `R`、`Rp` 是内置别名（`Invoke-History`、`Remove-ItemProperty`），不能用作自定义函数名（导致批量文本替换静默失败）。
  - Bash 工具禁止前台 `sleep` 等待；要等后台任务就用 `run_in_background` 加 `until` 循环，或直接轮询日志文件。
  - `dotnet new sln` 在 SDK 10 默认生成 `.slnx`；本仓库用经典 `.sln`（`--format sln`）。
  - C# 里 `Span<T>`（`stackalloc`）不能被 lambda 捕获；测试里要用普通循环。
  - 修改 `.md` 里的大段中文用 Edit 工具（或 UTF-8 无 BOM 的 `[IO.File]::WriteAllText`），不要用会改写编码的重定向。
- 最近一次完整验证（会话结束前）：build 0 warning/0 error；`dotnet test` 43/43；`pytest` 29 passed；golden 计数：固定深度 4,507 次/1,152,695 节点（900 局面 + 22 基准项）、ID 203 + 600（586 次被节点预算中断）+ 200 次直接带预算 minimax、评估 100,000 局面 / 400,000 + 1,592,088 次检查、规则轨迹 20,000 序列/182,910 步、TT key 3,000 家族/16,733↔16,733；分配测试 0 字节（1,000,002 次 make/undo；371,212 个搜索节点）。

### 12.5 关键设计决定（汇总；细节见 §7 各 Phase 记录与 §8、§4）

1. **Python 是参考实现**；已知怪异行为由 C# 复现并以 `// PY-QUIRK(Dxxx)` 标注：D-001（undo 把卡追加到手牌末尾）、D-003（显示边不随 undo 恢复，`evaluate_move` 读显示边）、D-004（手牌内重复 id → C# 在载入时抛异常）**【已更正：已修复（FIXED-BOTH），C# 接受重复 id，见 D-004/D-019】**、D-005/D-006（TT key 拼接冲突、TT 无 bound）、D-007（终局仍用启发式）、D-013（连携只有一层）、D-014（ID 不会提前停）。**D-002 已修复并同步到两边（FIXED-BOTH）。**
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
- **Phase 5（2026-10-08，数据与方法见 `docs/reports/PHASE5_BENCHMARK.md`）**：robust 残局全流程 72 个局面（占用 5–8 × 未知 0/1/2）Python 8.7144 s、C# 0.07969 s（3 次中位数）≈ **109×**，72/72 选择与 Python 相同；精确求解器 end5 / mid4 / mid3_ascension：C# 0.00004 / 0.00082 / 0.0592 s，Python 0.0038 / 0.0501 / 15.58 s（≈ 95× / 61× / 263×），值相同、0 字节分配；§3.4 的 22 个固定深度项 Python 2.603 s、C# ReferenceCompat 0.0524 s（≈ 49.7×）；服务端等价调用：Python 在 10 s 内完成深度 5/5/10，C# ReferenceCompat 10/9/10，Fast 在 0.93/1.54/0.01 s 内搜完整棵树（深度 9/9/6），走法相同。
- **（2026-10-08 更正）上面 Phase 5 的比值以 §10 的"5 更正"行和 `docs/reports/PHASE5_BENCHMARK.md` 的 Corrections 一节为准**：263× 撤回（161–167×）；49.7× 是冷 JIT（cold 53–63×，warm 119×）；robust 流水线 cold 111–112×、warm 178–189×；单一口径 "≈ 55–190×，对 CPython 3.11.4，PyPy 未测"。
- 还没有的：PyPy baseline（Q-6，按默认只用 CPython）、profiler 数据（Phase 6）。**没有数据支撑的性能结论不得写入文档。**

### 12.7 未解决的问题与待办（均不阻塞下一阶段，除非标明；Phase 5 之后的更新见 §11）

**需要用户回答**
- ~~Q-5~~ 已决定：选项 (b)（用户，2026-10-07），Phase 4 已实现并验证（§7 Phase 4）。
- ~~Q-2（D-004 重复手牌 id，Phase 7 前）~~（**已决定并完成：FIXED-BOTH**）、Q-3（连携只有一层是否有意，Phase 8 前）、Q-4（474/475 空行是否排除，Phase 5 前）、Q-6（是否需要 PyPy baseline，Phase 5 前）：默认值见 §9。
- 是否把 `csharp-solver-migration` 合并到 `main`（目前用户不希望提 PR）；是否撤销已推送的提交（需要强制推送授权）。

**审查遗留项（`docs/reviews/OPUS_PHASE3_REVIEW.md`；未处理）**
- M-1：golden 没有绑定参考源码（建议在文件头写入 `core/*.py`、`ai/ai.py`、`csv_compat.py`、生成器、CSV 的 SHA-256，并在测试里重算；校验 python 主次版本）。**建议在 Phase 4 依赖新 golden 之前处理。**
- M-2：搜索后的手牌顺序和显示边没有和 Python 对比（建议在生成器里每次搜索后导出 `export_state`，C# 比较完整规范化字符串）。**同上。**
- M-3：搜索 golden 偏向后期局面（85% 占用 ≥ 4；节点上限 5,000）；缺少永久的"开局 + 同类规则、双方手牌同 id、服务端风格原始 Card、474/475 与 1000+ id"的 golden（审查的对抗性生成器 V2 没有入库）。**同上。**
- L-1：`maxDepth > 128` 会 `IndexOutOfRangeException`（`SearchStats.MaxLevels`）；L-2：`default(SearchLimits)` = "已超时"，NaN/负时限未校验；L-3：把 "fail-hard" 注释/文档改为 fail-soft；L-4：`PythonStateKey` 的哈希注释引用了不存在的 `PythonStateKeyTests`（真实测试名 `Transposition_key_partition_equals_pythons_get_state_hash`），"2^-128" 说法偏乐观；L-5：`AllocationTests` 没有断言状态恢复；L-6：文档措辞过期（`.claude/phase_goal.md` 里的 "26 passed"、`CLAUDE.md` §7 的 26 基线应说明现为 29；`CLAUDE.md` 没有"人工决策协议"章节）；L-7：Q-2 决定前 Phase 7 的重复 id 输入会在 C# 抛异常；L-8：（已处理：现已提交并推送到分支）；L-9：`CardSnapshot.FromJson` 把 `"r"` 以外的 owner 都当蓝方；L-10：`MakeMove_rejects_a_full_undo_stack_…` 实际测的是"格子已占用"，undo 栈满分支没有直接测试；L-11：扩容瞬间新旧数组并存（≈ 1.5× 峰值内存）；L-12：服务端等价测试依赖墙钟（≈ 30 s，断言 ≥ 3 层且 < 20 s），在高负载 CI 上可能不稳定。
- O-1…O-6（只有在 CLAUDE.md §6 的 Baseline→…→Compare 流程下才可做）：每节点 `stackalloc` 清零、`IsTimeUp` 每节点读时钟、同类规则下每步快照 19 个修正值并全盘重算、`EvaluateState` 每个叶子全盘重算、`PythonStateKey` 每个内部节点构造 ≈ 220 字节缓冲区并哈希（**Fast 模式已用增量 Zobrist 取代**）、depth-1 节点也做完整排序（**Fast 已在 Phase 4 省略，经测量保留；compat 不变**）。

### 12.8 下一步行动（按顺序）

0. **（Phase 5 之后）** Phase 6 等待用户授权（可用 `/goal @.claude/phase_goal.md`，它会选中下一个未完成的阶段 = Phase 6；开始前先重读 §0 与 §7 Phase 6）。本节原来关于 Phase 5 的设计约束（精确评分有自己的 `EvalMode`、场景之间不共享表代、memo 键）已在 Phase 5 全部落实（§7 Phase 5）。
1. **Phase 6 的设计约束**（来自 Phase 3–5 的经验）：
   - 任何以性能为目的的改动都要走 CLAUDE.md §6（Baseline → Modify → Build → Test → Benchmark → Compare），一次一类改动，至少 3 次独立运行取中位数；不要引入未测量的 `unsafe`、SIMD、并行；root parallelism 需用户单独批准。
   - 现有的性能数据与线索：Fast 与 compat 的节点数差别不大（Phase 4：0.96），速度主要来自更便宜的键与跳过 depth-1 排序；精确求解器在 `mid3_ascension`（6 个空格）上 0.059 s / 241,264 次 make_move；robust 流水线 C# 约 0.08 s / 72 个局面，其中占用 5 的局面占大头（每个局面 3–7 ms）；Phase 5 的 `Select` 为简单起见每个场景克隆一次、每个 (走法, 场景) 做一次 Make/Undo，没有测量过其中的分配与克隆成本。
   - 零分配的热路径（Make/Undo、Evaluate、Order、AlphaBeta、TT、`ExactEndgameSolver`）必须保持，并由 `GC.GetAllocatedBytesForCurrentThread()` 测试断言。
   - 所有 parity 测试（78 个）与 golden 必须保持不变；Python 源码变动会让 golden 自动重新生成（M4-2）。
2. （建议，不强制）Phase 6 之前：对 Phase 5 做一次独立审查；处理 M4-3 的一部分（D-015 相关：把搜索后的手牌顺序纳入 golden）可以留到 Phase 7 之前。
3. **Phase 7 的注意事项**（记录在这里以免遗忘，Phase 7 尚未授权）：
   - D-015：robust 选择依赖搜索之后的手牌顺序；为了得到与 Python 相同的最终走法，标准搜索必须是 ReferenceCompat（复现 D-001），并把搜索后的 `GameState` 交给 `EndgameRobust.Select`；或者由用户决定一个规范手牌顺序。
   - 随机回退：`ScenarioBuild.RequiresFallback` 时由 Python 侧用 `handler.generate_opponent_cards` 采样并通过 `fallbackCandidates` 传入。
   - 请求解析（`parse_hand`/`parse_board`，含 D-011 左右互换）、未知牌采样、`ai_move` 的调度与文案仍在 Python（Phase 7）/ C#（Phase 8）。
4. 后续阶段顺序见 §0/§7：Phase 6（profiler 驱动优化）→ 7（Python 服务调用 C# 引擎，需先定 Q-2）→ 8（全量服务移植）→ 9（可选 CP-SAT）。

---

### 12.9 Phase 6 之后的补充（2026-10-10）

- **计时规则**：混合 CPU，所有计时必须固定到 P 核（`--affinity auto` / `pin.py`），记录掩码、warm/cold、≥3 次独立进程运行的 min/median/max。机器会漂移（同一代码一小时后 +6%），所以保留/回退只和**同窗口的基线构建**比较（`run_cs_pinned.py --exe <上一提交的 Bench 文件夹>`；该文件夹必须在仓库目录树内才能找到 `data/`，或用 `git worktree` 构建）。C# 单独跑：`python tools/bench/run_cs_pinned.py --modes search,fast,exact,endgame,makeundo --runs 5 --affinity auto [--cold] [--compare base.json] [--save out.json]`；profile：`python tools/bench/profile_summary.py --collect fast|compat|exact`。
- **确定性签名**：`Bench` 输出的 `SIG` 行（节点数、分数位、走法、exact 值、校验和、分配）在优化前后必须完全相同，`run_cs_pinned.py --compare` 会逐行检查。
- **golden**：重复 id（D-004）现在在 rules/search/endgame golden 里占 12.5%（robust 25%）；生成器版本 `py_rules_trace/4`、`py_search_trace/3`、`py_endgame_trace/2`；首次 `dotnet test` 会重新生成（约 8 分钟）。
- **C# 的重复 id 语义**：`FromSnapshot` 接受重复 id；机器人层按 Python 的"同 id 取手牌第一张"（D-019），在入口处（Make/Undo 重排手牌之前）解析。
- 未处理的审查遗留项不变（M4-3、L4-*、O4-*、Phase 3 的 L-*/O-*）；O-1（`SkipLocalsInit`，需要 `AllowUnsafeBlocks`）与 O-2（compat 的时钟读取，定深度运行不读时钟）按 `PHASE6_OPTIMIZATION.md` §3 的理由未做。

### 12.10 Phase 7 之后的补充（2026-10-10）

- **构建/运行引擎**：`dotnet build csharp/TTC.Siren.sln -c Release`；运行 `dotnet csharp/src/TTC.Siren.Engine/bin/Release/net10.0/TTC.Siren.Engine.dll --port 5300 --workers 8`（默认只监听 127.0.0.1:5300）；发布 `dotnet publish csharp/src/TTC.Siren.Engine -c Release -o publish/engine`（framework-dependent）；检查 `curl http://127.0.0.1:5300/v1/health`。Python 端：`TTC_CS_ENGINE=on|off`（默认 on）、`TTC_CS_ENGINE_URL` 等见 `docs/DEPLOY_WINDOWS.md` §6；`python tools/phase7/serve_flask.py --port 5001` 以无控制台面板的方式启动 Flask。
- **测试**：`dotnet test`（149 个；端到端比较需要 `tools/parity/golden/e2e_reference_g1_s20261010_n360.jsonl.gz`，缺失或源码哈希变化时由测试自动调用 `tools/phase7/gen_corpus.py` 与 `e2e_reference.py` 生成，约 1–2 分钟；golden 总体重新生成约 10–20 分钟）；`python -m pytest -q`（126 个；`test_engine_integration.py` 会启动真实引擎进程，需要 .NET 10）。
- **engine-off 等价**：`python tools/phase7/equivalence.py capture --root <旧提交的 worktree> --corpus … --out a.json`，在新代码上 `TTC_CS_ENGINE=off` 再跑一次，`compare a.json b.json`。
- **请求录制**：`TTC_RECORD_REQUESTS=<目录>`（默认关闭；会保存用户请求）；录制文件含请求 JSON 与 Python 随机数状态。
- **共享 C# 代码里 Phase 7 的改动**只有 L-1/L-2/L-9 与 `SearchStats` 的逐层累计（见 §7 Phase 7）；不要在引擎里依赖 `[ThreadStatic]` 的 memo（`EndgameRobust.Select` 的默认值仍是 Phase 6 的线程静态 memo，其他调用者会用到）。
- **本机测量不外推**：服务器（E5-2698B v3）上的 worker 数要用 `tools/phase7/run_load_matrix.py` 在服务器上测（`docs/DEPLOY_WINDOWS.md` §8）。

### 12.11 Phase 7 加固之后的补充（2026-10-10）

- **引擎是必需的**：minimax 路径只由 C# 引擎求解；没有引擎的机器必须设置 `TTC_CS_ENGINE=off`（人工开关，运行不变的 Python 求解器，约慢 100 倍），否则每个 minimax 请求返回 503 `connection_failed`。错误矩阵见 `docs/DEPLOY_WINDOWS.md` §6 与 `docs/reports/PHASE7_HARDENING.md` §3。Python 求解代码（`ai/ai.py`、`core/`、`_solve_minimax_python`、残局函数）仍是 parity 参考与 golden 生成器的依据，不要删除或改动。
- **启动**：`python app.py`（Flask 开发服务器，`threaded=True`，仍可用）；生产建议 `pip install waitress` 后 `python serve_waitress.py --host 127.0.0.1 --port 5000 --threads 16 [--no-console]`（与 `app.py` 相同的初始化；本机未安装 waitress，尚未用真实 waitress 运行）。引擎：`dotnet publish csharp/src/TTC.Siren.Engine -c Release -o publish/engine`；默认 `--queue 256 --queue-wait-ms 60000`，Server GC（非并发）；`/v1/health` 多了 `queueWaitMs`、`abandonedSkipped`、`served`、`serverGc`。
- **环境变量**：`TTC_CS_ENGINE`、`TTC_CS_ENGINE_URL`、`TTC_CS_ENGINE_CONNECT_TIMEOUT`、`TTC_CS_ENGINE_READ_MARGIN`、`TTC_CS_ENGINE_POOL_SIZE`（新）；`TTC_CS_ENGINE_BACKOFF` 已不存在（没有熔断器）。
- **测试**：`tests/engine_process.py` 是真实引擎进程/线程化 Flask/netstat 的测试辅助；`tests/test_engine_overload.py` 复现审查的过载与连接数探针；`tests/test_csharp_engine_pool.py`、`tests/test_serve_waitress.py`；`tools/phase7/gc_compare.py` + `gc_table.py` 做 GC 对比。改动 `ai_server.py`、`ai/csharp_engine.py` 或 `tools/phase7/e2e_reference.py` 会使 e2e golden（以及依赖 `ai_server.py` 的 endgame golden）失效，下一次 `dotnet test` 自动重新生成（e2e 约 1–2 分钟）。
- **计时**：绝对延迟随会话漂移（L7-2），只有同一会话内的比较有意义。

---

### 12.12 Phase 8A 之后的补充（2026-10-11）

- **启动**：`dotnet run --project csharp/src/TTC.Siren.Server -c Release`（配置 `appsettings.json`/环境变量；默认只监听 127.0.0.1）；密钥：`TTC.Siren.Server keys add|list|revoke|set-limit`；详见 `docs/DEPLOY_WINDOWS.md`、`API.md` §5。
- **测试数据**：8A 的 golden 由测试按需生成（`tools/phase8a/golden_sources8.py` 绑定 Python 源码 SHA-256；不提交）；语料在 `tools/phase8a/corpus/`（被 `.gitignore` 忽略）。解释快照 `tests/data/explanation_snapshots_v2.json` 已提交。
- **Python 业务代码未改**，仍是 reference 与 golden 生成器；`/api/*` 由服务器代理到 Python。
- **未做**：8B/8C/8D；robust 残局的细粒度进度（需改 `EndgameRobust`，未授权）。

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
