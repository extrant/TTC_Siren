# CLAUDE.md — TTC_Siren C# 迁移期间的长期规则

本文件是整个 Python → C# 迁移期间 **不可违反** 的工作规范。每个 Claude 会话开始时必须先读本文件，再读 `CSHARP_MIGRATION.md`。

- `CSHARP_MIGRATION.md` 是迁移进度的 **唯一事实来源**（架构、baseline、Phase 状态、差异、BLOCKED、下一步）。
- 任何阶段开始前，先确认 `CSHARP_MIGRATION.md` 中该 Phase 的范围与验收条件；任何阶段结束后，必须更新它。
- 用户每次只授权一个 Phase。未经用户明确指示，不得进入下一 Phase。

---

## 1. 正确性优先

1. **正确性优先于性能。** 任何性能改动都不得改变搜索结果语义（见 §4）。
2. **Python 是当前的 reference implementation。** 规则结算、评估函数、走法排序、搜索流程，以 `core/`、`ai/ai.py`、`ai_server.py` 的实际行为为准，而不是以 README、注释、游戏官方规则或"应该如何"为准。
3. Python 中已知的怪异行为/缺陷（见 `CSHARP_MIGRATION.md` 的 *Semantic Discrepancy* 表）在用户决定修复前，**C# 必须按 Python 实际行为复现**，并在代码中用 `// PY-QUIRK(Dxxx)` 注释标出位置。
4. 修复 Python 缺陷属于语义变更，必须：先由用户批准 → 同时修改 Python 与 C# → 补回归测试 → 在差异表中把状态改为 `FIXED-BOTH`。

## 2. 遇到无法解释的差异：BLOCKED，而不是猜测

- Python/C# 行为出现 **无法解释** 的差异时，必须停下，把该项写入 `CSHARP_MIGRATION.md` 的 BLOCKED 表（复现输入、Python 输出、C# 输出、已排除的假设），并向用户报告。
- 禁止通过"调参数直到对上"、放宽断言、改用近似比较来掩盖差异。
- 浮点分数比较一律按 **IEEE-754 double 位级相等**（Python `float.hex()` / C# `BitConverter.DoubleToInt64Bits`）。不得使用 epsilon 比较来宣称 parity。

## 3. 禁止机械逐行翻译

- 不得把 Python 的对象模型（`Card` 对象列表、`dict` 字符串键、`copy.deepcopy`、`SearchResult` 对象、`path + [move]`）原样搬到 C#。
- C# 目标是 **数据导向** 的实现：固定数组、`byte/sbyte` 索引、bitmask、`readonly struct`、预计算邻接表、undo 栈。
- "语义等价" 与 "结构等价" 是两回事：必须语义等价，不要求结构等价。

## 4. 不得为了性能改变游戏规则语义

以下内容在任何优化中都必须保持与 Python 完全一致（除非用户批准的语义修复）：
- 基础翻面、同数、加算、连携（仅同数/加算翻得的卡触发一层基础比较连携）、逆转、王牌杀手、同类强化/弱化（含手牌受影响、同时存在时强化优先）、秩序/混乱（`can_use` 过滤）。
- `evaluate_state` / `evaluate_move` 的每一项系数、累加顺序与 `min(…,1000.0)` 截断。
- 走法生成顺序（手牌当前顺序 × 行优先空格）与 **稳定** 排序（降序且相等元素保持原顺序；C# 的 `Array.Sort` 不稳定，禁止直接使用）。
- 手牌顺序在 make/undo 后的变化（undo 把卡追加到末尾）。
- 终局时仍使用启发式 `evaluate_state`（不是胜负分）。

## 5. 搜索架构原则

- 优先采用 **GenerateMoves → MakeMove → AlphaBeta → UndoMove**；搜索热路径内禁止复制 GameState。
- 搜索热路径（MakeMove/UndoMove/Evaluate/OrderMoves/AlphaBeta/TT probe）**稳态零托管分配**：不 `new` 对象、不用 LINQ、不用 lambda 捕获、不 boxing、不用 `string`、不用 `List<T>` 扩容、不用 `Dictionary` 作为 TT。
- 使用 `GC.GetAllocatedBytesForCurrentThread()` 在测试中断言热路径分配为 0。
- 先完成 **单线程** 高性能搜索并证明正确，再考虑 root parallelism / Lazy SMP。
- 搜索必须支持 "无时间限制的固定深度" 模式，用于确定性 parity 测试。

## 6. 性能修改流程（强制）

每一次以性能为目的的修改都必须走完：

**Baseline → Modify → Build → Test → Benchmark → Compare**

1. Baseline：在修改前用同一 benchmark 集记录数据（节点数、耗时、NPS、结果分数与走法）。
2. Modify：一次只做一类修改。
3. Build：`dotnet build -c Release`，警告视为错误。
4. Test：全部单元测试 + parity 测试必须通过。
5. Benchmark：同机器、Release、同输入，至少 3 次取中位数。
6. Compare：结果（分数/走法/固定深度节点数）必须不变或差异已解释；性能数据写入 `CSHARP_MIGRATION.md` 的 benchmark 历史。

禁止未经 benchmark 的"想当然"优化（例如未测量就引入 unsafe、SIMD、并行、对象池）。没有数据的性能结论不得写入文档。

## 7. 测试纪律

- 不允许跳过、删除、`[Skip]`、注释掉失败测试，也不允许修改断言来让测试通过（除非该断言本身被证明错误，并记录原因）。
- 现有 Python 测试（`python -m pytest -q`）必须始终保持通过；Phase 1 baseline 为 26 passed。
- C# 每个 Phase 都必须有 Python 生成的 golden 数据驱动的 differential 测试。
- 随机测试必须固定种子并把种子写入失败信息，保证可复现。

## 8. Git 与发布

- 不允许未经用户授权的 `git push`、`git push --force`、改写已推送历史、打 tag、发布 release、上传构建产物。
- 不得修改 `main` 分支；迁移工作在 `csharp-solver-migration` 分支进行。
- 未经用户要求不主动 commit；用户要求 commit 时，每个 commit 只包含一个逻辑变更。
- 不得删除或改写用户的 Python 代码，除非用户批准的语义修复（§1.4）。

## 9. CP-SAT 的定位

- CP-SAT（OR-Tools）将来 **只** 用于：隐藏牌组合/可能世界生成、卡组约束（★4+ ≤2、★5 ≤1、选拔星级配额）、deck optimization、scenario generation。
- CP-SAT **不取代** Minimax / Alpha-Beta / 精确残局搜索。

## 10. 目录约定（C# 部分，Phase 2 起生效）

```
csharp/                      C# 解决方案根目录
  src/TTC.Siren.Core/        卡牌数据、规则、GameState、MakeMove/UndoMove
  src/TTC.Siren.Search/      评估、排序、Minimax/AlphaBeta/ID、TT、残局
  src/TTC.Siren.Bench/       benchmark 控制台
  src/TTC.Siren.Engine/      引擎库 + 本机 HTTP 引擎服务（Phase 7；ASP.NET Core，只监听 127.0.0.1；工作线程池与求解服务 `SolveService` 在此，Server 项目在进程内复用；A-20：出错不再回退 Python）
  src/TTC.Siren.Compat/      Phase 8A：与 Python 兼容的请求层——CPython 兼容随机数 `PythonRandom`（MT19937）、Python 对象模型（`Legacy*`，只用于请求期兼容层，不在搜索热路径）、请求解析、未知牌采样（`UnknownCardHandler`）、旧版分析文案（`LegacyAnalysis`）、`PyJson`
  src/TTC.Siren.Explain/     Phase 8A：解释事实 v2（翻牌原因、暴露、主变例、备选、残局、未知牌、可信度）与中文模板文案（模板是数据文件）
  src/TTC.Siren.Server/      Phase 8A：产品服务器（ASP.NET Core）：原生 `/ai_move`、对局日志、实时事件（SSE）、API 密钥/限流/HTTPS/CORS、`/api/*` 代理、占位页
  tests/TTC.Siren.Tests/     xUnit 单元测试 + parity 测试
tools/parity/                Python 侧 golden 数据生成脚本（只读调用 reference 实现）
tools/phase7/                Phase 7 的语料、端到端参考、延迟与压测脚本
tools/phase8a/               Phase 8A 的 golden 生成器（RNG、解析、采样、端到端、数字格式）、请求语料生成器、HTTP 延迟脚本
```

- Python reference 代码不因 C# 迁移而改动；parity 工具只 import Python 模块，不修改它们。
- 运行环境：Windows 11，.NET SDK 10.0.401（目标 `net10.0`），CPython 3.11.4。
