# Triple Triad AI 统一服务 — 前端对接 API 文档

**Base URL:** `http://127.0.0.1:5000`

所有端点均为 `POST`，请求/响应均为 `application/json`。以下用 `Card` 表示统一的卡牌对象，`DeckResult` 表示分析结果对象——这两个结构在所有端点中保持一致。

---

## 1. 通用数据结构

### 1.1 Card（卡牌）

所有卡牌 JSON 统一为此结构：

```json
{
  "id":   310,       // int   — 卡牌数据库序号
  "up":   6,         // int   — 上边数值 (1~10)
  "right": 8,        // int   — 右边数值 (1~10)
  "down":  5,        // int   — 下边数值 (1~10)
  "left":  3,        // int   — 左边数值 (1~10)
  "star":  3,        // int   — 星级 (1~5)
  "type":  "蛮神"    // str   — 类型名，"无类型" 或 "" 表示无类型
}
```

### 1.2 DeckResult（分析结果）

推荐/选拔分析端点统一返回此结构：

```json
{
  "deck": [Card, Card, Card, Card, Card],   // 推荐的 5 张卡组
  "final_score": 64.32,                      // float — 综合评分 (0~100)
  "components": {                            // 各维度分数
    "base":          49.10,                  // 基础强度
    "rule":          66.80,                  // 规则适配
    "synergy":       92.00,                  // 组合协同
    "stability":     54.88,                  // 稳定性
    "verification":  33.28,                  // 深算验证
    "risk":          0.00                    // 风险惩罚（越低越好）
  },
  "reasons": [                               // str[] — 推荐理由
    "有多组常见加算和",
    "四向都有高边覆盖"
  ],
  "warnings": []                             // str[] — 风险提示
}
```

> **注意：** NPC 推荐 (`/api/recommend`) 的 `components` 字段略有不同，使用 `heuristic` / `simulation_average` / `simulation_floor` / `simulation_min` / `games`。

### 1.3 规则取值

| 值 | 规则名 |
|----|--------|
| `"同数"` | 同数 |
| `"加算"` | 加算 |
| `"逆转"` | 逆转 |
| `"王牌杀手"` | 王牌杀手 |
| `"同类强化"` | 同类强化 |
| `"同类弱化"` | 同类弱化 |
| `"秩序"` | 秩序 |
| `"混乱"` | 混乱 |
| `"选拔"` | 选拔 |

---

## 2. 端点详解

### 2.1 POST /ai_move — 求解器走子

请求 AI 在当前局面下做出最优落子决策。

**请求 JSON：**

```json
{
  "board": [
    {
      "pos":  [0, 0],       // [int, int]     — 行(0-2), 列(0-2)
      "numU": 6,             // int            — 上边数值
      "numL": 5,             // int            — 左边数值
      "numD": 4,             // int            — 下边数值
      "numR": 3,             // int            — 右边数值
      "owner": 1             // int            — 1=蓝方, 2=红方
    }
  ],
  "myHand": [
    {"numU": 8, "numR": 4, "numD": 7, "numL": 2, "canUse": true, "inHand": true},
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0, "canUse": false, "inHand": false}
  ],
  "oppHand": [
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0},
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0},
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0},
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0},
    {"numU": 0, "numR": 0, "numD": 0, "numL": 0}
  ],
  "npcIds": [44, 45, 46, 47, 48, 49, 50],
  "npcName": "示例 NPC",
  "myOwner": 1,              // int            — 1=蓝方, 2=红方
  "currentPlayer": 1,        // int            — 1=蓝方回合, 2=红方回合, 0=未知
  "rules": "全明牌,加算,同数", // str           — 逗号分隔规则；"全明牌"/"三明牌" 控制明牌模式
  "solver": "minimax",       // str            — "minimax"；"monte_carlo" 已下线，仍可传，但由 minimax 引擎回答（见下）
  "max_depth": 10,           // int            — Minimax 最大思考深度
  "max_time": 10,            // float          — Minimax 最大思考秒数
  "mc_simulations": 150,     // int            — 已忽略（蒙特卡洛已下线，仅为兼容旧客户端保留）
  "show_search_progress": false  // bool       — 是否返回搜索进度详情
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `board` | Object[] | 是 | 棋盘上已放置的卡牌，空棋盘传 `[]` |
| `myHand` | Object[] | 是 | 我的当前手牌，已知卡牌填四边数值，未知填全0；`inHand=false` 表示已知但已打出，不可作为候选 |
| `oppHand` | Object[] | 是 | 对手手牌，通常全为未知（全0）；明牌模式下填已知数值；`inHand=false` 表示已知但已打出 |
| `npcIds` | int[] | 否 | NPC 对局时传当前 NPC 已知牌库 ID；后端优先从此牌库约束 `oppHand` 中未知槽位的候选卡 |
| `npcName` | str | 否 | 当前 NPC 名称，仅用于日志和调试 |
| `myOwner` | int | 是 | `1`=蓝方, `2`=红方 |
| `currentPlayer` | int | 否 | 当前回合：`1`=蓝方, `2`=红方, `0`=自动推断 |
| `rules` | str | 否 | 逗号分隔规则，"全明牌"/"三明牌" 控制对手手牌可见度 |
| `solver` | str | 否 | 求解器类型，默认 `"minimax"`。`"monte_carlo"` 已下线（Phase 8A，决定 A-22）：请求照常被接受，由 minimax 引擎（含精确/信息集残局）回答，响应追加 `solver_used`、`solver_note` |
| `max_depth` | int | 否 | Minimax 最大思考深度，默认 10 |
| `max_time` | float | 否 | Minimax 最大思考秒数，默认 10 |
| `mc_simulations` | int | 否 | 已忽略（蒙特卡洛已下线） |
| `show_search_progress` | bool | 否 | 默认 false，开启后在响应中附加搜索过程数据 |

> **棋盘位置约定：** `pos: [行, 列]`，`[0, 0]` 为左上角，`[2, 2]` 为右下角。
>
> **手牌卡牌格式：** 已知卡牌传入四边数值（`numU`=上, `numR`=右, `numD`=下, `numL`=左），系统自动从数据库匹配卡牌 ID；未知卡牌四边全填 `0`，AI 内部会进行行为建模推断。`canUse=false` 表示当前规则下不可点击，`inHand=false` 表示该槽位只是历史记忆，不能进入可落子候选。
>
> **注意 `numL`/`numR` 命名：** `numL` 对应左值，`numR` 对应右值。在棋盘卡牌对象中也用同样命名。早期客户端可能使用不同命名，请以本文档为准。

**响应 JSON：**

```json
{
  "card": "U8 R4 D7 L2 星级:3",
  "card_id": 310,
  "pos": [1, 1],
  "opponent_hand_analysis": {
    "predicted_cards": [
      {"card": "重复数值卡牌", "confidence": 0.7, "reasoning": "同数规则下偏好设置连携陷阱"}
    ],
    "total_unknown": 5,
    "strategy_analysis": "对手策略：严格控制星级配额..."
  },
  "win_probability": {
    "current": 0.450,
    "after_move": 0.620,
    "confidence": 0.78
  },
  "recommendation": {
    "move_reasoning": "占据角落位置，具有防御优势",
    "strategic_value": "含1或A可打王牌杀手；控制中心战略位置",
    "alternative_moves": [
      {
        "card": "U6 R8 D5 L3 星级:3",
        "pos": [0, 2],
        "value": 2.350
      }
    ]
  },
  "performance_stats": {
    "nodes_searched": 15234,
    "search_depth": 8,
    "tt_hit_rate": 42.5,
    "cutoff_rate": 38.2,
    "unknown_cards_processed": 5,
    "performance_optimizations_active": true
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `card` | str | 人类可读的卡牌描述 |
| `card_id` | int | 选中的卡牌数据库 ID |
| `pos` | [int, int] | 落子位置 `[行, 列]` |
| `opponent_hand_analysis` | Object | 对手手牌分析（行为建模推断） |
| `win_probability` | Object | `current`=当前胜率, `after_move`=落子后胜率, `confidence`=置信度 |
| `recommendation` | Object | 走法理由 + 3 个备选方案 |
| `performance_stats` | Object | 搜索性能统计 |
| `engine` / `engine_info` | str / Object | 恒为 `"csharp"`；`engine_info` 含 `protocol`、`worker`、`robust`（是否运行了信息集残局）、`overridden`（残局层是否改了走法）、`scenarios`、`timings` |
| `search_progress` / `search_summary` | Object[] / Object | 仅当请求 `show_search_progress` 为真时出现：每完成一层一条（`depth`、`max_depth`、`best_move`、`best_score`、`nodes_searched`、`time_elapsed`、`time_remaining`、`nodes_per_second`、`tt_hit_rate`、`cutoff_rate`、`branching_factor`）与汇总 |
| `solver_used` / `solver_note` | str | **追加字段**：仅当请求的 `solver` 为 `"monte_carlo"` 时出现；`solver_used` 为 `"minimax"`，`solver_note` 是中文说明 |

> **兼容性约定（Phase 8A）：** 以上所有字段的键名、取值类型与含义都与 Python 版 `ai_server` 在引擎路径上的响应一致（数字按 Python 的浮点写法输出，例如 `1.0`、`0.45`）；新版只会 **追加** 字段，不会改名或删除。无合法走法时返回 `{"move": null, "msg": "无可用动作", "engine": "csharp"}`（HTTP 200）。

---

### 2.2 POST /api/recommend — NPC 卡组推荐

玩家拥有上百张卡牌，面对某个 NPC（已知其牌库），系统根据规则推荐最优的 5 张上场卡组。

**请求 JSON：**

```json
{
  "ownedIds": [1, 2, 3, 5, 8, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35,
               38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70,
               72, 75, 78, 80, 82, 85],
  "npcIds":   [44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
  "rules":    ["加算", "同数"],
  "topN":     5
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ownedIds` | int[] | 是 | 玩家拥有的卡牌 ID 列表（可上百张） |
| `npcIds` | int[] | 是 | NPC 已知的卡牌 ID 列表（≥5 张时系统从中采样可能上场组合） |
| `rules` | str[] 或 str | 否 | 特殊规则列表或逗号分隔字符串，默认为空 |
| `topN` | int | 否 | 返回推荐数量，默认 5 |

**响应 JSON：**

```json
{
  "process": [
    "玩家持有牌：输入35个，命中35张",
    "NPC牌库：输入10个，命中10张",
    "玩家侧应用限制：★4以上≤2，★5≤1",
    "NPC侧不应用玩家选卡限制，只从已知牌库选择5张上场牌",
    "规则：加算",
    "NPC候选上场组合：10组",
    "玩家合法候选卡组：48组进入模拟",
    "最高分：64.32",
    "推荐理由：加算常见和较多",
    "推荐理由：对10组NPC可能上场牌模拟20局",
    "推荐理由：低分位仍保持不落后"
  ],
  "npc_decks": [
    [Card, Card, Card, Card, Card],
    [Card, Card, Card, Card, Card],
    [Card, Card, Card, Card, Card]
  ],
  "best": {
    "deck": [Card, Card, Card, Card, Card],
    "final_score": 64.32,
    "components": {
      "heuristic":           59.63,
      "simulation_average":  72.91,
      "simulation_floor":    55.48,
      "simulation_min":      10.53,
      "games":               20
    },
    "reasons": [
      "加算常见和较多",
      "对10组NPC可能上场牌模拟20局",
      "低分位仍保持不落后"
    ],
    "warnings": []
  },
  "results": [
    { /* DeckResult — 第1推荐 */ },
    { /* DeckResult — 第2推荐 */ },
    { /* DeckResult — 第3推荐 */ }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `process` | str[] | 分析过程日志（前端可展示） |
| `npc_decks` | Card[][] | 系统推断的 NPC 最可能上场组合（最多 3 组） |
| `best` | DeckResult | 最优推荐 |
| `results` | DeckResult[] | 排名推荐列表（长度 = topN） |

> **NPC 推荐的 `components` 说明：**
>
> | 子字段 | 说明 |
> |--------|------|
> | `heuristic` | 启发式评分（卡牌强度 + 规则适配 + 对位优势） |
> | `simulation_average` | 模拟对局平均胜率 |
> | `simulation_floor` | 模拟对局 P20 下限（最差 20% 的平均值） |
> | `simulation_min` | 模拟对局最低胜率 |
> | `games` | 模拟总局数 |
>
> 最终综合分 = `heuristic × 0.45 + simulation_average × 0.40 + simulation_floor × 0.15`

---

### 2.3 POST /api/draft/analyze — 选拔深算分析

对候选卡组在「选拔 + 可选额外规则」下进行深算评分，返回最优解。除固定卡组和卡池枚举外，也支持 UI 选拔结构：备选1/2 各选一组双卡，备选3 选一张单卡，后端会展开为完整 5 张卡组后评分。

**请求 JSON：**

```json
{
  "mode": "decks",
  "candidates": [
    [Card, Card, Card, Card, Card],
    [Card, Card, Card, Card, Card],
    [Card, Card, Card, Card, Card]
  ],
  "extraRules": ["加算"],
  "weights": {
    "base":         0.18,
    "rule":         0.24,
    "synergy":      0.16,
    "stability":    0.10,
    "verification": 0.32,
    "risk":         0.12
  },
  "timeBudget": 7.0
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `mode` | str | 否 | `"decks"` = 固定卡组评分（默认）；`"pool"` = 从卡池中枚举 5 张组合；`"selection"` = 三段 UI 选拔候选 |
| `rules` | str[] | 否 | 完整规则列表，如 `["选拔", "加算"]`；不传时也可继续用 `extraRules` |
| `candidates` / `groups` | DraftStage[] | 是 | selection 模式传三段候选；也可用 `draftChoices` 或 `choices` 代替 |
| `extraRules` | str[] | 否 | 额外规则，默认无。自动叠加「选拔」，最多额外 2 条 |
| `weights` | Object | 否 | 各维度权重，不传使用默认值（见下表） |
| `timeBudget` | float | 否 | 深算秒数预算，默认 7.0，范围 5~20 |

**selection 模式传参：**

前端可先调用独立分析器的随机接口获取候选：

```http
POST /api/random
Content-Type: application/json

{}
```

随机接口响应：

```json
{
  "mode": "selection",
  "rules": ["选拔"],
  "candidates": [
    [
      {"left": Card, "right": Card},
      {"left": Card, "right": Card},
      {"left": Card, "right": Card}
    ],
    [
      {"left": Card, "right": Card},
      {"left": Card, "right": Card},
      {"left": Card, "right": Card}
    ],
    [Card, Card, Card]
  ],
  "constraints": {
    "stage1": "3组选1，每组2张卡",
    "stage2": "3组选1，每组2张卡",
    "stage3": "3张选1",
    "expandedDecks": 27,
    "deckSize": 5,
    "starLimit": "★4以上≤2，★5≤1"
  }
}
```

然后将 `candidates` 原样传给分析接口：

```json
{
  "mode": "selection",
  "rules": ["选拔", "加算"],
  "groups": [
    [
      {"left": {"id": "node11L", "up": 7, "right": 8, "down": 1, "left": 6}, "right": {"id": "node11R", "up": 3, "right": 7, "down": 8, "left": 10}},
      {"left": {"id": "node12L", "up": 7, "right": 4, "down": 4, "left": 8}, "right": {"id": "node12R", "up": 7, "right": 9, "down": 1, "left": 10}},
      {"left": {"id": "node13L", "up": 8, "right": 6, "down": 7, "left": 1}, "right": {"id": "node13R", "up": 4, "right": 8, "down": 10, "left": 6}}
    ],
    [
      {"left": {"id": "node25L", "up": 4, "right": 3, "down": 7, "left": 3}, "right": {"id": "node25R", "up": 7, "right": 7, "down": 9, "left": 4}},
      {"left": {"id": "node26L", "up": 7, "right": 5, "down": 5, "left": 3}, "right": {"id": "node26R", "up": 6, "right": 8, "down": 4, "left": 7}},
      {"left": {"id": "node27L", "up": 4, "right": 1, "down": 6, "left": 7}, "right": {"id": "node27R", "up": 7, "right": 9, "down": 8, "left": 1}}
    ],
    [
      {"id": "node39", "up": 4, "right": 3, "down": 2, "left": 3},
      {"id": "node40", "up": 4, "right": 4, "down": 3, "left": 4},
      {"id": "node41", "up": 2, "right": 5, "down": 2, "left": 5}
    ]
  ],
  "timeBudget": 7.0
}
```

`备选1` / `备选2` 也可以直接传 6 张卡的扁平列表，后端会按顺序每两张分成一组；`备选3` 可以传 3 张卡的列表，每张视为一组。

`Card` 也可以传 `original`/`raw` 四元组，后端会按 UI 原始顺序转换为 `上/右/下/左 = 原始第4/第1/第3/第2`；例如 `"original": "7,A,8,3"` 会解析为 `up=3,right=7,down=8,left=10`。

**默认权重：**

| 维度 | 默认值 | 说明 |
|------|--------|------|
| `base` | 0.18 | 基础强度（卡牌面板数值） |
| `rule` | 0.24 | 规则适配（特殊规则下的卡牌强度） |
| `synergy` | 0.16 | 组合协同（卡牌之间的配合度） |
| `stability` | 0.10 | 稳定性（开局和手牌顺序的稳定程度） |
| `verification` | 0.32 | 深算验证（蒙特卡洛模拟对局结果） |
| `risk` | 0.12 | 风险惩罚（扣分项，越小越好） |

**响应 JSON：**

```json
{
  "process": [
    "规则：选拔, 加算",
    "权重：{'base': 0.18, 'rule': 0.24, ...}",
    "深算预算：7.0秒",
    "对手采样：约2/3随机合法选拔牌组 + 1/3高威胁合法选拔牌组",
    "选拔段：3段",
    "展开完整卡组：27套",
    "合法完整卡组：27套",
    "步进深算：第一段全量3.0秒，第二段TopK追加4.0秒",
    "第一段全量深算：6个进程，27套，每套约0.60秒，共7300局",
    "第二段追加深算：第一段前8套进入追加验证",
    "第二段有界多进程：6个进程，每套约2.00秒",
    "排序策略：27套均已深算，TopK候选使用两段合并样本",
    "实际耗时：8.0秒",
    "最高分：55.73",
    "推荐选择：备选1-候选2 + 备选2-候选1 + 备选3-候选3",
    "推荐理由：有多组常见加算和",
    "推荐理由：牌组加算组合密度高"
  ],
  "recommendation": [
    {
      "stage": 1,
      "group": 2,
      "label": "备选1-候选2",
      "cards": [
        {"id": 12, "up": 7, "right": 4, "down": 4, "left": 8, "star": 1, "type": "无类型"},
        {"id": 13, "up": 7, "right": 9, "down": 1, "left": 10, "star": 1, "type": "无类型"}
      ]
    },
    {"stage": 2, "group": 1, "label": "备选2-候选1", "cards": [Card, Card]},
    {"stage": 3, "group": 3, "label": "备选3-候选3", "cards": [Card]}
  ],
  "best": {
    "deck": [Card, Card, Card, Card, Card],
    "final_score": 55.73,
    "components": {
      "base":         49.10,
      "rule":         66.80,
      "synergy":      92.00,
      "stability":    54.88,
      "verification": 33.28,
      "risk":         0.00
    },
    "reasons": [
      "有多组常见加算和",
      "四向都有高边覆盖",
      "牌组加算组合密度高",
      "深算830局，威胁采样138轮，均值48.01，P10下限5.92"
    ],
    "warnings": []
    "groups": [ /* 与 recommendation 相同 */ ]
  },
  "results": [
    { /* DeckResult — 第1名 */ },
    { /* DeckResult — 第2名 */ },
    { /* DeckResult — 第3名 */ }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `process` | str[] | 分析过程日志 |
| `best` | DeckResult | 评分最高的卡组 |
| `results` | DeckResult[] | 所有卡组按评分降序排列；selection 模式下每项额外包含 `selection`，如 `["备选1-候选2", "备选2-候选1", "备选3-候选3"]` |

> **并行深算：** 当 `mode="decks"` 且候选卡组为 2~3 套时，系统使用多进程并行模拟，每套独立分配时间预算，大幅缩短总耗时。

---

## 3. 典型前端调用流程

### 流程 A：对局求解

```
POST /ai_move  →  返回 {card_id, pos, ...}
    ↓ 前端执行落子动画
    ↓ 对手行动后更新 board / myHand / oppHand
POST /ai_move  →  循环
```

### 流程 B：NPC 对战前卡组推荐

```
前端展示 NPC 信息 → 玩家选择出战卡牌（或自动全部选择）
    ↓
POST /api/recommend  {ownedIds: [...], npcIds: [...], rules: [...]}
    ↓
前端展示推荐结果（best + results 列表），玩家点击确认出战
```

### 流程 C：选拔三选一

游戏内已随机出 3 套卡组供玩家选择。前端直接将这 3 套卡组传入分析：

```
游戏内展示 3 套卡组 → 玩家浏览，前端将 3 套卡组数据准备好
    ↓
POST /api/draft/analyze  {mode: "decks", candidates: [deck1, deck2, deck3], extraRules: [...]}
    ↓
前端展示最优推荐（best + results），辅助玩家决策
```

---

## 4. 错误处理

所有端点统一错误格式：

```json
{
  "error": "Invalid input data"
}
```

HTTP 状态码：
- `200` — 成功
- `400` — 请求参数不合法
- `401` — 云端服务器：没有 API 密钥或密钥无效（见 §5.4）
- `413` — 请求体超过 1 MiB（`{"error": "Request body too large (limit 1048576 bytes)"}`；Phase 8B 之前返回 400）
- `403` — API 密钥已被吊销
- `429` — 超过该密钥的速率或并发限制，见响应头 `Retry-After`
- `500` — 服务器内部错误
- `502` / `504` — `/api/*`：后端 Python 服务不可达 / 超时
- `503` — 求解引擎繁忙或拒绝了局面（`engine_error`）

引擎相关错误（A-20：不再回退到 Python 求解）：`{"error": "C# engine: <原因>(<说明>)", "engine": "csharp", "engine_error": "<原因>"}`，HTTP 503（引擎繁忙 `busy`、请求被引擎拒绝 `invalid_request`、引擎内部错误 `engine_error` 等）；`max_depth`/`max_time` 不是数字 → HTTP 400，`engine_error: "unsupported_parameter"`。请求体不是合法 JSON → HTTP 400 `Invalid input data`（Python 版此时返回 500，见 `CSHARP_MIGRATION.md` D-020）。

常见错误场景：

| 场景 | 响应 |
|------|------|
| 棋盘/手牌/owner 缺失 | `400` — `"Invalid input data"` |
| 卡牌数值不在数据库中 | `500` — `"Board card not found..."` / `"Hand card not found..."` |
| 玩家可用牌 < 5 张 | `/api/recommend` 返回空 results + process 说明 |
| NPC 已知牌 < 5 张 | `/api/recommend` 返回空 results + process 说明 |
| 无合法卡组 | `/api/draft/analyze` 返回空 results + process 说明 |

---

## 5. Phase 8A 新增（云端服务器；全部为追加，旧插件无需改动）

### 5.1 `/ai_move` 的新请求参数与响应字段

| 位置 | 字段 | 说明 |
|---|---|---|
| 请求 | `gameId` | 可选。字符串，`[A-Za-z0-9_-]`，1–64 位。同一局的所有请求带同一个 id，服务器就把它们归为一局（对局日志、实时事件、复盘用）。不带时服务器按"后继局面"自动推断（同密钥、同规则与颜色、轮到同一方、棋盘上原有的牌都在原位、间隔不超过 15 分钟）；无效的 id 被忽略 |
| 请求 | `explain` | 可选。`"brief"`（默认）或 `"full"`（额外给出备选走法的评分与劣势原因） |
| 响应 | `explanation` | `{version: 2, level, language, facts: [...], text: {brief, full}}`。`facts` 是可复算的结构化事实（立即效果与翻牌原因、暴露与威胁、对手最强回应、主变例与预期结果、备选走法、残局信息集、未知牌假设、可信度）；`text.brief` 是一两句话，`text.full` 是分段详述，文字只陈述事实里有的内容。字段说明见 `docs/design/EXPLANATION_FACTS.md`。生成失败时改为 `explanation_error`（走法不受影响） |
| 响应 | `game_id` | 本次请求归属的对局 id |
| 响应 | `engine_info.timings.explainMs` | 生成解释用时 |

### 5.2 对局日志

`GET /v1/games?offset=&limit=`（列表）、`GET /v1/games/{id}`（JSON：全部记录）、`GET /v1/games/{id}/download`（JSON Lines 下载）、`DELETE /v1/games/{id}`。每个密钥只能看到自己的对局（别人的对局返回 404）。记录格式见 `docs/design/GAME_LOG_SCHEMA.md`；默认保留 30 天。

**Phase 8B（日志格式版本 2）：** 整数永远写成整数（`4` 而不是 `4.0`）；`request_raw` 是服务器收到的请求原文（成员顺序、数字写法都保持原样，只去掉词法单元之间的空白），所以紧凑的请求体下载回来逐字节相同；`GET /v1/games/{id}` 返回的就是日志文件的各行，不再重新序列化，`schema` 是其中最高的 `v`（版本 1 的旧文件照常读取）。推断出的对手牌（`opponent_move`）多了 `ambiguous` 与 `candidate_ids`：请求里只有四条边没有卡 id，数据库中边相同的卡不止一张时 `ambiguous` 为 `true`，`candidate_ids` 列出全部候选（按数据库顺序），`card_id` 是其中第一张（与 Python 的 `find_card_id_by_stats` 一致）；界面应显示 `card_edges` 而不是只显示 id。

### 5.3 实时事件流

`GET /v1/live[?game_id=<id>]`，Server-Sent Events，只推送该密钥自己的事件：`request.received`（收到请求，含局面）、`search.depth`（每完成一层：深度、评分、最佳走法、节点数、用时）、`robust.progress`（残局信息集开始/结束）、`decision`（走法与简要解释）、`game.updated`（写入日志的记录）、`game.ended`。落后的客户端会丢弃最旧的事件并收到 `lagged` 通知（含丢弃数量），服务器不会因慢客户端而变慢。每 15 秒有一条心跳注释。

### 5.4 云端服务器：认证、限流、HTTPS

- **地址：** 云端部署后基地址是 `https://<你的域名>`（不再是 `http://127.0.0.1:5000`）；证书由服务器配置。
- **API 密钥：** 每个用户一个密钥，由运维用 `TTC.Siren.Server keys add --name <用户>` 生成，**只显示一次**。请求头二选一：`Authorization: Bearer <密钥>` 或 `X-Api-Key: <密钥>`。没有密钥或密钥不对 → `401`（响应头 `WWW-Authenticate: Bearer`）；密钥已吊销 → `403`。`/health` 与首页 `/` 不需要密钥。
- **限流：** 每个密钥一个令牌桶（每分钟速率 + 突发量）和最大并发请求数；超过时 `429`，body `{"error": "...", "retry_after_seconds": n}`，响应头 `Retry-After: n`。`/v1/live` 另有每个密钥的同时打开流数限制。
- **`/api/recommend`、`/api/draft/analyze`、`/api/winrate`：** 由云端服务器转发到本机 Python 服务，请求/响应原样，需要同样的密钥；Python 服务不可达返回 `502 {"error": "the Python app is not reachable"}`，超时返回 `504`。其他 `/api/*` 路径不转发（`404`）。
- **浏览器跨域：** 只允许运维配置的精确来源（没有通配符），预检请求（`OPTIONS`）对其他来源返回 `403`。
- **隐私：** 对局日志保存在服务器上，按密钥隔离，默认保留 30 天；用户可用自己的密钥下载或删除（§5.2）。服务器的日志里永远不会出现完整密钥。

