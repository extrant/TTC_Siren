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
  "solver": "minimax",       // str            — "minimax" 或 "monte_carlo"
  "max_depth": 10,           // int            — Minimax 最大思考深度
  "max_time": 10,            // float          — Minimax 最大思考秒数
  "mc_simulations": 150,     // int            — 蒙特卡洛模拟次数（仅 monte_carlo 有效）
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
| `solver` | str | 否 | 求解器类型，默认 `"minimax"` |
| `max_depth` | int | 否 | Minimax 最大思考深度，默认 10 |
| `max_time` | float | 否 | Minimax 最大思考秒数，默认 10 |
| `mc_simulations` | int | 否 | 蒙特卡洛模拟次数，默认 150 |
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
- `500` — 服务器内部错误

常见错误场景：

| 场景 | 响应 |
|------|------|
| 棋盘/手牌/owner 缺失 | `400` — `"Invalid input data"` |
| 卡牌数值不在数据库中 | `500` — `"Board card not found..."` / `"Hand card not found..."` |
| 玩家可用牌 < 5 张 | `/api/recommend` 返回空 results + process 说明 |
| NPC 已知牌 < 5 张 | `/api/recommend` 返回空 results + process 说明 |
| 无合法卡组 | `/api/draft/analyze` 返回空 results + process 说明 |
