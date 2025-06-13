<div align=center>
<img src="https://raw.githubusercontent.com/extrant/IMGSave/refs/heads/main/8edcffed756a48e0ac6018c3ef2cab66_th.jpg" width="400px" align="center">
</div>

---

<h1 align="center"><b>TTC_Siren</b></h1>

# 基于《最终幻想14》规则的幻卡对战求解器

本项目实现了一个幻卡对战AI求解器，采用博弈树搜索算法与并行处理技术。核心为游戏状态评估和移动预测引擎，并提供可选的HTTP服务器接口以便集成。

<div align=center>
<img src="https://github.com/extrant/IMGSave/blob/main/E9A888E96BB05418C8E12CE8FD63EE5C.png" width="400px" align="center">
</div>

## 特性 Highlights
- **完全支持 FFXIV 幻卡（Triple-Triad）核心规则**，并新增"选拔 (Draft)"等扩展规则。
- **智能未知卡牌推断**：使用加权采样结合星级配额、局面分析与角落策略，对对手隐藏手牌进行概率建模。
- **多层评估函数**：位置价值 + 星级/边值 + 角落高边检测 + 规则适配 + 捕获潜力。
- **迭代加深 α-β 搜索**：带置换表、历史启发与动态时间管理，在 5 秒内可搜索 >70k 结点。
- **HTTP 服务接口**：基于 Flask / Flask-CORS，可直接嵌入网页或 Bot。
- **配置驱动**：`config/unknown_card_config.py` 中即可开关 `performance_mode`、`aggressive_sampling` 等参数。
- **纯 Python 依赖**，无需编译扩展，支持 Windows / macOS / Linux。

## 核心求解器功能
### AI引擎（ai.py）
实现以下关键能力：
- **搜索算法**：基于Alpha-Beta剪枝的极小化极大值搜索 + 迭代深化时间管理
- **评估体系**：多重启发式函数驱动的复杂状态评估
- **性能优化**：置换表缓存、历史移动排序、并行计算支持


## 游戏状态评估模型
评估函数综合以下维度：
| 评估维度       | 具体规则                          |
|----------------|-----------------------------------|
| **位置权重**   | 角落：1.5倍 • 中心：1.2倍 • 其他：1.0倍 |
| **卡牌属性**   | 星级权重 + 数值翻转潜力           |
| **棋盘控制**   | 领地占有率分析 + 翻转连锁反应预测 |


## 游戏规则支持
当前内置规则（可组合）：
- **全明牌**
- **三明牌**
- **同数 / 加算**
- **同类强化 / 同类弱化**
- **逆转**
- **王牌杀手**
- **选拔 (Draft)** — 每星级最多 2 张，AI 智能分配星级配额。


## HTTP服务器接口（可选）
基于Flask实现的集成接口：

### 接口：`/ai_move`
**请求方式**：POST  
**请求体结构**：
```
{
  "board": [
    {
      "pos": [row, col],       // 位置坐标 [0-2, 0-2]
      "numU": number,          // 上方数字 (1-9)
      "numR": number,          // 右方数字 (1-9)
      "numD": number,          // 下方数字 (1-9)
      "numL": number,          // 左方数字 (1-9)
      "owner": number          // 所属方（1=蓝方，2=红方）
    }
  ],
  "myHand": [
    {
      "numU": number,          // 手牌数字（未知卡牌用[0,0,0,0]表示）
      "numR": number,
      "numD": number,
      "numL": number
    }
  ],
  "oppHand": [
    {
      "numU": number,          // 对手手牌（未知用[0,0,0,0]）
      "numR": number,
      "numD": number,
      "numL": number
    }
  ],
  "myOwner": number,         // 当前玩家阵营（1/2）
  "rules": string            // 游戏规则（如"三明牌+加算"）
}
```

**响应结构**：
```
{
  "card": "U8 R9 D7 L6 星级:3",  // 选中卡牌属性
  "pos": [row, col]              // 放置位置坐标
}
```


## 快速开始
```bash
# 克隆仓库
$ git clone https://github.com/yourname/TTC_Siren.git
$ cd TTC_Siren

# 创建虚拟环境（可选）
$ python -m venv venv && source venv/bin/activate  # Windows 用 venv\Scripts\activate

# 安装依赖
$ pip install -r requirements.txt

# 或启动 HTTP 服务 (默认 5000 端口)
$ python ai_server.py
```

## 项目依赖
- Python ≥3.8
- Flask
- pandas
- numpy

> 依赖均为纯 Python 实现，无需额外系统库。


## 幻卡数据库结构

| 字段名   | 类型   | 说明                     |
|----------|--------|--------------------------|
| 序号     | int    | 唯一标识                 |
| Top      | int    | 上方数字（1-10）          |
| Right    | int    | 右方数字（1-10）          |
| Bottom   | int    | 下方数字（1-10）          |
| Left     | int    | 左方数字（1-10）          |
| 星级     | int    | 卡牌星级（影响基础权重）|

