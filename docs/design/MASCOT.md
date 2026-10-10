# 看板娘设定与生图规范 — 瑟琳（Seren Nocturne）

用途：Web UI（Phase 8B）的看板娘，按事件切换表情或姿态，并配合台词库（8D）。本文档供用户用 GPT 生成图片，也供实现者接入。**这是原创角色**：不得模仿任何现有游戏或动画角色（尤其避免与 FFXIV 角色、原神"珊瑚宫心海""芙宁娜"等海洋主题角色相似）。

## 1. 角色设定

| 项目 | 设定 |
|---|---|
| 名字 | 瑟琳·诺克图恩（Seren Nocturne），简称"瑟琳"。Seren 呼应项目名 Siren，在威尔士语中也有"星"的意思 |
| 身份 | 深海歌剧院"潮汐牌桌"的首席荷官；传说中能用歌声"听见"每张牌的塞壬 |
| 年龄感 / 身高 | 约 19–20 岁的成年女性外观，165 cm，身形修长 |
| 发型 | 及腰长直发，从发根的午夜蓝渐变到发梢的珍珠白，发尾微卷如浪；左侧编一缕细辫；右侧别一枚银色贝壳发饰 |
| 耳饰 | 半透明、青色、鳍形的耳饰（是饰品，不是精灵耳） |
| 眼睛 | 海玻璃青绿色，瞳孔中有一个极小的菱形（卡牌形）高光；右眼下有一颗泪痣 |
| 服装 | 改良荷官礼服：<br>· 白色高领衬衫，配嵌珍珠的深蓝丝绒领结；<br>· 午夜蓝修身马甲，带银色细竖条纹；<br>· 不对称燕尾短外套，内衬是海浪渐变丝绸，外套后摆是多层半透明、像鱼鳍一样的薄纱；<br>· 高腰长裙，一侧开衩；深蓝短靴，带珍珠纽扣 |
| 手套 / 配饰 | 白色荷官手套，右手手背用银线绣九宫格；腰间挂银链，坠子是迷你 3×3 棋盘；袖扣是两张小卡牌 |
| 道具 | 一副半透明水晶质感的卡牌，卡背是浪花与星图纹样；身边漂浮少量小气泡和光粒 |
| 配色 | 主色午夜蓝 `#1B2A4A`，辅色海玻璃青 `#4FD1C5`，珍珠白 `#F4F1EA`，点缀香槟金 `#C9A86A` |
| 性格 | 清冷优雅，话少而准；分析时语气平稳；遇到险招会微微挑眉；偶尔带一点温柔的揶揄；赢了只是浅浅一笑并行礼；指出失误时不嘲讽 |
| 口癖 / 台词风格 | "有趣。" "潮汐会记住每一张牌。" "这一步……让海潮倒灌了。" "放心，我已经听见答案了。" |

## 2. 交付要求（请按此生成和命名）

**存放位置：** `assets/mascot/raw/`（仓库根目录下，已建好）。8B 的构建流程会把它们转换成 WebP，并放进前端目录，原图保持不动。

**通用要求**

- **格式与尺寸：** PNG，**透明背景**；竖幅 **1024×1536**（GPT 原生竖幅尺寸）。全身图同尺寸。
- **构图（全部表情图必须一致）：**
  - 半身到大腿中部；角色水平居中，面向观众略偏左。
  - 头顶距上边缘约 6–8%，下端在画面底部裁切。
  - **每张图的位置、大小、身体姿势尽量相同，只改变表情和手部动作**，方便网页无缝切换。
- **风格：** 高质量日系二次元插画，干净的线稿，柔和赛璐珞加轻度厚涂高光；不要写实、不要 3D 渲染感、不要文字和水印、不要多余人物。
- **一致性做法：**
  1. 先生成 **01 角色设定图**并定稿。
  2. 之后每次生成都把定稿的设定图作为**参考图**上传，并在提示词中写明 "same character as the reference image, same outfit and colors"。
  3. 生成表情图时，再把定稿的 E01 一并上传，作为构图参考。

**必需清单**

| 文件名 | 内容 | 网页中的触发场景 |
|---|---|---|
| `seren_ref_sheet.png` | 设定图：正面、侧面、背面全身三视图 + 色板；可用白底，尺寸可用 1536×1024 横幅 | 只作参考，不在网页显示 |
| `seren_e01_neutral.png` | 默认：平静、微笑 | 待机 |
| `seren_e02_thinking.png` | 思考：手指轻抵下巴，视线向上，身边浮着两张卡 | 求解进行中 |
| `seren_e03_confident.png` | 自信：微挑一边眉，指间夹一张卡 | 找到最佳着 / 已解出必胜 |
| `seren_e04_surprised.png` | 惊讶：睁大眼，手掩口 | 对手打出强力连携、局势突变 |
| `seren_e05_worried.png` | 担忧：眉头微蹙，卡牌抱在胸前 | 局面不利、高风险 |
| `seren_e06_happy.png` | 喜悦：闭眼浅笑，微微欠身 | 胜利 |
| `seren_e07_explaining.png` | 讲解：戴手套的食指指向画面左侧 | 对局指正、讲解走法 |
| `seren_e08_disappointed.png` | 失落：轻叹，半闭眼，视线低垂 | 失败、用户大失误 |

**可选清单（有更好，没有也能上线）**

| 文件名 | 内容 |
|---|---|
| `seren_p01_shuffle.png` | 全身：双手之间洗牌，卡牌划出弧线（加载画面） |
| `seren_p02_present.png` | 全身：一手向观众递出一张卡（开局 / 欢迎页） |
| `seren_e09_sleepy.png` | 打哈欠（长时间无操作） |
| `seren_chibi_01..04.png` | Q 版 4 张，512×512 透明背景：点赞、思考、惊讶、鞠躬（通知图标） |

## 3. 生图提示词（英文给 GPT，效果更稳定）

### 01 角色设定图（先做这一张）

```
Original anime character design reference sheet, full body turnaround: front view, side view, back view, plus a color palette strip.
Character: "Seren", an elegant, calm young woman (about 20), a siren-themed card dealer (croupier) of a deep-sea opera casino.
Hair: waist-length straight hair with a gradient from midnight blue at the roots to pearl white at the tips, tips softly curled like waves, one thin braid on the left side, a small silver seashell hair clip on the right.
Ears: normal human ears wearing translucent teal fin-shaped ear ornaments.
Eyes: sea-glass teal, a tiny diamond (card-shaped) highlight in each pupil, a beauty mark under the right eye.
Outfit: white high-collar shirt with a deep navy velvet bow tie set with a pearl; fitted midnight-blue waistcoat with thin silver pinstripes; asymmetric cropped tailcoat lined with wave-gradient silk, its back hem made of layered translucent fin-like chiffon; high-waisted long skirt with a side slit; navy ankle boots with pearl buttons.
White croupier gloves, the back of the right glove embroidered with a silver 3x3 grid; a silver waist chain with a tiny 3x3 game-board pendant; small card-shaped cufflinks.
Holding a few translucent crystal-like playing cards whose backs show waves and a star map.
Color palette: midnight blue #1B2A4A, sea-glass teal #4FD1C5, pearl white #F4F1EA, champagne gold accents #C9A86A.
Style: high-quality Japanese anime illustration, clean lineart, soft cel shading with light painterly highlights, consistent proportions across views.
Plain white background, no text, no watermark, single character only. Original design, not resembling any existing game or anime character.
```

### 表情图（每张都上传定稿设定图作参考；从第二张起再加上 E01）

先用下面的通用前缀，再接上各张的专属描述。

**通用前缀：**

```
Same character as the reference image (Seren), same outfit, colors, hairstyle and accessories.
Half-body portrait from head to mid-thigh, character centered, facing the viewer turned slightly to the left, top of the head about 7% below the top edge.
Keep the same pose, framing, size and position as the neutral reference; change only the facial expression and the hand gesture described.
Japanese anime illustration, clean lineart, soft cel shading. Transparent background, PNG, 1024x1536. No text, no watermark, no other characters.
```

**各表情专属描述：**

- E01 neutral：`Calm, gentle closed-mouth smile, relaxed eyes looking at the viewer, hands lightly holding a fanned set of three crystal cards at waist height. A few tiny bubbles float around her.`
- E02 thinking：`Thoughtful expression, eyes looking up and to the side, index finger of the gloved right hand resting on her chin, two crystal cards floating beside her shoulder with a faint teal glow.`
- E03 confident：`Composed confident look, one eyebrow slightly raised, subtle knowing smile, holding one crystal card between two gloved fingers near her face.`
- E04 surprised：`Surprised, eyes wide, small open mouth, gloved hand raised near her mouth, hair and fin-like coat hem slightly lifted as if by a sudden current.`
- E05 worried：`Worried expression, brows slightly knitted, lips pressed, holding a crystal card close to her chest with both hands.`
- E06 happy：`Gentle happy smile with eyes closed, a slight elegant bow, one hand on her chest, sparkling bubbles around her.`
- E07 explaining：`Calm instructive expression, mouth slightly open as if speaking, gloved index finger pointing to the left side of the image, the other hand holding a card at her side.`
- E08 disappointed：`Quiet disappointed sigh, eyes half closed and looking down, shoulders slightly lowered, one card hanging loosely from her fingers.`

### 可选：全身姿态与 Q 版

- P01 shuffle（全身，1024×1536，透明背景）：`Same character as the reference image, full body, standing, shuffling crystal cards between her hands so the cards form a flowing arc, elegant motion, coat hem fluttering. Transparent background.`
- P02 present（全身）：`Same character as the reference image, full body, stepping forward slightly and offering one crystal card toward the viewer with her right hand, warm faint smile. Transparent background.`
- E09 sleepy（同表情构图）：`Covering a small yawn with her gloved hand, eyes half closed, sleepy but graceful.`
- Chibi（512×512，透明背景，一次生成一张）：`Chibi (super-deformed) version of the same character, big head small body, simple shading, transparent background, pose: [thumbs up / thinking / surprised / bowing].`

## 4. 接入约定（给实现者，8B）

- 原图放在 `assets/mascot/raw/`，不要修改。构建脚本生成 WebP（例如 512 px 和 1024 px 两档）放进前端目录。缺少的表情回退到 `e01_neutral`。
- 切换方式：交叉淡入淡出（约 150 ms）；尊重 `prefers-reduced-motion`。
- 事件 → 表情映射：
  - `search.started` → e02；`search.solved_win` → e03；`opponent.combo` → e04；`eval.dropping` → e05；
  - `game.won` → e06；`coach.explaining` → e07；`game.lost` / `coach.blunder` → e08；`idle` → e01（长时间无操作 → e09）。
- 台词库（8D）：按事件分组，每组若干条，语气参照第 1 节。台词只说引擎事实支持的内容，例如不能在没有解出必胜时说"赢定了"。
