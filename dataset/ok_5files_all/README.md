# DataVideo OK 5-file Dataset

`ok_5files_all/` 是 340 条真实 data-video clip 的标注包。目录名里的 "5files" 是历史名称；
现在每条 clip 的文件数随审核进度不同。

每条 clip 都带 `clip.mp4`（原始视频片段），整个目录约 305 MB。

## 1. 每条 clip 的文件

已按新协议审核过的 clip（目前是 `bar_1`–`bar_6`）：

| 文件 | 作用 |
|---|---|
| `data_table.csv` | 审核后的数据表，是生成端的公开输入 |
| `intent.json` | 一句话公开 intent；保留 `narrative_goal`（叙事目标）和溯源字段 |
| `g1.json` | G1：要求（R）+ 单元（U）+ 权重 |
| `chart_spec.json` | 完整图型 spec，供 review、G2、审计使用 |
| `input_spec.json` | 精简 spec，只给生成模型：family / sub_type / orientation（动画图多一个 time_behavior） |
| `narration.jsonl` | ASR 句子与时间戳 |
| `semantic.svg` | 视觉参考 |
| `keyframe.png` | 参考关键帧 |
| `clip.mp4` | 原始 clip 视频（约 3–10 MB，供合作者核对） |

尚未按新协议审核的 clip 仍保留旧的 `g1_draft_b2.json` 和 `g2_evidence.json`；
迁移在逐条进行。

## 2. 审核顺序：表 → intent → G1

每条 clip 都按这个顺序处理，前一步没定就不做下一步。

### 2.1 数据表

长表，一行一个 `(entity, x/series)`。推荐列：

| 列 | 必填 | 说明 |
|---|---|---|
| `entity` | 是 | 类目/实体全名，不用缩写 |
| `series` | 多系列时 | 系列全名 |
| `x` | 时间/类目轴时 | 年份、月份等语义值，不用像素 |
| `metric` | 是 | 量的是什么，如 `illiteracy_rate`、`travel_days` |
| `value` | 是 | 数值；没有数值时留空 |
| `unit` | 是 | `%`、`days`、`USD`、`count`、`relative`、`million pounds` 等 |
| `label` | 否 | 图上印刷的原文，如 “a lot of time”“33.5%” |
| `role` | 否 | `bar`（默认）或 `annotation`（不是柱子的图上事实） |

规则：

1. **数值以旁白为准**；图上印刷值为辅；几何反推只用于 relative-only 图。
2. **不用缩写**：写 `Car`、`Boeing 747`、`Transit-oriented developments`、`million pounds`。
3. **不发明缺失值**。汽车只说 “a lot of time” 就留空 value，用 label 表达。
4. **非柱事实用 `role=annotation`**，如 bar_5 的 £105m 额外收入。
5. **relative-only 图**用 `metric` + `unit=relative`，不要把 0.087 当 8.7%。
6. 份额图每组应合计 100；按刻度估计的值要调整到合理范围。
7. 表保持精简：`x_px`、`color`、`source_type`、`entity_id` 等放 sidecar，不进输入表。

## 3. intent

输入是四样：**叙事目标（现有 intent_en）+ narration + data_table + semantic.svg**。
不用关键帧。

写法：

- 一句话，2–3 个分句，最多 32 词；
- **承载 narration 的叙事结论**（例如 “Sub-Saharan Africa improved more slowly”）；
- 年份可以出现，其他数值不出现；
- 不写图表描述、不提表格/空单元格、不写表里没有的事实；
- 保留 `narrative_goal`、`intent_machine`、`previous_intent` 作为溯源。

生成后做五类事实检查：entity / direction / scope / measure / answerable。

## 4. G1

两层结构：

```json
{
  "requirements": [
    {"id":"R1","question":"...","relation":"all_of|any_of|single","unit_ids":["U1"]}
  ],
  "units": [
    {"id":"U1","requirement_id":"R1","claim":"...","accept":["..."],
     "reject":["..."],"evidence":"..."}
  ],
  "weight_mode":"uniform_over_units",
  "unit_weights":{"U1":1.0}
}
```

- **R**：作品必须回答的问题；`all_of` 表示每个 U 都要，`any_of` 表示任选，`single` 表示一个 U 内部有多种可接受写法。
- **U**：可独立判定的一项内容。`accept` 是合法表达，`reject` 是常见错误，`evidence` 是视觉证据。
- **一个纯数值变化 = 一个 U**（bar_1、bar_2、bar_4）。
- 多个子答案都必须出现时，可以合并成一个 U（bar_2 的四个地区、bar_6 的三种非汽车方式）。
- 权重按 U 均分。

## 5. 怎么判断 U 是否达成

不是只看旁白有没有提到。一个 U 达成要同时满足：

- **C（内容）**：命题被表达出来——旁白或可读标注都算；
- **V（视觉）**：图本身能独立支持这个命题，不是只有字幕在说；
- **T（时间）**：讲到它的时候，证据是可见的。

只把数值念一遍：C 可能过，V 不过。例如 bar_3 若只说 “88 vs 85” 而不展示两者接近，不算达成。

## 6. 已审核样例（bar_1–bar_6）

| clip | 数据表要点 | intent 要点 | G1 |
|---|---|---|---|
| bar_1 | 汽车无数值，用 label；747 28 天、spaceship 2 天 | 比较三种方式的旅行时间 | 4 U：三项数值 + 顺序 |
| bar_2 | 4 地区 × 1990/2017 | 强调 SSA 相对改善最慢 | 2 U：四地区变化 + 相对降幅排序 |
| bar_3 | cyclists 88%、drivers 85% | 两者接近 | 1 U：差 3 个百分点，接近 |
| bar_4 | 单柱 1950–2017（5 个年份） | 非洲儿童死亡率长期进步 | 1 U：32.0→7.5%，降 24.5 点 |
| bar_5 | 0% vs 100% + £105m annotation | 说明征税差异与额外收入 | 2 U：0/100 对比 + £105m |
| bar_6 | 四种方式 × 全国/TOD | TOD 少开车、多公交/骑车/步行 | 2 U：汽车更低 + 三种方式更高 |

## 7. 已知问题

- bar_3 的 narration 说 “any other time”，但表里只有 cyclists vs drivers；已在该 clip 的 `intent.json` 加 `narration_scope_note`。
- 部分表是 relative-only（没有绝对值），不要补造。
- 其余 334 条仍待按本流程审核。

## 8. 相关文件

- 完整 spec：`chart_spec.json`
- 生成输入 spec：`input_spec.json`
- 协议：`benchmark_draft/gt/evaluation_protocol.md`、`CHART_SPEC.md`
