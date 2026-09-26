# Core v1 数据卡

## 身份与用途

`core_v1_candidate` 用于 shared Yes/No 决策模型的外部诊断，不是训练集、盲测认证或统一排行榜。固定来源、选择版本及 ID 见 [core_v1.json](manifests/core_v1.json) 和 [core_v1_retained_ids.json](manifests/core_v1_retained_ids.json)。打包格式为 `benchmark_core_v1`，保留 `candidate_not_release_ready` 状态；成功准备不等于发布批准。

当前准备仅构建 17 套、6,768 个候选，再保留 6,744 条；不会先下载完整历史 20 套、19,826 条。CLINC、BoolQ 因已见来源排除，ForecastBench 因回溯预测性质排除，不进入当前 Core。

## 固定计数

| 套件 | 候选 | 保留 |
|---|---:|---:|
| JevBench public original | 72 | 72 |
| JevBench public easy | 48 | 48 |
| JevBench public hard | 111 | 111 |
| Jabr v2 | 866 | 866 |
| Nimble VitaminC dev | 599 | 599 |
| Nimble MASSIVE en-US | 350 | 348 |
| Nimble MASSIVE de-DE | 350 | 348 |
| Nimble SQuAD2 | 299 | 299 |
| Nimble PAWS | 250 | 250 |
| Nimble MultiNLI | 299 | 299 |
| Nimble Civil Comments | 300 | 300 |
| Nimble Aegis2 | 250 | 250 |
| Nimble HelpSteer2 | 249 | 249 |
| Nimble SummEval relevance | 240 | 237 |
| Nimble SummEval consistency | 144 | 144 |
| Nimble PubMedQA | 250 | 250 |
| ContractNLI | 2,091 | 2,074 |
| **合计** | **6,768** | **6,744** |

4 条训练字面匹配及翻译家族传播排除、18 条重复排除、2 条目标歧义隔离。计数以“决策”为单位，并非相互独立的文档数；同文档多题、改写和跨语言组仍需按组解释。

## 格式与转换

| 文件 | 字段与角色 |
|---|---|
| `inputs.jsonl` | `id`, `suite`, `state`, `qdef`；模型输入只取后两项 |
| `targets.jsonl` | `id`, `gold_idx`, `soft`, `gold_score`, `option_ids`；仅评分使用 |
| `provenance.jsonl` | `id`, `suite`, `group_id`, `metadata`, `source`, `original_id` |
| `manifest.json` | 计数、路径、套件顺序、审计范围 |
| `decisions.jsonl`, `retained_ids.json` | 冻结的逐 ID 决策与保留顺序 |
| `audit.json`, `AUDIT_REPORT.md` | 公开成员回放摘要，不是新暴露审计 |
| `nimble_upstream_checksums.json` | Nimble 转换的上游校验记录 |

`qdef` 含 `type`、`instructions`、`criteria`；类型为 `choice`、`noul`、`score`。选择项顺序和 score 等级顺序有语义，不能随意排序。硬目标、人工投票软目标、精确机制软目标及标量评分分别保留；不能把 soft target 一律称为人工标注。

JevBench 使用三个固定公开文件，Jabr 使用 v2 TOML；Nimble 使用固定官方 subset ID 及其顺序（去除 BoolQ）；ContractNLI 使用 test 文档/假设对。转换沿用研究 v2 的输入与目标语义，不根据模型预测、tokenizer 长度或质量结果补样、换题。Jabr 是合成 benchmark；JevBench hard 含 LLM 编写/审阅标签。来源标签并不全是人工真值。

## 审计与使用边界

准备脚本只回放冻结成员并核对 ID、顺序和数量；原始暴露审计相对已知 S0/S1 训练、选优 dev、诊断 dev，不能保证未见预训练数据或未发现的语义近重复。公开题已被本项目观察过，不称为绝对无泄漏或盲测。完整说明见[重叠审计](../docs/overlap_audit.md)。

输入/目标分文件以及单独评分进程减少意外泄漏，但不是防恶意代码的安全隔离。超长记录属于运行覆盖问题，不从固定数据成员中移除。

按[来源与使用条款](../docs/benchmark_sources.md)自行获取数据；来源含毒性/安全内容、新闻、摘要与法律文本。来源许可、隐私及底层文本权利各不相同；仓库代码许可不统一覆盖这些材料。新原始数据不随此文档发布，重建文本也不应自动公开。此数据不适合作为医疗、法律或安全决策的独立依据。
