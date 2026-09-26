# 重叠审计与公开成员回放

Core 的审计结论是相对已知 S0/S1 监督训练、选优 dev 和诊断 dev 暴露的有限结论，不是对预训练、语义重叠或全部潜在泄漏的证明。公开题已经被本项目观察过，不能称为盲测。

## 当前选择

来源政策先排除 CLINC、BoolQ 和 ForecastBench，公开准备仅构建剩余 17 套的 6,768 个候选。保留数量为：

| 决策 | 数量 | 说明 |
|---|---:|---|
| 暴露/家族排除 | 4 | MASSIVE 英文训练字面匹配及对应德文翻译家族传播 |
| 重复排除 | 18 | ContractNLI 重复文档的 17 个决策，加 SummEval relevance 1 条 |
| 歧义隔离 | 2 | SummEval relevance 重复输入的目标歧义 |
| 保留 | 6,744 | 固定 ID 与顺序，无补样 |

规则、排除 ID、重复 canonical ID 和最终成员见 [core_v1.json](../benchmark/manifests/core_v1.json) 与[保留 ID](../benchmark/manifests/core_v1_retained_ids.json)。不是根据某个模型得分、预测或上下文预算挑选保留项。

## 历史方法，不是本次重新执行

冻结配置记载历史词法审计使用 5-word shingles，精确匹配最少 3 词；近重复最少 12 词、Jaccard 0.8、containment 0.9、共享词最少 12，group 最少 40 词，未设置 candidate cap。词法匹配不自动证明语义等价；改写、翻译和未列入暴露清单的数据仍可能漏检。

`scripts/prepare_core.py` 的公开流程是 `frozen_public_membership_replay`：核对候选 suite/ID/group、显式排除和隔离、最终保留顺序及计数，然后打包。它不读取私有暴露清单，不重跑原完整词法/家族审计，也不需要私有目录、训练片段、预测或原始日志。

输出 `audit.json` 中的 `status=complete` 只表示所有候选都已应用冻结决策且检查通过；`AUDIT_REPORT.md` 是公开回放摘要，不是新的无污染认证。公开包与私有研究包的输入/目标语义保持，但审计与 provenance 公开范围不同。

## 如何解读

- 不声称绝对 leak-free、未见预训练或全面语义去重；未知预训练暴露保持未知。
- 不把成员审计等同于许可清理；下载和再分发边界见[来源说明](benchmark_sources.md)。
- 不把超长输入移出数据集：它们仍在固定成员内，由 evaluator 显式记录 unsupported。
- 不把文件分离与独立评分进程称为安全沙箱；自定义模型代码仍在用户权限下执行。
- 不把当前 Core、旧版六套评测、官方公开 231 题和 sealed leaderboard 的分数混成一个结论。

套件计数见[数据卡](../benchmark/DATA_CARD.md)，结果和负面发现见[结果说明](results.md)。
