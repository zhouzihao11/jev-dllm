# 协议：benchmark_core_v1

## 选择与审计

原始 `full_eval_v2` 的所有行必须出现在 `decisions.jsonl`（当前原始集合预期 19,826 行，以 profile 数量逐一核对）。每行含 `id,suite,group_id,decision,reasons`，decision 为 retain/exclude/quarantine。`retained_ids.json` 含原 `profile_id`、无重复的 `retained_ids` 和整数 `counts_by_suite`。还需真实 `audit.json` 与 `AUDIT_REPORT.md`。

打包逐 ID 核对审计、原始 suite/group、保留集合及计数。CLINC、BoolQ、ForecastBench 在 core 中强制禁止，即使审计误标 retain 也报错，不自动覆盖审计决定。不得按模型成绩、成功运行与否或上下文长度选择样本。原始文件只读；不生成新 ID、目标或概率分布，不改变 state/qdef、选项或行顺序。审计仍依赖已识别来源和审计方法，不意味着检测到所有隐性暴露。

manifest 的 `schema_version=benchmark_core_v1`、`profile_id=core_v1_candidate`、`source_profile_id` 固定标识协议与来源，包含每 suite 原始计数和保留计数、相对文件路径、审计范围、选择版本与决定计数。公开包保留固定成员决定与审计摘要，不分发私有训练片段或路径。准备步骤是冻结成员回放，不是重新执行完整训练重叠审计，也不宣称绝对无污染。

## 文件边界

| 文件 | 行字段 | 使用方 |
| --- | --- | --- |
| inputs.jsonl | id, suite, state, qdef | 推理包装层；adapter 只接收 state/qdef |
| targets.jsonl | id, gold_idx, soft, gold_score, option_ids | 独立 scorer |
| provenance.jsonl | id, group_id, suite, metadata, source, original_id | 独立 scorer |

`source` 保存原 suite 描述，不能作为可执行下载路径使用。文件字典 insertion order 保留。manifest 不内嵌 targets/provenance；只有对应路径。推理包装层不会打开这些文件，也不调用需要 labels 的 `validate_row`。adapter 延迟导入既有 Evaluator，仅构造 state、qdef、primitive、按继承 `render_options(to_internal(qdef))` 派生的 options 和常量占位 ID。

这是程序数据流隔离，不是恶意 checkpoint 的操作系统沙箱；现有 backend 仍使用 `trust_remote_code`。需要安全隔离时由部署方限制模型进程文件权限。

## 执行

沿用 aligned T=1、suite-local contiguous windows、原选项顺序及 first-argmax。DLLM 使用继承 Yes/No mask 路径；Laya 使用继承 native aligned 序列，noul 的 rubric 处理、literal-mask 替换和 runtime 数值策略不变。4096 为请求上限，还受模型实际上限限制。长度超限为 `unsupported_length`；DLLM literal mask 为 `unsupported_source_mask`，均保留在分母中，不截断。

继承路径的 logits/probability/transfer/marker 异常或 OOM 是 fatal error，不替换成成功预测，不重试为小 batch。已经写出的预测保留，失败 batch IDs 和未完成 IDs 写入 inference metadata；`run.json` 分阶段记录失败。此类未完成运行不产出正式完整分数。正常 unsupported 行保留 null probs/pred_idx。批内 supported 数量、window size、请求 batch size 和同步 forward timing 分开记录；除法分配的 row seconds 不是 request latency。

smoke 是每 suite 保留顺序前 N 行，只验证执行。独立 scorer 要求显式 run manifest，重建并核对 expected IDs；不允许任意缺行借 smoke 名义变成完整分数。`--predict-only` 明确标记 unscored，不产生质量结论。

## 评分

直接复用继承 `bench_extended.summarize`，避免改变浮点、排序或 round 语义。hard 指标保留原四位小数 round：accuracy、15-bin ECE `(lo,hi]`、sum-over-classes Brier、NLL、mean confidence、50% coverage accuracy；不保留无意义的可变 label-index macro F1。soft CE/Brier、binary ptrue MSE、score MAE 和 RPS 保留继承未统一 round 的值和样本 n。不重新归一化目标或任意模型概率。

score levels 为从零开始的有序索引，不是自然语言评分标签的数值。MAE 优先 gold_score，其次 soft 期望，再 hard index；RPS 为 K-1 个 CDF 边界的未归一化平方差和，soft 优先、否则 hard one-hot，scalar-only 不产生 RPS。硬标签不替代 soft 分布，null gold 不计硬真值。

按 suite、primitive、task 输出成功覆盖率、所有尝试 hard accuracy、status counts 及组 exact match。unsupported 对 hard all-attempted accuracy 计错。组必须覆盖原审计中的完整 group 且所有成员有 hard truth；被 audit、smoke 或报告分区截断的组不作为完整组。每 suite 分母、n 与排除组数必须一并解释。无全局 micro winner。

历史模式仅全量 retained core，验证原始 profile state/qdef、targets、option IDs、provenance 和旧 artifacts 中目标与 renderer options 后按 ID 重评分，报告显著标记 historical_rescore，不宣称重新运行或消除既有选择暴露。
