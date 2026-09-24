# 高级用法

先按[根 README](../README.md)安装、下载模型并准备数据；以下命令沿用其中的路径变量。完整 S1 训练和全量评测命令见 README。

## S0 与单卡训练

训练 S0 时，把 README 训练命令中的 train/dev 路径换为 `$DATA_ROOT/s0/canonical/train.jsonl` 和 `$DATA_ROOT/s0/canonical/dev.jsonl`，`RUN_DIR` 换成全新输出目录；仍从 `BASE_MODEL` 启动。

单 GPU 将 `CUDA_VISIBLE_DEVICES=0,1` 改为一个可用设备，`--nproc_per_node=1 --micro-batch 4 --grad-accum 8`，保持有效 batch 32。若暴露物理 GPU 3（`CUDA_VISIBLE_DEVICES=3`），进程内仍使用逻辑设备 `cuda:0`。训练需要 GPU。

## 恢复与短运行

在原训练命令后添加 `--resume "$RUN_DIR/resume-latest"`；保留原模型、数据、输出目录及训练配置（含 GPU 数和 batch 配置）。`resume-latest/` 包含 optimizer、scheduler 与 RNG 状态，`best/` 和 `epoch-*` 则用于评测。重新训练或更换配置时使用新的输出目录。

在全量评测命令后添加 `--smoke` 并换全新 `EVAL_DIR`，可检查每套三个决策；正式评测去掉 `--smoke`。训练短运行可在主命令添加 `--train-limit 64 --dev-limit 16 --max-updates 2` 并另选 `RUN_DIR`。这些短运行结果不用于质量比较。

## 内部评测与输出

统一 runner 自动运行内部 S0 test（1,000 个决策），无须额外命令；DLLM 内部始终 batch 1。`external.json` 的 `suites` 列出六套外部任务，`internal.json` 报告内部指标和 `by_source` / `by_primitive` / `by_K` / `by_target_kind` 分组，两者分别计数。预测写入 `external_predictions.jsonl` 和 `internal_predictions.jsonl`；`run.json` 记录模式、配置、计数及运行状态。每次评测使用全新的 `EVAL_DIR`。

| 指标 | 含义 |
|---|---|
| `accuracy` | 决策 argmax 准确率 |
| `nll` / `ece` | gold 平均负对数概率 / 校准差 |
| `ms_per_decision` / `decisions_per_second` | 前向均摊耗时 / 吞吐 |
| `n_decisions` / `dropped` | 已评测 / 丢弃的决策数 |

## 外部吞吐与 Laya

质量对照使用 DLLM BF16 batch 1。外部吞吐对照在 README 评测命令中改为 `--batch-size 32`，并另选新 `EVAL_DIR`；内部始终 batch 1。可加 `--warmup-batches 3` 重复每套外部数据首批的预热前向（不计分或计时），比较时保持预热设置相同。`--dtype float32` 可用于精度检查。计时不含分词、传输、CPU softmax、I/O 或排队；均摊前向耗时不是单请求延迟。

如已有 Laya checkpoint，在统一 runner 的 README 命令中改为 `--backend laya --model-path "$LAYA_CHECKPOINT"`，并设置 `--laya-max-seqs 16 --laya-max-tokens 8192`，保持默认 `--batch-size 1 --warmup-batches 0`。Laya 保留原生 FP32 动态 batching；输出为 `laya.json`、`laya_predictions.jsonl` 和 `run.json`，主结果位于 `laya.json` 的 `conditions.aligned_unit.suites`。这与 DLLM batch 1/32 不是同配置速度对照。

## 可选入口

从来源重新构建数据见 [raw/prepared 参考](data_and_models.md) 和 [S1 builder](../research/S1_DATA_BUILDER.md)；已有 canonical 可直接训练。统一 runner 的旧 `--profile` 模式供已有 bundle 工作流使用，配置格式见 [profile 模板](full_eval_profile.template.json)；日常全量评测使用 README 的 `--data-root` 模式。
