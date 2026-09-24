# 高级用法

先完成[根 README](../README.md) 的环境、模型下载和数据准备。本文沿用其中的路径变量；完整 S1 训练和六套全量评测命令只在 README 维护。

## S0 与单卡训练

训练 S0 时，将主命令的 train/dev 路径改为 `$DATA_ROOT/s0/canonical/train.jsonl` 和 `$DATA_ROOT/s0/canonical/dev.jsonl`，并将 `RUN_DIR` 设为新的输出目录。模型起点仍为 `BASE_MODEL`，不是已有的 S1 权重。

单 GPU 仍通过 `torchrun` 启动 NCCL：将 `CUDA_VISIBLE_DEVICES` 设为一个可用设备，`--nproc_per_node=1 --micro-batch 4 --grad-accum 8`，有效 batch 保持 `1 × 4 × 8 = 32`。trainer 使用逐样本 forward 累积梯度，micro-batch 并非四样本张量 forward；它不支持 CPU 训练。

`CUDA_VISIBLE_DEVICES=3` 表示只暴露物理 GPU 3；脚本内对应的逻辑设备仍为 `cuda:0`，不要写成 `cuda:3`。

## 恢复训练

在原训练命令后添加：

```bash
--resume "$RUN_DIR/resume-latest"
```

保持原 `--model-path`、`--output-dir`、数据文件及训练配置不变，包括 world size、epochs、micro-batch 和 grad-accum。恢复检查会比较配置及数据路径、大小、mtime；从双卡改成单卡不是同一次运行的恢复。

`resume-latest/` 含 optimizer、scheduler、RNG 等完整状态；`best/` 和 `epoch-*` 用于模型评测，不代替完整恢复点。中断恢复时可检查保留的 `resume-previous/` 是否为完整 checkpoint。原 S1 记录按 dev mean KL 在 step 1,300 选优，新训练不保证在同一步选优。

## 小规模检查与内部评测

在 README 的外部评测命令中，将 `--limit-per-suite 0` 改为 `--limit-per-suite 3`，并使用新的 output/predictions 文件即可检查每套三个决策；仍需六套输入文件，其余 test-limit 保持 0。小样本不用于报告全量质量。

训练入口也提供 `--train-limit`、`--dev-limit` 和 `--max-updates`。例如在主训练命令中添加 `--train-limit 64 --dev-limit 16 --max-updates 2`，并另选 `RUN_DIR`，可检查短运行；不要用这个截断配置恢复正式训练。

内部测试独立于六套外部数据：

```bash
CUDA_VISIBLE_DEVICES=0 python research/scripts/shared_yesno_supervised.py \
  --model-path "$CHECKPOINT" \
  --data "$DATA_ROOT/bench/internal_s0_test.jsonl" \
  --output "$EVAL_DIR/internal.json" \
  --predictions "$EVAL_DIR/internal_predictions.jsonl" \
  --device cuda:0 --amp-dtype bfloat16 --max-length 4096 --limit 0
```

评测 base 时将 `CHECKPOINT` 改为 `BASE_MODEL`，并另选输出文件；S1 的 `new_dev.jsonl` 是诊断集，不是内部 test。

## 质量与吞吐指标

外部 JSON 的 `suites` 按测试集组织结果：

| 字段 | 含义 |
|---|---|
| `accuracy` | 决策 argmax 准确率，0 到 1 |
| `nll` | gold 的平均负对数概率 |
| `ece` | 置信度与正确率的分箱校准差 |
| `ms_per_decision` | 同步前向总耗时除以决策数，毫秒 |
| `decisions_per_second` | 同一计时范围的前向吞吐 |
| `n_decisions` / `dropped` | 评测决策数 / 丢弃数 |

质量主表使用 DLLM BF16 batch 1；吞吐对照只改 `--batch-size 32` 并保持同模型、设备、精度与预热。计时不包含分词、传输、CPU softmax、I/O 或排队，均摊前向耗时不等于单请求延迟。原 BF16 B1/B32 对照有 72/17,006 个 argmax 翻转；FP32 检查用 `--dtype float32`，不要默认 batching 逐位等价。

Laya 对照保留原生头与 FP32 动态 batching（最多 16 sequences / 8,192 tokens），不是 DLLM batch 1 或 batch 32 的同配置速度对照。

## 可选：从来源重建数据

已有 canonical 可直接训练。只有需要改变数据构建时，才按[raw/prepared 契约](data_and_models.md)另行准备完整来源；S1 细节见 [S1_DATA_BUILDER](../research/S1_DATA_BUILDER.md)。重建使用新的数据目录，不覆盖已解压的输入。

## 可选：历史 runner 与 profile

`research/scripts/run_full_benchmark.py` 用于 bundle/profile 工作流，不是主流程的前置要求。以 [profile 模板](full_eval_profile.template.json) 为起点，在独立目录准备：

- `sources`：六套 fixtures 和内部 S0 test，可从 `$DATA_ROOT/bench` 复制。
- `code.path`：实际使用的代码目录；`code.commit` 如实填写对应版本。
- `runtime.dllm_stub`：可复制本仓库 `support/dllm_stub/` 到 bundle 的同名目录。
- `runtime.python`、`runtime.home`、`runtime.hf_home`：执行环境的真实绝对路径。

code、stub、sources 的路径相对 profile 目录，不能包含 `..`；JSON 不展开 `$VAR` 或 `~`。runner 使用指定的 Python 和代码目录、设置 HOME/HF_HOME 并强制离线，数据 helper 不生成 profile。

例如将适配好的 profile 路径赋给 `PROFILE` 后：

```bash
python research/scripts/run_full_benchmark.py \
  --profile "$PROFILE" --backend dllm --model-path "$CHECKPOINT" \
  --output-dir "$EVAL_DIR/profile_smoke" \
  --device cuda:0 --dtype bfloat16 --smoke
```

正式运行去掉 `--smoke` 并换全新 output-dir。runner 没有 `--batch-size`，DLLM 默认 batch 1。Laya 使用 `--backend laya` 并提供相应英文 checkpoint，保留原生 FP32 配置。

### 历史版本参考

| 用途 | 原研究 SHA |
|---|---|
| 冻结质量评测，profile `full_test_v1_20260923_r2` | `fb672477594f29681b90d13ec679819ab8033c5c` |
| S1 构建/训练 | `80efa54951000a7b6f1637ea2bf146332540f9e2` |
| 精选源码基线，含批处理评测 | `5d1b7de221579af74bde3131bc4eb8e61bba7c0f` |

这些历史对象不包含在本仓库中。运行当前源码是新评测，使用新的 profile_id；严格复现原冻结质量快照才需要另行取得对应 bundle。原研究数值见博客，数据准备验证记录保留在 [manifest](../datasets/manifest.json)。
