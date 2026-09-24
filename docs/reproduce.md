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

在 README 的统一评测命令中添加 `--smoke`，并使用全新的 `--output-dir` 即可检查每套三个决策；仍需完整 `DATA_ROOT/bench` 七个输入文件。外部与 Laya 内部选取前缀，DLLM 内部沿用 seed42 whole-group spread，样本不完全一致；这是执行检查，不用于质量比较。正式运行去掉 `--smoke`，默认所有 test limit 为 0。

训练入口也提供 `--train-limit`、`--dev-limit` 和 `--max-updates`。例如在主训练命令中添加 `--train-limit 64 --dev-limit 16 --max-updates 2`，并另选 `RUN_DIR`，可检查短运行；不要用这个截断配置恢复正式训练。

统一 runner 已自动评测内部测试，DLLM 内部始终 batch 1、无预热，不受外部 `--batch-size` / `--warmup-batches` 影响。只有需要单独运行内部测试时才使用以下底层入口；`INTERNAL_ONLY_DIR` 的父目录需存在，且输出文件必须全新：

```bash
export INTERNAL_ONLY_DIR="$PWD/outputs/internal_only"
mkdir -p "$INTERNAL_ONLY_DIR"
CUDA_VISIBLE_DEVICES=0 python research/scripts/shared_yesno_supervised.py \
  --model-path "$CHECKPOINT" \
  --data "$DATA_ROOT/bench/internal_s0_test.jsonl" \
  --output "$INTERNAL_ONLY_DIR/internal.json" \
  --predictions "$INTERNAL_ONLY_DIR/internal_predictions.jsonl" \
  --device cuda:0 --amp-dtype bfloat16 --max-length 4096 --limit 0
```

评测 base 时将 `CHECKPOINT` 改为 `BASE_MODEL`，并另选输出目录；S1 的 `new_dev.jsonl` 是诊断集，不是内部 test。

### 统一输出与状态

DLLM 输出 `$EVAL_DIR/run.json`、`external.json`、`external_predictions.jsonl`、`internal.json`、`internal_predictions.jsonl`。内部指标位于 `internal.json` 顶层及 `by_source` / `by_primitive` / `by_K` / `by_target_kind`，不混入外部均值。

Laya 保留 `$EVAL_DIR/run.json`、`laya.json`、`laya_predictions.jsonl`，不改写为 DLLM 的概率或指标格式。`laya.json` 的 `conditions.aligned_unit.suites` 是主结果，内部 suite 名为 `internal`；`aligned_sdk_temperature`、`sdk_unit`、`sdk_native` 是三个参考条件，预测文件以 `condition` 区分。

`run.json` 记录运行模式、模型路径、代码路径/HEAD（无 Git 快照为 `unknown`，不是历史版本声明）、实际使用的 Python、有效精度与 batching 配置、预期/计划/实际计数、子命令和阶段退出码。默认全量期望决策数为 typed 2,000、AG News 7,600、Emotion 2,000、Banking77 3,080、Prompt Injection 116、SST-5 2,210、内部 1,000。各阶段成功且计数匹配后 `status=complete`；全量 `full_status=complete`，smoke 则为 `not_full_smoke`。失败阶段保存 `error` / `error_details`，后续阶段保留 `pending`，已有输出不删除；重跑必须使用新目录。启动参数/缺失路径错误在创建目录前直接报错。

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

质量主表使用 DLLM BF16 batch 1；吞吐对照只改 `--batch-size 32` 并保持同模型、设备、精度与预热。batching 仅覆盖外部六套数据，尾批可小于指定 batch；实际 batch 大小和预热 forward 数见 `external.json`。`--warmup-batches` 默认为 0，可设为 3，重复各 suite 首批，不计分、不计时。计时不包含分词、传输、CPU softmax、I/O 或排队，均摊前向耗时不等于单请求延迟。原 BF16 B1/B32 对照有 72/17,006 个 argmax 翻转；FP32 检查用 `--dtype float32`，不要默认 batching 逐位等价。

Laya 对照通过同一 runner 的 `--backend laya --model-path "$LAYA_CHECKPOINT"` 运行，保留原生头与 FP32 动态 batching（`--laya-max-seqs 16 --laya-max-tokens 8192`），不是 DLLM batch 1 或 batch 32 的同配置速度对照。DLLM 专用 `--batch-size` 非 1 或 `--warmup-batches` 非 0 会报错；`--dtype` 只影响 DLLM，不改变 Laya FP32。

## 可选：从来源重建数据

已有 canonical 可直接训练。只有需要改变数据构建时，才按[raw/prepared 契约](data_and_models.md)另行准备完整来源；S1 细节见 [S1_DATA_BUILDER](../research/S1_DATA_BUILDER.md)。重建使用新的数据目录，不覆盖已解压的输入。

## 可选：历史 runner 与 profile

`research/scripts/run_full_benchmark.py` 的 `--data-root` 与 `--profile` 必选其一。主流程使用 `--data-root`，不需要冻结 bundle；当前 Python、HOME、HF 缓存和源码供子进程使用，并设置仓库根目录、`research/scripts`、`support/dllm_stub` 的 PYTHONPATH 与离线环境。高级 `--profile` 模式仍用于历史 bundle 工作流。以 [profile 模板](full_eval_profile.template.json) 为起点，在独立目录准备：

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

正式运行去掉 `--smoke` 并换全新 output-dir。历史冻结 evaluator 不接受新的 batch/warmup 参数，因此 profile 模式仅允许默认 `--batch-size 1 --warmup-batches 0`，且不会向子命令传递这两个新参数；需要外部 batching/预热请使用 `--data-root`。原 profile 代码路径、运行时、limit 和精度默认值保持不变。Laya 使用 `--backend laya` 并提供相应英文 checkpoint，保留原生 FP32 配置。

### 历史版本参考

| 用途 | 原研究 SHA |
|---|---|
| 冻结质量评测，profile `full_test_v1_20260923_r2` | `fb672477594f29681b90d13ec679819ab8033c5c` |
| S1 构建/训练 | `80efa54951000a7b6f1637ea2bf146332540f9e2` |
| 精选源码基线，含批处理评测 | `5d1b7de221579af74bde3131bc4eb8e61bba7c0f` |

这些历史对象不包含在本仓库中。运行当前源码是新评测；`--data-root` 不声明历史 profile_id，若自行创建当前源码的 profile 则使用新的 profile_id。严格复现原冻结质量快照才需要另行取得对应 bundle。原研究数值见博客，数据准备验证记录保留在 [manifest](../datasets/manifest.json)。
