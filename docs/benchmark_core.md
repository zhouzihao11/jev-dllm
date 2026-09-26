# Core 评测细节

日常入口见[根 README](../README.md)，文件索引见 [benchmark README](../benchmark/README.md)。所有命令从仓库根目录执行；模型路径、缓存、输出和运行配置都是运行时参数。

## 固定成员，独立评分

`scripts/prepare_core.py` 获取 14 个固定来源条目，转换 6,768 个候选，回放冻结排除决策，生成 17 套、6,744 条的输入/目标/来源分离包。`--output-dir` 必须不存在。若已有完整缓存，可加 `--offline`；不要用替代来源填补失败下载。

推理子进程读取 `manifest.json` 与 `inputs.jsonl`，adapter 仅接收 `state` / `qdef`。随后 `benchmark/score.py` 在另一个进程读取 targets/provenance 并评分。这约束正常实现的数据流，不是阻止恶意 checkpoint 访问磁盘的 OS 沙箱。

公开准备验证的是成员身份、顺序和数量，不重新审计私有训练/dev 文本；具体边界见[审计说明](overlap_audit.md)。

## 运行

```bash
export CORE_DATA="$PWD/local_data/core-v1"
mkdir -p outputs
python benchmark/evaluate.py \
  --data-root "$CORE_DATA" --backend dllm --model-path "$CHECKPOINT" \
  --device cuda:0 --dtype bfloat16 --batch-size 32 --max-length 4096 \
  --log-every-batches 1 --output-dir "$PWD/outputs/core-s1-full"
```

先检查流程可添加 `--smoke --smoke-limit 3` 并使用另一个新输出目录。smoke 取每套前缀，不随机采样，不作为质量结果。默认日志每 10 个 batch 以及套件边界打印；首批也会打印。显存不足可以减小 batch，但不能删题或把修改过的长度限制隐藏在对照中。

Laya 对照沿用同一入口，将 backend 改为 `laya`，提供自己的本地 checkpoint，并显式使用 `--dtype float32 --batch-size 1`。这不是与 DLLM BF16 batch 32 相同的速度实验；无需为运行 S1 下载 Laya。

## 状态与分母

| 状态 | 含义 |
|---|---|
| prediction `ok` | 有有效概率及预测索引 |
| `unsupported_length` | 实际渲染后长度超过预算，无截断替代预测 |
| `unsupported_source_mask` | 来源中的 mask 与继承读出约束不兼容 |
| run `complete` | 所有预期记录均成功，评分完成 |
| `completed_with_unsupported` | 全成员都有记录，但包含 unsupported |
| `failed` | 真正异常；保留部分证据，不伪装成完整结果 |
| `predictions_complete_unscored` | 使用 `--predict-only`，尚未评分 |
| `full_status=not_full_smoke` | 仅 smoke，不论其子集是否成功 |

完整评分拒绝缺失、额外或重复 ID。unsupported 必须保留为空概率记录，不可静默 drop。概率指标只在有有效预测且有相应目标的记录上定义；all-attempted accuracy 将 unsupported 计错。跨模型比较应同时报告各自成功数、共同成功集和全部尝试分母，特别是 ContractNLI。

## 指标与计时

- Core 硬标签 accuracy 使用 argmax；score 的硬标签诊断使用原始离散化目标，不能代替标量/软目标指标。
- ECE 使用 15 个等宽置信度 bin；NLL 使用自然对数；Brier 为各类别平方误差之和，二元时是 `P(true)` 硬目标 MSE 的两倍。
- human-votes 与 exact-mechanism soft targets 分开解释；保留 soft 和 score 诊断，不将所有目标强行当成 one-hot。
- 各套分类数、任务与目标语义不同，不声称一个总分或 overall winner。Nimble12 等子集平均是等权宏平均，不是合并全体样本后的 ECE。
- batch 计时是继承实现的同步前向耗时，排除分词、传输和 I/O；逐条摊销时间不是单请求 HTTP 延迟。不同精度、设备、并发运行不可直接作匹配速度比较。

以 `scores/report.json` 的 `metric_definitions`、`suites` 和计数为准；公开 [JevBench CLI](jevbench_public.md) 使用 ECE10，不能与本节 ECE15 直接互换。

## 显式独立评分

正常 evaluate 会自动评分。若使用 `--predict-only`，可在推理完成后运行（输出仍需全新）：

```bash
python benchmark/score.py --data-root "$CORE_DATA" \
  --predictions "$EVAL_DIR/predictions.jsonl" \
  --run-manifest "$EVAL_DIR/run.json" \
  --output-dir "$EVAL_DIR/scores"
```

smoke 重评分须同时加 `--smoke`，并保留 run manifest。历史产物重评分是单独的 `--historical-root` / `--source-profile` 模式，输出标注 `historical_rescore`、`new_inference=false`，不能称为新推理。旧版六套 runner 的参数和输出见 [legacy reproduce](reproduce.md)，不可混用。
