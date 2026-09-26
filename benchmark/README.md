# Core benchmark

默认入口是固定的 `core_v1_candidate`：17 套、6,744 个决策。安装、下载 S1、准备和首次评测见[根 README](../README.md)。这不是旧版六套外部评测，也不是完整历史 20 套 profile。

## 准备与缓存

在仓库根目录执行，先阅读[来源条款](../docs/benchmark_sources.md)：

```bash
export CORE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/jev-dllm"
export CORE_DATA="$PWD/local_data/core-v1"
python scripts/prepare_core.py --cache-dir "$CORE_CACHE" --output-dir "$CORE_DATA"
```

缓存原始树为 `$CORE_CACHE/core-v1/raw`。脚本使用[固定来源配置](manifests/core_v1.json)，获取 14 个来源条目，再转换并验证冻结成员及顺序。它不下载 checkpoint；新原始数据和重建的文本数据不应提交到 Git。

可分两步获取与转换；两条命令共用同一个缓存，输出仍须是新目录：

```bash
python scripts/prepare_core.py --cache-dir "$CORE_CACHE" --download-only
python scripts/prepare_core.py --cache-dir "$CORE_CACHE" --offline \
  --output-dir "$PWD/local_data/core-v1-offline"
```

`--download-only` 只获取/检查来源文件，不验证转换输出等价性。`--offline` 缺文件即失败。`--raw-root /path/to/complete/raw` 可复用完整固定来源树，脚本不会向该显式目录补下载。公开准备回放历史选择，不重跑私有暴露审计。

## 评测与文件

```bash
mkdir -p outputs
python benchmark/evaluate.py --data-root "$CORE_DATA" \
  --backend dllm --model-path "$CHECKPOINT" \
  --device cuda:0 --dtype bfloat16 --batch-size 32 --max-length 4096 \
  --output-dir "$PWD/outputs/core-s1-new"
```

模型只收到 `state` / `qdef`；推理结束后另起进程评分。这是软件接口隔离，不是针对恶意模型代码的 OS 沙箱。

| 文件 | 用途 |
|---|---|
| `run.json` | 配置、预期 ID、各阶段与完整运行状态 |
| `inference.json` | 推理进度、实际 runtime、失败 batch |
| `predictions.jsonl` | 每个预期 ID 的概率或 unsupported 状态 |
| `batches.jsonl` | batch 成员及前向计时 |
| `scores/report.json` | 各套指标、覆盖与分母、指标定义 |

`--smoke --smoke-limit 3` 每套取前三条，必须另选输出目录；`full_status=not_full_smoke` 不能当作全量结果。默认进度间隔 10 batch，可设 `--log-every-batches 1`。`--predict-only` 只产生未评分预测。

完整运行要求全部 6,744 个 ID 有记录。`completed_with_unsupported` 表示遍历完整但有不能推理的输入，不是全数预测成功；真正异常标为 `failed`，保留部分证据且不自动评分。不要删掉失败记录后称为完整评测。

## 导航

- [数据卡](DATA_CARD.md)：套件计数、字段和用途
- [评测细节](../docs/benchmark_core.md)：指标、状态、独立评分
- [重叠审计](../docs/overlap_audit.md)：冻结选择和已知限制
- [HTTP 接口](JEVBENCH.md)与[官方公开集运行](../docs/jevbench_public.md)
- [结果](../docs/results.md)、[训练与 legacy 入口](../docs/training.md)
