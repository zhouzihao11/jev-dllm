# 官方 JevBench 公开集

本流程使用未修改的上游 v1.4.2 harness，固定 revision `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`，协议 `jevbench::v1.4`。范围仅为 easy 48、original 72、hard 111，共 231 个公开任务；不含 sealed，不产生官方 v1.4 综合分或排名，不自动提交 GitHub issue。

HTTP 请求/响应见[协议说明](../benchmark/JEVBENCH.md)。本页是独立于 Core 本地评分器的高级流程，不需要把 Core 预测转换成官方结果。

## 复用固定 harness

先按[根 README](../README.md)安装、下载 checkpoint 和准备 Core；固定来源缓存已包括公开题和 CLI 模块，无需再 clone 上游：

```bash
export CORE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/jev-dllm"
export HARNESS="$CORE_CACHE/core-v1/raw/core/jevbench"
export HARNESS_REVISION=1bcc55eb6c8cffde2306b3db03ede39b61c6152a
export CODE_COMMIT="$(git rev-parse HEAD)"
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

`CODE_COMMIT` 必须来自当前公开仓库的 HEAD，不填写私有历史实验 SHA；它是运行身份记录，不替代对未提交修改的检查。缓存是选定文件快照，不要求它含上游 `.git`。结果目录必须位于 harness 之外。

## 模型矩阵与全量一致性验证

`--models` 接受 JSON 文件路径，不接受内联 JSON 参数。将下列示例保存为本地 `local_models.json`；从仓库根目录调用时，相对 checkpoint 路径相对当前目录解析。矩阵可增加 base/S0，但每个 name 必须唯一；不含凭据，不应提交机器专用路径。

```json
[
  {
    "name": "Shared-YesNo-Qwen3-0.6B-S1",
    "model_path": "models/shared-yesno-s1",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "max_length": 4096
  }
]
```

```bash
python benchmark/run_jevbench_public.py --mode validate \
  --harness-root "$HARNESS" --harness-revision "$HARNESS_REVISION" \
  --code-commit "$CODE_COMMIT" --models "$PWD/local_models.json" \
  --output-dir "$PWD/outputs/jev-public-validation"
```

输出必须全新。验证阶段为每个模型启动 loopback 服务，比较全部 231 题的直接调用、HTTP 以及官方 TypeSafeAdapter 概率映射；也检查三种 primitive、模型别名、非法 schema/model、超长拒绝、多问题独立映射和 usage。它不是只取几题的 smoke。直接模型与服务模型可能同时占用 GPU 显存，应预留两份加载的空间。

`identity.json` 冻结配置，`matrix.json` 记录各模型状态，模型子目录的 `validation.json` / `validation.jsonl` 保留逐题证据。成功验证不能替代正式结果；精确概率平局造成的 native/official argmax 差异单独记录。

## 全新服务运行官方 CLI

使用相同模型矩阵、代码提交和 harness 身份，再选择全新输出目录：

```bash
python benchmark/run_jevbench_public.py --mode run \
  --harness-root "$HARNESS" --harness-revision "$HARNESS_REVISION" \
  --code-commit "$CODE_COMMIT" --models "$PWD/local_models.json" \
  --validation-dir "$PWD/outputs/jev-public-validation" \
  --output-dir "$PWD/outputs/jev-public-results"
```

runner 要求匹配的 231 题验证通过，重新加载服务，再调用上游 `python -m jevbench.cli run` 和 `summarize`。batch 1、逐请求串行、T=1、原生概率；不修改 renderer/readout，不混入验证计时，不进行预测 warmup。运行完关闭服务。模型子目录包含：

| 文件 | 含义 |
|---|---|
| `results.jsonl` | 官方逐题结果及请求耗时 |
| `summary.json` | 官方 public export |
| `raw/`, `manifest.json`, `ledger.jsonl` | 官方响应证据、配置与预算账本 |
| `public_result.json` | 完整性、失败数、有效性与公开范围摘要 |
| `server.log`, `harness.log`, `summary.log` | 服务及官方 CLI 日志 |

实际产物可能包含模型路径和任务文本；分享前检查，不把 raw 目录或私有运行身份自动提交。

## 手动调用官方 CLI

已按[协议说明](../benchmark/JEVBENCH.md)在端口 8000 启动服务时，可直接运行官方命令。此方式不自动完成前述一致性验证。

```bash
export TASKS="$HARNESS/datasets/public/easy.jsonl,$HARNESS/datasets/public/original.jsonl,$HARNESS/datasets/public/hard.jsonl"
export OFFICIAL_OUT="$PWD/outputs/jev-public-manual"
mkdir -p outputs
mkdir "$OFFICIAL_OUT"
unset TYPESAFE_PRICE_INPUT_PER_M TYPESAFE_PRICE_OUTPUT_PER_M
PYTHONPATH="$HARNESS${PYTHONPATH:+:$PYTHONPATH}" python -m jevbench.cli run \
  --tasks "$TASKS" --adapter typesafe --endpoint http://127.0.0.1:8000 \
  --model Shared-YesNo-Qwen3-0.6B-S1 --key-env '' \
  --cost-basis self_hosted_no_billable_api --reserve-usd 0 --cap-usd 1 \
  --results "$OFFICIAL_OUT/results.jsonl" --raw-dir "$OFFICIAL_OUT/raw" \
  --ledger "$OFFICIAL_OUT/ledger.jsonl" --manifest "$OFFICIAL_OUT/manifest.json"
PYTHONPATH="$HARNESS${PYTHONPATH:+:$PYTHONPATH}" python -m jevbench.cli summarize \
  --tasks "$TASKS" --results "$OFFICIAL_OUT/results.jsonl" \
  --ledger "$OFFICIAL_OUT/ledger.jsonl" --public-export "$OFFICIAL_OUT/summary.json"
```

这里的 `PYTHONPATH` 只定位公开外部 harness，不是本项目私有路径。`reserve-usd=0` 表示无可计费 API 预留；自托管硬件/电力成本未知，不表示免费推理。未提供价格时保留未知成本，而非填零成本。

## 结果边界

官方摘要使用 ECE10；Core 使用 ECE15。官方 summary 内的 `jevbench-v1` 是继承的聚合文件格式标签，不应篡改为 v1.4；实际协议和 revision 由外层身份记录标明。NLL 和 HTTP mean latency 可另作补充诊断，但不是官方综合分。

延迟是本机 loopback 端到端请求时间，包含 HTTP、分词与推理，不是 GPU forward，也不是官方网络测量。已有公开结果见[结果说明](results.md)。公开题已被观察过；本流程不宣称盲测、预训练无污染、sealed 能力或官方排名。发布版远程复现验证尚待补充，历史成功不等于已验证此次干净安装。
