# Jev-DLLM: Shared Yes/No Decision Models

在每个候选旁放置一个 mask，用 masked diffusion language model 的共享 Yes/No 权重一次前向得到结构化概率，支持动态选择、二元判断和有序评分。
本仓库提供 S0/S1 数据、监督训练和六套外部评测入口；方法与实验结果见[中文博客](docs/blog_zh.md)。

## 1. 安装环境

以下命令均在仓库根目录执行。推荐 Linux、NVIDIA GPU 与支持 CUDA 12.1 wheel 的驱动，使用 Python 3.10。
[requirements.txt](requirements.txt) 固定实际研究运行环境中的相关包版本（Python 3.10.20），并包含 PyTorch CUDA wheel 索引；它不是完整 pip freeze，也尚未经过干净机器安装验证。

```bash
conda create -n shared-yesno python=3.10 -y
conda activate shared-yesno
python -m pip install -r requirements.txt
```

从源码导入研究脚本与 SDK：

```bash
export PYTHONPATH="$PWD:$PWD/research/scripts:$PWD/support/dllm_stub${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
```

主流程不要求安装 SDK 包。如需 editable 安装，可在上述依赖安装后运行 `python -m pip install --no-deps -e .`；继承的包名和版本仍为 `laya 0.3.5`。

## 2. 下载原始模型

先统一设置本次运行路径：

```bash
export BASE_MODEL="$PWD/models/qwen3-mdlm"
export DATA_ROOT="$PWD/local_data"
export RUN_DIR="$PWD/outputs/s1"
export EVAL_DIR="$PWD/outputs/eval_s1"
mkdir -p "$(dirname "$BASE_MODEL")" "$(dirname "$RUN_DIR")" "$(dirname "$EVAL_DIR")"
```

模型权重不随仓库提供；下载原始 MDLM（不是 S0 checkpoint）及其 tokenizer/custom code：

```bash
hf download dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --revision c8d24a3f4adaeef46881b450e1bf7d1005203bd7 \
  --local-dir "$BASE_MODEL"
```

此 revision 来自原研究记录。加载器使用 `trust_remote_code=True`，请审阅下载的模型代码。下载完成后再启用离线加载：

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 3. 准备数据

仓库已包含压缩的训练数据与完整评测输入，无需另行下载 raw 上游数据：

```bash
python scripts/prepare_datasets.py --output-dir "$DATA_ROOT"
```

该 helper 仅需 Python 标准库；目标目录必须不存在。训练与评测读取解压后的文件，不直接读取 `.gz`。

| 用途 | 路径（相对 `DATA_ROOT`） | 决策数 |
|---|---|---:|
| S0 train / dev / test | `s0/canonical/{train,dev,test}.jsonl` | 10,000 / 1,000 / 1,000 |
| S1 train | `s1/canonical/train.jsonl` | 40,000 |
| S1 选优 dev | `s1/canonical/dev.jsonl` | 1,000 |
| S1 诊断 dev | `s1/canonical/new_dev.jsonl` | 2,000 |
| 六套外部评测 | `bench/` | 17,006 |
| 内部测试 | `bench/internal_s0_test.jsonl` | 1,000 |

S1 train 已包含 S0 train；选优 dev 是原 S0 dev，`new_dev` 仅诊断。完整文件表见[数据说明](datasets/README.md)。

## 4. 训练 S1

下面显式选择 `supervised` / `shared_yesno`，以原始 base 开始训练。GPU 编号 `0,1` 请按可用设备调整；单卡配置见[高级用法](docs/reproduce.md)。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  research/scripts/train_qwen_masked_typed.py \
  --model-path "$BASE_MODEL" \
  --train-data "$DATA_ROOT/s1/canonical/train.jsonl" \
  --dev-data "$DATA_ROOT/s1/canonical/dev.jsonl" \
  --output-dir "$RUN_DIR" \
  --loss-mode supervised --scoring-mode shared_yesno \
  --seed 42 --epochs 3 --micro-batch 4 --grad-accum 4 \
  --learning-rate 2.5e-5 --min-lr 1e-6 --weight-decay 0.01 \
  --warmup-ratio 0.05 --rps-weight 0.25 \
  --eval-every 100 --save-every 200 \
  --save-epoch-checkpoints --save-predictions \
  --max-length 4096 --amp-dtype bfloat16
```

有效 batch 为 `2 × 4 × 4 = 32`。新训练拒绝非空输出目录；`best/` 按 dev mean KL 选优，`epoch-*` 保存各 epoch checkpoint。

## 5. 全量评测

待测 checkpoint 与原始 base 分开指定：

```bash
export CHECKPOINT="$RUN_DIR/best"
```

统一入口使用当前源码、当前 Python 和 HOME/HF 缓存，无需历史冻结代码 bundle。`--data-root` 指向 helper 输出目录，默认全量运行六套外部评测及内部 S0 test，所有 test limit 为 0：

```bash
CUDA_VISIBLE_DEVICES=0 python research/scripts/run_full_benchmark.py \
  --data-root "$DATA_ROOT" --backend dllm \
  --model-path "$CHECKPOINT" \
  --output-dir "$EVAL_DIR" \
  --device cuda:0 --dtype bfloat16 \
  --batch-size 1 --warmup-batches 0
```

质量比较使用 DLLM batch 1；吞吐对照改为 `--batch-size 32`，仅外部评测批处理，内部始终 batch 1。`--warmup-batches` 默认 0，可设为 3，只对每套外部数据首批做不计分、不计时的重复预热；吞吐比较保持预热配置一致。FP32 排查改为 `--dtype float32`。每次另选全新 `--output-dir`，不要预先创建该目录，runner 拒绝覆盖。仅执行检查时添加 `--smoke`，不用于质量比较。

Laya 使用同一入口，改为 `--backend laya` 并提供 Laya checkpoint；保留原生 FP32 与 `--laya-max-seqs 16 --laya-max-tokens 8192`。四种条件中 `aligned_unit` 为主结果；Laya 不接受非默认 DLLM batch/warmup 配置，输出格式见[高级用法](docs/reproduce.md)。

## 6. 查看输出

| 输出 | 内容 |
|---|---|
| `$RUN_DIR/best/` | dev KL 最佳模型与 tokenizer |
| `$RUN_DIR/resume-latest/` | 可恢复训练的完整状态 |
| `$EVAL_DIR/run.json` | 模式、源码版本、精度、外部/内部 batch、计划/实际计数与阶段状态 |
| `$EVAL_DIR/external.json` | 分 suite 的质量、计时与计数 |
| `$EVAL_DIR/external_predictions.jsonl` | 逐决策概率与预测 |
| `$EVAL_DIR/internal.json` | 内部 S0 test 指标与分组统计 |
| `$EVAL_DIR/internal_predictions.jsonl` | 内部逐决策概率与预测 |

快速查看各 suite 指标及总决策数：

```bash
python - "$EVAL_DIR/external.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    suites = json.load(f)["suites"]
for name, result in suites.items():
    print(name, {k: result[k] for k in
                 ("n_decisions", "accuracy", "nll", "ece", "ms_per_decision")})
print("total decisions:", sum(r["n_decisions"] for r in suites.values()))
PY
```

全量外部应为 17,006 个决策；内部 1,000 条由 runner 自动评测、单独报告。`run.json` 的 `status` 与 `full_status` 均为 `complete` 才表示全量计数检查通过；smoke 的 `full_status` 为 `not_full_smoke`。失败时保留输出、退出码与错误详情。单卡、恢复训练、内部评测、smoke 与计时解释见[高级用法](docs/reproduce.md)。

## FAQ 与参考

**为什么有 `dllm_stub`？** 原模型 `modeling_qwen3.py` 的 `dllm` 导入仅位于 `if __name__ == '__main__':` 演示代码中，模型类本身使用 torch 与 transformers。Transformers 4.57 的动态代码依赖扫描会检查该演示导入，因此本项目提供仅含 docstring 的兼容导入标记；它不提供完整 diffusion 训练或生成库。

- [研究博客](docs/blog_zh.md)：设计动机、质量与吞吐结果。
- [研究脚本索引](research/README.md)、[Canonical schema](docs/SCHEMA.md)。
- [从来源重建数据](docs/data_and_models.md)：仅在需要重新构建时使用。

继承 SDK 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，保留 Convai Innovations 元数据与 [Apache-2.0 LICENSE](LICENSE)。第三方归属见 [THIRD_PARTY](THIRD_PARTY.md)，数据来源与许可见 [SOURCES](datasets/SOURCES.md) / [LICENSES](datasets/LICENSES.md)。
