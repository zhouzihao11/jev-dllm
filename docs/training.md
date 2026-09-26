# S0/S1 训练与旧版复现

只想评测或调用模型时，使用[根 README](../README.md)下载 S1 即可，不需要训练。本页保留原 README 的训练参数和旧版数据/runner 路径；默认 Core 6,744 条不用于训练。

## 环境与固定模型

先完成根 README 的 Python 3.10 / CUDA 12.1 依赖安装。在仓库根目录设置训练路径；legacy trainer 使用公开仓库内的导入路径，不需要任何私有目录：

```bash
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:$PWD/research/scripts:$PWD/support/dllm_stub${PYTHONPATH:+:$PYTHONPATH}"
export BASE_MODEL="$PWD/models/qwen3-mdlm"
export DATA_ROOT="$PWD/local_data/training-legacy"
export RUN_DIR="$PWD/outputs/s1"
export EVAL_DIR="$PWD/outputs/legacy-eval-s1"
mkdir -p "$(dirname "$BASE_MODEL")" "$(dirname "$RUN_DIR")" "$(dirname "$EVAL_DIR")"
hf download dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --revision c8d24a3f4adaeef46881b450e1bf7d1005203bd7 \
  --local-dir "$BASE_MODEL"
```

加载使用 `trust_remote_code=True`。审阅固定 checkpoint 的自定义代码；完成所需下载后，可设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线运行。

直接下载 S0 作对照：

```bash
hf download SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S0 \
  --revision 265560cdcea68443fed89525437f4be7235574db \
  --local-dir "$PWD/models/shared-yesno-s0"
```

S1 固定 revision 为 `1f1c29ff9fc6f6e9dc066b03089878a7dab8b6a0`，下载命令见根 README。S0/S1 训练均从原始 `BASE_MODEL` 开始，不将此 S1 命令改称从 S0 继续训练。

## 准备旧 canonical 数据

仓库已有压缩数据，输出路径必须不存在：

```bash
python scripts/prepare_datasets.py --output-dir "$DATA_ROOT"
```

| 用途 | 相对 DATA_ROOT 的路径 | 决策数 |
|---|---|---:|
| S0 train / dev / test | `s0/canonical/{train,dev,test}.jsonl` | 10,000 / 1,000 / 1,000 |
| S1 train / 选优 dev / 诊断 dev | `s1/canonical/{train,dev,new_dev}.jsonl` | 40,000 / 1,000 / 2,000 |
| 旧六套外部 / 内部 S0 test | `bench/` / `bench/internal_s0_test.jsonl` | 17,006 / 1,000 |

S1 train 包含 S0 train；选优 dev 使用 S0 dev。文件映射见[数据说明](../datasets/README.md)。此旧 bundle 与 `scripts/prepare_core.py` 生成的 Core bundle 不可互换。

## 训练 S1

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

有效 batch 为 32；按 dev mean KL 选择 `$RUN_DIR/best`，恢复点位于 `$RUN_DIR/resume-latest`。训练需要 GPU，按设备调整 `CUDA_VISIBLE_DEVICES`。评测自己训练的模型时设置 `export CHECKPOINT="$RUN_DIR/best"`，然后可运行根 README 的 Core 评测。

S0 将 train/dev 换成 `s0/canonical/train.jsonl` 与 `s0/canonical/dev.jsonl`，仍从 `BASE_MODEL` 开始，另选 `RUN_DIR`。单卡设置 `--nproc_per_node=1 --micro-batch 4 --grad-accum 8` 保持有效 batch 32。恢复用 `--resume "$RUN_DIR/resume-latest"`，保留原配置；训练 smoke 可加 `--train-limit 64 --dev-limit 16 --max-updates 2` 并另选输出目录，不能用于质量比较。

## Legacy 六套评测

以下是原 README 的旧入口，不是 Core，也不是官方 JevBench HTTP：

```bash
CUDA_VISIBLE_DEVICES=0 python research/scripts/run_full_benchmark.py \
  --data-root "$DATA_ROOT" --backend dllm \
  --model-path "$CHECKPOINT" --output-dir "$EVAL_DIR" \
  --device cuda:0 --dtype bfloat16 --batch-size 1
```

使用全新 `EVAL_DIR`。旧 runner 评测六套外部任务 17,006 个决策，内部 S0 test 1,000 个单独报告。输出 `run.json`、`external.json`、`internal.json`、`external_predictions.jsonl`、`internal_predictions.jsonl`，与 Core 的 `scores/report.json` 不同。

[docs/reproduce.md](reproduce.md) 保留原高级 API、恢复、旧 batch/吞吐与 Laya 参数说明；其中“README 训练命令”和旧路径变量现在对应本页，而非新的根 README Core 命令。旧文档不作为新 Core flags 的参考。数据重建见 [raw/prepared 参考](data_and_models.md)、[S1 builder](../research/S1_DATA_BUILDER.md) 和 [schema](SCHEMA.md)。

这些是继承参数与运行入口说明，不声称此次发布已通过干净环境训练验证。
