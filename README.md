# Jev-DLLM: Shared Yes/No Decision Models

在每个候选旁放置一个 mask，用 masked diffusion language model 的共享 Yes/No 权重一次前向得到结构化概率，支持动态选择、二元判断和有序评分。
本仓库包含代码、S0/S1 训练数据与六套评测输入；已发布的 [S0](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S0) / [S1](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1) 模型可直接下载，方法与结果见[研究博客](docs/blog_zh.md)。

## 1. 安装与路径

在仓库根目录执行，使用 Python 3.10、NVIDIA GPU 和兼容 CUDA 12.1 的驱动：

```bash
conda create -n shared-yesno python=3.10 -y
conda activate shared-yesno
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/research/scripts:$PWD/support/dllm_stub${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false

export BASE_MODEL="$PWD/models/qwen3-mdlm"
export DATA_ROOT="$PWD/local_data"
export RUN_DIR="$PWD/outputs/s1"
export EVAL_DIR="$PWD/outputs/eval_s1"
mkdir -p "$(dirname "$BASE_MODEL")" "$(dirname "$RUN_DIR")" "$(dirname "$EVAL_DIR")"
```

## 2. 选择模型

**直接评测 S1：** 下载已训练 checkpoint，并在第 5 节选择它：

```bash
hf download SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1 --local-dir "$PWD/models/shared-yesno-s1"
export CHECKPOINT="$PWD/models/shared-yesno-s1"
```

也可下载 [S0](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S0)，将 `CHECKPOINT` 指向 S0 的下载目录分别评测。

**从原始模型训练：** 下载包含 tokenizer/custom code 的固定版本 MDLM 到 `BASE_MODEL`：

```bash
hf download dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --revision c8d24a3f4adaeef46881b450e1bf7d1005203bd7 \
  --local-dir "$BASE_MODEL"
```

下载完成后，如需离线运行，再设置 `export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。加载模型会使用 `trust_remote_code=True`。

## 3. 准备数据

仓库已包含压缩数据，准备到全新目录（目标路径不能已存在）：

```bash
python scripts/prepare_datasets.py --output-dir "$DATA_ROOT"
```

| 用途 | 相对 `DATA_ROOT` 的路径 | 决策数 |
|---|---|---:|
| S0 train / dev / test | `s0/canonical/{train,dev,test}.jsonl` | 10,000 / 1,000 / 1,000 |
| S1 train / 选优 dev / 诊断 dev | `s1/canonical/{train,dev,new_dev}.jsonl` | 40,000 / 1,000 / 2,000 |
| 六套外部评测 / 内部 S0 test | `bench/` / `bench/internal_s0_test.jsonl` | 17,006 / 1,000 |

S1 train 包含 S0 train；选优 dev 使用 S0 dev。文件映射见[数据说明](datasets/README.md)。

## 4. 训练 S1

从 `BASE_MODEL` 开始；按实际设备调整 `CUDA_VISIBLE_DEVICES`，单卡配置见[高级用法](docs/reproduce.md)。

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

有效 batch 为 32；训练按 dev mean KL 将选中模型保存于 `$RUN_DIR/best`，并在 `$RUN_DIR/resume-latest` 保存恢复点。评测自己训练的模型时设置 `export CHECKPOINT="$RUN_DIR/best"`。

## 5. 统一评测

选择第 2 节下载的 S0/S1 checkpoint，或第 4 节训练的 `best/`，设置 `CHECKPOINT` 后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python research/scripts/run_full_benchmark.py \
  --data-root "$DATA_ROOT" --backend dllm \
  --model-path "$CHECKPOINT" --output-dir "$EVAL_DIR" \
  --device cuda:0 --dtype bfloat16 --batch-size 1
```

runner 评测六套外部任务（17,006 个决策）和单独报告的内部 S0 test（1,000 个决策）；输出目录必须全新。结果在 `$EVAL_DIR/run.json`、`$EVAL_DIR/external.json` 和 `$EVAL_DIR/internal.json`，逐决策结果在同目录的 `external_predictions.jsonl` 与 `internal_predictions.jsonl`。快速查看各套指标：

```bash
python - "$EVAL_DIR/external.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    suites = json.load(f)["suites"]
for name, result in suites.items():
    print(name, {k: result[k] for k in ("n_decisions", "accuracy", "nll", "ece", "ms_per_decision")})
PY
```

单卡、恢复、smoke、外部 batch 32 吞吐对照与 Laya 原生对照见[高级用法](docs/reproduce.md)；batch 32 均摊前向耗时不是单请求延迟。数据重建见[raw/prepared 参考](docs/data_and_models.md)，字段见 [schema](docs/SCHEMA.md)。继承 SDK 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)；许可与归属见 [LICENSE](LICENSE) 和 [THIRD_PARTY](THIRD_PARTY.md)。
