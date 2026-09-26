# Jev-DLLM: Shared Yes/No Decision Models

在每个候选旁放置 mask，用 masked diffusion language model 的共享 Yes/No 权重读取结构化概率：动态选择、二元判断和有序评分。无需生成答案文本，也不需要先训练即可体验。

[代码](https://github.com/zhouzihao11/jev-dllm) · [S1 模型](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1) · [结果与限制](docs/results.md)

## 1. 安装

首次使用先获取仓库；已有本地仓库则跳过前两行，其余命令在仓库根目录执行。使用 Python 3.10、支持 BF16 的 NVIDIA GPU 和兼容 CUDA 12.1 的驱动；显存需求随输入长度和 batch 改变。

```bash
git clone https://github.com/zhouzihao11/jev-dllm.git
cd jev-dllm
conda create -n shared-yesno python=3.10 -y
conda activate shared-yesno
python -m pip install -r requirements.txt
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
```

入口自动处理仓库和 DLLM stub 的导入路径，无需手动设置私有 `PYTHONPATH`。已通过全新来源下载、独立代码目录重建、S1 全量评测和 HTTP 验证，见[验证记录](docs/release_validation.md)。验证复用了已有依赖环境，未验证从零安装所有依赖；`requirements.txt` 不是完整锁文件。

## 2. 下载 S1

```bash
export CHECKPOINT="$PWD/models/shared-yesno-s1"
hf download SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1 \
  --revision 1f1c29ff9fc6f6e9dc066b03089878a7dab8b6a0 \
  --local-dir "$CHECKPOINT"
```

加载器使用 `trust_remote_code=True`，会执行 checkpoint 中的自定义模型代码；请审阅并信任固定来源。S0 和原始 MDLM 的固定版本见[训练说明](docs/training.md)。

## 3. 准备 Core benchmark

先阅读[来源与使用条款](docs/benchmark_sources.md)，再按自己的使用权限下载。准备脚本获取 14 个固定来源条目（不是模型），重建 17 套、6,744 个决策；原始数据留在仓库外缓存，不向 Git 添加新原始数据。

```bash
export CORE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/jev-dllm"
export CORE_DATA="$PWD/local_data/core-v1"
python scripts/prepare_core.py \
  --cache-dir "$CORE_CACHE" --output-dir "$PWD/local_data/core-v1"
```

输出目录必须不存在。该步骤回放冻结的审计成员选择，不重新运行完整私有暴露审计；仅相对已知 S0/S1 训练及 dev 暴露审计，预训练重叠未知，公开题也已被观察过。详见[数据卡](benchmark/DATA_CARD.md)。

## 4. 评测

```bash
mkdir -p outputs
export EVAL_DIR="$PWD/outputs/core-s1"
python benchmark/evaluate.py \
  --data-root "$CORE_DATA" --backend dllm --model-path "$CHECKPOINT" \
  --output-dir "$EVAL_DIR" --device cuda:0 --dtype bfloat16 \
  --batch-size 32 --max-length 4096
```

每次换一个不存在的输出目录。可先添加 `--smoke --smoke-limit 3`，并改用 `outputs/core-s1-smoke`（每套前三条，不是完整结果）。默认每 10 batch 及套件边界打印进度；加 `--log-every-batches 1` 可逐 batch 查看。显存不足时减小 batch，并记录配置变化。

结果位于 `$EVAL_DIR/scores/report.json`，逐条预测为 `$EVAL_DIR/predictions.jsonl`。超长输入记录为 `unsupported_length`，不截断、不静默丢弃；覆盖率和准确率分母见[评测说明](docs/benchmark_core.md)。

## 5. 启动原生概率 HTTP

评测结束后，在同一环境运行：

```bash
python benchmark/serve.py \
  --model-path "$CHECKPOINT" --model-name Shared-YesNo-Qwen3-0.6B-S1 \
  --device cuda:0 --dtype bfloat16 --max-length 4096 --port 8000
```

看到 ready 后，在另一个终端请求（仅监听 `127.0.0.1`）：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{"state":"The delivery arrived two days late.","model":"Shared-YesNo-Qwen3-0.6B-S1","questions":{"sentiment":{"type":"choice","instructions":"Classify the customer sentiment.","criteria":{"Positive":"The customer is satisfied.","Negative":"The customer is dissatisfied."}}}}'
```

返回 `answers.sentiment.choice` 和以原始候选字符串为键的 `probabilities`。这是原生 shared Yes/No 读出，不是从生成文本伪造概率。三种问题格式与限制见 [HTTP 协议](benchmark/JEVBENCH.md)；服务用 Python 标准库，无需 FastAPI。

## 更多

- [Core 操作与输出](benchmark/README.md)、[数据卡](benchmark/DATA_CARD.md)、[重叠审计范围](docs/overlap_audit.md)
- [官方 JevBench 公开 231 题流程](docs/jevbench_public.md)：先验证，再通过官方 CLI 运行；不含 sealed 或排行榜提交
- [四模型 Core 与公开 HTTP 结果](docs/results.md)：同时报告准确率、校准和失败覆盖
- [S0/S1 训练](docs/training.md)、[旧版六套 benchmark / 恢复 API](docs/reproduce.md)（legacy，不是本页 Core 命令）
- [研究博客（历史背景）](docs/blog_zh.md)、[旧数据与模型参考](docs/data_and_models.md)、[字段说明](docs/SCHEMA.md)

继承 SDK 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)。许可与归属见 [LICENSE](LICENSE)、[THIRD_PARTY](THIRD_PARTY.md)。
