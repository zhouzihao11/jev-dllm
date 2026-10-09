# Jev-DLLM: Shared Yes/No Decision Models

在每个候选旁放置 mask，用 masked diffusion language model 的共享 Yes/No 权重读取结构化概率：动态选择、二元判断和有序评分。无需生成答案文本，也不需要先训练即可体验。

[代码](https://github.com/zhouzihao11/jev-dllm) · [S1 模型](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1) · [实验结果表](docs/experiment_results.md) · [结果与限制](docs/results.md) · [Kev 五模型比较](docs/kev_comparison.md) · [S1 温度校准](docs/temperature_calibration.md) · [Dream zero-shot](docs/dream_zero_shot.md)

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

## 6. 可选 Kev-0.8B 比较

Kev 是独立的原生概率头对照，不改变上述 S1 安装、下载、评测和 HTTP quickstart。按 [Kev 复现流程](benchmark/KEV.md)准备固定代码/模型和独立依赖环境，无需启动完整上游 serve；主 `requirements.txt` 不包含 Kev 专用版本。

```bash
export KEV_DIR="$HOME/model-resources/jev-dllm-kev"
python scripts/prepare_kev.py --output-dir "$KEV_DIR"
```

保持原 benchmark 环境不变，另建 Python 3.12 环境并安装可选依赖；安装细节与限制见上述复现流程：

```bash
conda create -y -n kev-eval python=3.12
PYTHONNOUSERSITE=1 conda run -n kev-eval python -m pip install -r benchmark/requirements-kev.txt
export KEV_PYTHON="$(conda run -n kev-eval python -c 'import sys; print(sys.executable)')"
```

再从原 benchmark 环境启动；推理使用 `KEV_PYTHON`，评分仍在原环境执行：

```bash
python benchmark/evaluate.py \
  --data-root "$CORE_DATA" --backend kev \
  --backend-python "${KEV_PYTHON:?set from benchmark/KEV.md}" \
  --kev-source "$KEV_DIR/source" \
  --base-path "$KEV_DIR/models/base" \
  --model-path "$KEV_DIR/models/kev-0.8b" \
  --output-dir "$PWD/outputs/core-kev" --device cuda:0 --dtype float32 \
  --batch-size 1 --max-length 4096
```

输出目录必须不存在；可另用新目录加 `--smoke --smoke-limit 3`。不传 `--kev-temperature`，保留 checkpoint 校准。该命令运行完整 Core17 并保留所有覆盖记录，发布质量比较再按下面来源范围筛选，不静默删样本。旧版 typed-decisions 的单独准备与运行命令也见 [Kev 复现流程](benchmark/KEV.md)。

[比较结果](docs/kev_comparison.md)仅列 Core14（排除已确认训练见源 MultiNLI/Aegis2 与开发接触 PAWS），另列 legacy typed-decisions；保留项未确认来源重叠，不等于保证无污染。其他四模型保留旧 renderer 预测，Kev 用 checkpoint 温度，不能混入较新的 JevBench HTTP 结果、解释为架构因果或公平吞吐排名。历史 Kev 推理成功不代表本次发布迁移已重新运行验证。

## 更多

- [Dream-7B zero-shot Core14](docs/dream_zero_shot.md)：同分母平均ACC64.08%、ECE15 17.74%、NLL0.9186、Brier0.4817；相较0.6B base更强，与S1总体ACC接近。仅发布已完成zero-shot结果，不含后续微调或公开Dream运行入口。
- [S1 全局温度校准与 Core14 比较](docs/temperature_calibration.md)：六套拟合 T=1.87397；同分母平均 ECE15 10.33%、NLL 0.8808、Brier 0.4637；部分任务退化。仅发布后处理结果，默认评测/HTTP 仍为 T=1。
- [Core 操作与输出](benchmark/README.md)、[数据卡](benchmark/DATA_CARD.md)、[重叠审计范围](docs/overlap_audit.md)
- [官方 JevBench 公开 231 题流程](docs/jevbench_public.md)：先验证，再通过官方 CLI 运行；不含 sealed 或排行榜提交
- [五模型来源筛选比较与既有公开 HTTP 结果](docs/results.md)：同时报告准确率、校准、失败覆盖和协议边界
- [S0/S1 训练](docs/training.md)、[旧版六套 benchmark / 恢复 API](docs/reproduce.md)（legacy，不是本页 Core 命令）
- [研究博客（历史背景）](docs/blog_zh.md)、[旧数据与模型参考](docs/data_and_models.md)、[字段说明](docs/SCHEMA.md)

继承 SDK 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)。许可与归属见 [LICENSE](LICENSE)、[THIRD_PARTY](THIRD_PARTY.md)。
