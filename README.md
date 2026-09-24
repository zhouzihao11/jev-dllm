# Shared Yes/No DLLM 研究源码

将 masked diffusion language model 的一次前向计算用于动态候选选择、二元判断和有序评分：候选 mask 读取共享 Yes/No logits，得到结构化概率。本仓库整理 S0/S1 数据构建、监督训练和评测代码，不是新的通用训练框架。

**状态：私有研究源码与数据包；数据准备已验证，完整模型环境需另行准备。** 仓库包含 S0/S1 canonical、六套全量冻结评测 fixtures、manifest 和 14 份来源卡片。2026-09-24 已完成远端数据准备与独立 Parquet 计数验证；11 份输出与原研究文件逐字节一致，两份派生复制也完全一致，没有重新划分数据或改变 schema。完整模型流水线尚未在干净机器验证，训练/评测未重跑。私有纳入已获用户授权，不等于公开再分发许可已解决。权重、完整 raw 上游 train、外部模型 custom code、`dllm_stub` 和历史冻结代码 bundle 均不提供。

## 数据快速准备（无需网络或模型）

需要此 GitHub 私有仓库的访问权限。`git clone` 或 `git pull` 获取本版本时已包含压缩数据及 `datasets/manifest.json`，无需额外的数据下载链接。在仓库根目录运行：

```bash
python scripts/prepare_datasets.py --output-dir "$PWD/local_data"
export DATA_ROOT="$PWD/local_data"
```

只需 Python 标准库；已在现有 Python 3.10 环境验证，无需为数据准备安装模型依赖。目标目录必须不存在，不覆盖已有文件。可用 `--dataset-dir /absolute/path/to/datasets` 指定另一份资产副本；缺 manifest 会明确报错，不下载替代数据。11 份数据文件共 15,801,569 字节（约 15.07 MiB，四舍五入为 15.1 MiB），对应约 112.5 MiB 未压缩数据（另有两份 canonical 复制）。训练/评测入口不会自动读取这些 `.gz`，必须先解压。

| 用途 | 准备后路径（相对 `DATA_ROOT`） | 行数 / 决策数 |
|---|---|---|
| S0 train | `s0/canonical/train.jsonl` | 10,000 / 10,000 |
| S0 dev / test | `s0/canonical/dev.jsonl` / `s0/canonical/test.jsonl` | 各 1,000 / 1,000 |
| S1 完整 train | `s1/canonical/train.jsonl` | 40,000 / 40,000 |
| S1 选优 dev（原 S0 dev 的复制） | `s1/canonical/dev.jsonl` | 1,000 / 1,000 |
| S1 仅诊断 dev | `s1/canonical/new_dev.jsonl` | 2,000 / 2,000 |
| 六套外部全量评测 | `bench/`，文件明细见[数据说明](datasets/README.md) | 17,006 个决策 |
| 内部测试（原 S0 test 的复制） | `bench/internal_s0_test.jsonl` | 1,000 / 1,000 |

S1 完整 train 按字节保留 S0 train 加新增 delta；无需单独 `new_train`。使用已打包 canonical 训练/评测无需下载 raw 上游；只有从来源重建才需要完整 raw/prepared train。按[原训练配方与全量直接评测命令](docs/reproduce.md)运行，直接评测使用 `$DATA_ROOT/bench`；这些命令仍需另行准备合法模型与运行环境。helper 验证 JSONL/CSV 行数，对 Parquet 仅打印 manifest 声明；本版本另以 PyArrow 验证了 typed 的 400 行 / 2,000 决策和 PI 的 116 行，外部共 17,006 决策、内部 1,000 决策，记录见 [manifest](datasets/manifest.json) 的 `verification` 和[验证范围](docs/reproduce.md#6-本次验证范围)。根 Apache-2.0 不覆盖数据，见[来源](datasets/SOURCES.md)及[许可未决项](datasets/LICENSES.md)。

## 阅读入口

- [中文研究博客](docs/blog_zh.md)：实验方法、质量/吞吐汇总及失败边界。
- [实际 CLI 与依赖](docs/reproduce.md)：数据准备命令已验证；模型环境、训练与评测命令仍需另行准备，未在本次重跑。
- [数据与模型准备契约](docs/data_and_models.md)：原始来源、prepared 文件和全量评测 fixture。
- [Canonical schema](docs/SCHEMA.md)：一行一个决策，gold 与模型输入隔离。
- [首版范围与后续计划](docs/open_source_plan_zh.md)、[第三方归属与许可](THIRD_PARTY.md)。

## 研究边界

S1 记录运行是 40,000 条 train、原 1,000 条 dev 选优，新增 2,000 条 dev 仅诊断。最佳 checkpoint 为 step 1,300，不是 epoch 1 末尾的 step 1,250。六套外部评测共 17,006 个决策，内部 1,000 条另报。构建配额不保证任意输入都达到这些数量，需检查短缺和碰撞报告。

S1 相对 base 的收益依赖任务；Prompt Injection 退化，概率校准也非全面改善。BF16 batch 32 相对同模型 batch 1 的约 3.85 倍前向吞吐收益，不是单请求延迟，也不是 diffusion 对 AR 的架构优势。质量主表保持 batch 1；BF16 batching 有 72/17,006 个 argmax 翻转。

## 来源版本，不是本仓库历史

| 用途 | 原研究 SHA |
|---|---|
| 冻结质量评测，profile `full_test_v1_20260923_r2` | `fb672477594f29681b90d13ec679819ab8033c5c` |
| S1 构建/训练 | `80efa54951000a7b6f1637ea2bf146332540f9e2` |
| 本次精选源码基线，含批处理评测 | `5d1b7de221579af74bde3131bc4eb8e61bba7c0f` |

新仓库的首次提交不包含这些历史对象。历史质量 bundle 未交付，不能直接 checkout 上述 SHA，也不能把 `run_full_benchmark.py` 指向当前源码后称为原冻结快照。当前代码可用于明确标注版本的新评测；是否重现历史结果需独立验证。

## 归属与安装名称

继承 SDK 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，元数据作者为 Convai Innovations。保留 `laya/`、原 Apache-2.0 `LICENSE`、原 `pyproject.toml`/`setup.py` 元数据和源码归属。本研究增量不代表整个 SDK、上游模型或训练成果均为原创，也不表示上游背书。

外部 MDLM 来源为 [dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)。其权重、custom code 和兼容 stub 不随本仓库分发。

本仓库只做源码安装；`pip install -e .` 使用继承的包名 **laya 0.3.5**，并非发布新的 PyPI 包，不应以 `pip install shared-yesno-dllm` 安装。`research/` 不随 SDK wheel 打包，需从源码目录执行。首次私有源码整理沿用 Apache-2.0，新增文件权属仍需维护者在公开前确认；该许可证不扩展到任何数据或权重。

没有复制发布 workflows、notebooks、缓存、私有环境、模型、完整上游 raw train、预测或实验输出 JSON。私有数据仅包含明确白名单中的 canonical 与冻结 fixtures、manifest 和来源卡片。生成的报告可能包含原文、token IDs 和运行路径，应留在仓库外，不能直接公开。
