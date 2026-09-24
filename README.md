# Shared Yes/No DLLM 研究源码

将 masked diffusion language model 的一次前向计算用于动态候选选择、二元判断和有序评分：候选 mask 读取共享 Yes/No logits，得到结构化概率。本仓库整理 S0/S1 数据构建、监督训练和评测代码，不是新的通用训练框架。

**状态：私有首版源码准备中，未验证干净机器复现。** 不提供一键运行承诺。数据、权重、外部模型 custom code、`dllm_stub` 和历史冻结 bundle 均未上传；其访问与再分发许可仍需分别确认。

## 阅读入口

- [中文研究博客](docs/blog_zh.md)：实验方法、质量/吞吐汇总及失败边界。
- [实际 CLI 与依赖](docs/reproduce.md)：准备完成后的运行命令，本次未执行。
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

没有复制发布 workflows、notebooks、缓存、私有环境、模型、数据、预测或实验输出 JSON。生成的报告可能包含原文、token IDs 和运行路径，应留在仓库外，不能直接公开。
