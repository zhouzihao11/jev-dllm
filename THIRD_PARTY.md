# 第三方归属与许可边界

本表记录来源与许可信息；代码许可证不自动覆盖训练文本、评测文本、模型权重、tokenizer 或外部 custom code。具体许可信息以各来源说明为准。

## 随源码保留

- Laya SDK：上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，包元数据署名 Convai Innovations，Apache-2.0。`laya/` 原样保留，根 `LICENSE` 原样保留；源树没有根 `NOTICE`，未虚构上游通知文件。
- `pyproject.toml` / `setup.py` 保留继承元数据；其项目名、版本和上游 URL 不是本研究的新包发布声明。
- 研究脚本包含上游 suite/SDK 工具与 Shared Yes/No 增量；构建数据的来源版本见[数据 manifest](datasets/manifest.json)。
- `support/dllm_stub/dllm/__init__.py` 是动态模型代码导入检查所需的兼容标记，不包含上游 `dllm` 库实现；环境配置见 [README](README.md#1-安装与路径)。
- 新增文档和整理增量沿用根目录 Apache-2.0；这不改变第三方资产的许可。

## 外部模型，不随仓库分发

| 对象 | 来源 | 已知版本/边界 |
|---|---|---|
| MDLM base / tokenizer | [dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) | README 使用 revision `c8d24a3f4adaeef46881b450e1bf7d1005203bd7` |
| Laya 英文 checkpoint | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | checkpoint revision 未在本副本锁定；不冒充可逐位复现 |
| MDLM custom code | 随上述外部模型下载 | 未复制进本仓库，适用上游条款；本项目兼容导入标记不替代这些模型类 |
| S0/S1 权重 | [S0](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S0) / [S1](https://huggingface.co/SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1) | 模型已公开，权重不随源码仓库分发；许可见各模型页面，不能从源码 Apache 推导 |

模型加载使用 `trust_remote_code=True`，即使 `local_files_only=True` 仍会执行本地模型代码。先审阅来源与代码；不要把本仓库的许可证当作执行外部代码的安全保证。

## 数据来源

本仓库公开提供 canonical、冻结 fixtures、manifest 和来源卡片；以下是捕获的来源声明，具体许可信息以各来源说明为准。根 Apache-2.0 不重新许可数据。见[来源说明](datasets/SOURCES.md)和[许可说明](datasets/LICENSES.md)。

| 数据 | 官方/记录来源 | 声明与未决项 |
|---|---|---|
| CLINC | [clinc/oos-eval](https://github.com/clinc/oos-eval) | CC BY 3.0；核对版本及归属 |
| SNLI | [Stanford SNLI](https://nlp.stanford.edu/projects/snli/) | corpus-level CC BY-SA 4.0 记录；保留底层来源义务 |
| ARC | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA 4.0 记录，需版本级复核 |
| SGD | [Google SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | CC BY-SA 4.0 记录 |
| DBpedia | [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | CC BY-SA 3.0 / GNU FDL 注意事项，核对导出和底层内容 |
| GoEmotions | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache-2.0 记录，仍核对数据卡与归属 |
| BoolQ | [google/boolq](https://huggingface.co/datasets/google/boolq) | CC BY-SA 3.0 记录 |
| typed-decisions | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 记录，底层权利未完整审查 |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | 原记录许可不清晰，需回查原始发布来源 |
| Emotion | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | education/research/other 等信息需核实，非 GoEmotions |
| Banking77 | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | CC BY 4.0 记录；本协议使用原版 3,080 行 test CSV，不替换成裁剪镜像 |
| Prompt Injection | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 捕获的卡片存在 Apache / CC BY 4.0 声明差异，具体条款以来源为准 |
| SST5 | [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) | 具体使用导出及许可未完整锁定，不推断通用再分发许可 |

仓库不含完整 raw 上游 train、模型权重、示例预测或运行日志。依赖库由使用者另行安装，其许可证各自适用。
