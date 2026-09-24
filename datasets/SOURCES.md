# 数据来源、归属与捕获卡片

[manifest.json](manifest.json) 记录本包的上游元数据、版本、协议说明及 `source_cards` 清单。以下 14 份捕获文件已随私有仓库提供；上游链接用于归属与条款核查，不是准备本包数据的额外下载步骤。卡片是捕获的来源声明，不是完整法律审查。原 revision 表与转换规则保留在[数据与模型说明](../docs/data_and_models.md)。

| 用途 / 来源 | 上游归属 | 已收录的来源材料 |
|---|---|---|
| S0 / S1 复用 CLINC | [CLINC / oos-eval](https://github.com/clinc/oos-eval) | [s0_clinc_LICENSE.txt](source_cards/s0_clinc_LICENSE.txt) |
| S0 / S1 复用 SNLI | [Stanford SNLI](https://nlp.stanford.edu/projects/snli/) | [s0_snli_official_page_declaration.txt](source_cards/s0_snli_official_page_declaration.txt) |
| S0 ARC | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | [s0_arc_README.md](source_cards/s0_arc_README.md) |
| S0 / S1 复用 SGD | [Google SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | [s0_sgd_LICENSE.txt](source_cards/s0_sgd_LICENSE.txt) |
| S1 DBpedia | [DBpedia 14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | [s1_dbpedia.README.md](source_cards/s1_dbpedia.README.md) |
| S1 GoEmotions | [Google GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | [s1_goemotions.README.md](source_cards/s1_goemotions.README.md) |
| S1 BoolQ | [Google BoolQ](https://huggingface.co/datasets/google/boolq) | [s1_boolq.README.md](source_cards/s1_boolq.README.md) |
| Typed 评测 | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | [bench_cards_typed_decisions_README.md](source_cards/bench_cards_typed_decisions_README.md) |
| 新闻评测 | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | [bench_cards_ag_news_README.md](source_cards/bench_cards_ag_news_README.md) |
| 情绪评测（不是 GoEmotions） | [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion) | [bench_cards_emotion_README.md](source_cards/bench_cards_emotion_README.md) |
| Banking77 评测 | [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | [bench_banking77_README.md](source_cards/bench_banking77_README.md) |
| 注入评测 | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | [bench_cards_prompt_injection_README.md](source_cards/bench_cards_prompt_injection_README.md) |
| SST5 评测 | [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) | [bench_cards_sst5_README.md](source_cards/bench_cards_sst5_README.md)、[bench_sst5_README.md](source_cards/bench_sst5_README.md) |

S0/S1 还包含程序生成的规则、状态和概率任务，其构建代码位于 `research/scripts/`。公共来源沿用既有标注，没有调用生成式 API 重新标注。打包保留原研究 canonical 和冻结 fixtures 的字节内容、顺序、划分与 schema，不是再次构建或重新划分数据。该包不是完整 raw 上游训练档案，也不替代上游通知义务。

许可声明与未决项见[第三方表](../THIRD_PARTY.md)和[LICENSES](LICENSES.md)。准备 helper 不联网获取新 revision 或卡片，也不作法律判断；完成数据计数和字节验证不代表完成个人信息或公开再分发审查。
