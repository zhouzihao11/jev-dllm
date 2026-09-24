# Canonical 数据与评测输入

本目录包含 S0/S1 训练数据及六套完整外部评测输入，随仓库获取。在仓库根目录解压：

```bash
python scripts/prepare_datasets.py --output-dir "$PWD/local_data"
```

仅需 Python 标准库；目标目录必须不存在。`--dataset-dir` 可指定另一份资产目录。训练与评测读取解压后的文件。

## 准备后的文件

以下路径相对 `local_data/`；JSONL/CSV 从 gzip 解压，Parquet 按原文件复制。

| 路径 | 行数 / 决策数 |
|---|---:|
| `s0/canonical/train.jsonl` | 10,000 / 10,000 |
| `s0/canonical/dev.jsonl` | 1,000 / 1,000 |
| `s0/canonical/test.jsonl` | 1,000 / 1,000 |
| `s1/canonical/train.jsonl` | 40,000 / 40,000 |
| `s1/canonical/dev.jsonl` | 1,000 / 1,000 |
| `s1/canonical/new_dev.jsonl` | 2,000 / 2,000 |
| `bench/ag_news_test.jsonl` | 7,600 / 7,600 |
| `bench/emotion_test.jsonl` | 2,000 / 2,000 |
| `bench/sst5_test.jsonl` | 2,210 / 2,210 |
| `bench/banking77_test.csv` | 3,080 / 3,080 |
| `bench/typed_decisions_test.parquet` | 400 / 2,000 |
| `bench/prompt_injection_test.parquet` | 116 / 116 |
| `bench/internal_s0_test.jsonl` | 1,000 / 1,000 |

S1 train 已包含 S0 train。S1 选优 dev 与内部 test 分别复制 S0 dev/test；`new_dev` 仅诊断。六套外部共 17,006 个决策，内部 1,000 个另报。

文件映射与计数见 [manifest.json](manifest.json)。

训练与全量评测见[快速开始](../README.md)，字段格式见 [schema](../docs/SCHEMA.md)，从来源重建见[构建参考](../docs/data_and_models.md)。
