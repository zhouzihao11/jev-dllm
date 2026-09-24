# 私有 canonical 数据与冻结评测输入

本目录已包含 [manifest.json](manifest.json)、11 份数据文件和 14 份来源卡片，不是待下载的占位目录。有 GitHub 私有仓库访问权限后，`git clone` 或 `git pull` 即可获取本版本的数据，无需额外下载链接。这是授权私有纳入，不是公开数据集发布或完整许可审查。

2026-09-24 已在远端现有 Python 3.10 环境执行标准库 helper，并单独以 PyArrow 验证 Parquet。11 份输出与原研究文件逐字节一致，两份派生复制也完全一致；文本、标签、顺序、原划分和 schema 均未改变。没有重跑模型训练/评测，也没有进行环境安装；完整干净机器模型流水线仍未验证。

在仓库根目录运行：

```bash
python scripts/prepare_datasets.py --output-dir "$PWD/local_data"
export DATA_ROOT="$PWD/local_data"
```

helper 只使用 Python 标准库，不需要网络、模型或 tokenizer。`--dataset-dir` 可指定另一份资产目录，默认是仓库的 `datasets/`。使用真实路径，不传字面 `$VAR` 占位符；上述 shell 展开的变量可以使用。目标目录必须不存在，拒绝符号链接及输入/输出目录重叠；仓库内仅允许写入忽略的 `local_data/`。输入不会被修改。若失败，检查残留输出并另选新目录，helper 不删除或续写残留文件。

## 文件与准备后路径

`.jsonl.gz` / `.csv.gz` 为 gzip 压缩的 JSONL / CSV；两份 `.parquet` 按原文件复制。

| 包内文件 | 输出路径（相对 DATA_ROOT） | 行数 / 决策数 |
|---|---|---|
| `s0/train.jsonl.gz` | `s0/canonical/train.jsonl` | 10,000 / 10,000 |
| `s0/dev.jsonl.gz` | `s0/canonical/dev.jsonl` | 1,000 / 1,000 |
| `s0/test.jsonl.gz` | `s0/canonical/test.jsonl` | 1,000 / 1,000 |
| `s1/train.jsonl.gz` | `s1/canonical/train.jsonl` | 40,000 / 40,000 |
| `s1/diagnostic_dev.jsonl.gz` | `s1/canonical/new_dev.jsonl` | 2,000 / 2,000 |
| `bench/ag_news_test.jsonl.gz` | `bench/ag_news_test.jsonl` | 7,600 / 7,600 |
| `bench/emotion_test.jsonl.gz` | `bench/emotion_test.jsonl` | 2,000 / 2,000 |
| `bench/sst5_test.jsonl.gz` | `bench/sst5_test.jsonl` | 2,210 / 2,210 |
| `bench/banking77_test.csv.gz` | `bench/banking77_test.csv` | 3,080 / 3,080 |
| `bench/typed_decisions_test.parquet` | `bench/typed_decisions_test.parquet` | 400 / 2,000 |
| `bench/prompt_injection_test.parquet` | `bench/prompt_injection_test.parquet` | 116 / 116 |

manifest 还指定将 S0 dev 按字节复制到 `s1/canonical/dev.jsonl`，将 S0 test 复制到 `bench/internal_s0_test.jsonl`。外部六套合计 17,006 决策，内部 1,000 决策单列。S1 train 是原 S0 train 加新增 delta，未单独打包 `new_train`，训练也不需要它。2,000 行 `new_dev` 仅供诊断，不能替换选优 dev。

11 份包内数据文件共 15,801,569 字节（约 15.07 MiB，通常记作 15.1 MiB），解压后约 112.5 MiB，另加两份 canonical 复制。继承的训练/评测 loader 不会自动解压这些 gzip 文件。

## 验证范围

helper 检查字节长度、完整读取 gzip（含 CRC）、逐行验证 JSONL、按 CSV 记录计数且不计表头。其 Parquet 处理不依赖 PyArrow，只复制并输出 manifest 声明的行数/决策数；所有格式的决策数打印值均来自 manifest，不是 helper 的语义标注审计。

本版本已另外用 PyArrow 验证 typed 的 400 行 / 2,000 决策及 Prompt Injection 的 116 行；JSONL/CSV 行数、外部 17,006 决策及内部 1,000 决策也已核实。结果记录在 [manifest 的 verification](manifest.json) 中。首次验证因 helper 放置层级错误触发仓库路径安全策略，在解压前被拒绝；按实际仓库 `scripts/` 布局修正后验证通过。这是验证目录布局问题，不是数据损坏。

这些验证不代表完整模型流水线复现，也不是穷尽的个人信息或法律审查。helper 不生成报告、校验和、时间戳或 runtime profile。

## Manifest 接口

helper 要求整数 `schema_version: 1` 及非空 `files` 列表；`copies` 可省略，默认空列表。版本标识、资产选择和预期计数来自 manifest，不是 helper 中的固定检查。本包的 `release_id` 为 `s0_s1_full_bench_v1`，包含表中 11 个映射。

每个文件含 `path`（相对 datasets）、`output_path`（相对输出根）、`compression`（`gzip` 或 `none`）、`format`（`jsonl`、`csv` 或 `parquet`），及非负整数 `rows`、`decisions`、`bytes`、`uncompressed_bytes`。输入扩展名须匹配格式，gzip 另加 `.gz`；输出扩展名须匹配未压缩格式。`copies` 中的 `source` 和 `output_path` 均相对输出根，每次复制必须引用文件输出或更早的复制结果。重复来源、重复或重叠目标均被拒绝，路径不能越界或使用符号链接。仓库内的 `local_data/` 限制是源码树安全策略，不是某次实验的数据选择。

`upstreams`、`source_cards` 和协议说明记录来源，不构成许可批准。若缺 manifest，请获取包含数据的私有仓库版本或使用 `--dataset-dir`；helper 不下载替代数据。

训练和直接全量评测见[原配方与 CLI](../docs/reproduce.md)。六套外部输入全部准备到 `bench/`；raw 上游仅用于重建，不是使用本包 canonical 训练/评测的前置条件。可选冻结 runner 仍需单独准备历史代码、stub、runtime 并手工适配 profile；fixtures 本身不是历史代码 bundle。再分发前请阅读[来源](SOURCES.md)及[许可未决项](LICENSES.md)。
