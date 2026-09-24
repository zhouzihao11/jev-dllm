# 研究脚本索引

从仓库根目录执行；安装、训练和评测见[快速开始](../README.md)，单卡与恢复见[高级用法](../docs/reproduce.md)。

| 文件 | 用途 |
|---|---|
| `scripts/build_shared_yesno_data.py` | S0 grouped 数据构建 |
| `scripts/build_shared_yesno_s1.py`、`s1_data_config.json` | S1 数据扩充与外部文本排除 |
| `scripts/shared_yesno_sources.py` | S0 prepared train 转换 |
| `scripts/shared_yesno_synthetic.py`、`scripts/shared_yesno_s1_synthetic.py` | 有限规则/状态/概率任务 |
| `scripts/check_shared_yesno_labels.py` | S0 数据标签检查 |
| `scripts/train_qwen_masked_typed.py` | DDP 训练；配方指定 supervised/shared_yesno |
| `scripts/shared_yesno_supervised.py` | canonical loader、监督损失、内部评测 |
| `scripts/bench_diff_yesno.py` | 底层 DLLM 外部 batch 1/32 评测；通常由统一 runner 调用 |
| `scripts/bench_laya_aligned.py` | Laya 原生头、输入与温度对齐评测 |
| `scripts/run_full_benchmark.py` | 统一 DLLM/Laya 全量外部与内部评测；`--data-root` 使用当前源码，`--profile` 保留历史 bundle 路径；二者必选其一 |
| `scripts/bench_ar_matrix.py`、`scripts/bench_matrix.py`、`scripts/bench_local.py` | suite、metrics 和 SDK 支持模块 |

数据解压入口为根目录的 `scripts/prepare_datasets.py`。监督训练读取 canonical JSONL，内部测试使用 S0 test。数据重建接口见 [S1_DATA_BUILDER](S1_DATA_BUILDER.md) 与 [raw/prepared 参考](../docs/data_and_models.md)。
