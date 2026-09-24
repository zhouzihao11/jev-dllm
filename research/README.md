# 研究入口与继承模块

从仓库根目录执行；安装、训练和全量评测见[快速开始](../README.md)，单卡、恢复与内部评测见[高级用法](../docs/reproduce.md)。这些脚本不提供安装后的 console commands。

| 文件 | 用途 |
|---|---|
| `scripts/build_shared_yesno_data.py` | S0 grouped 数据构建 |
| `scripts/build_shared_yesno_s1.py`、`s1_data_config.json` | S1 扩充及外部文本排除审计 |
| `scripts/shared_yesno_sources.py` | S0 prepared train 转换 |
| `scripts/shared_yesno_synthetic.py`、`scripts/shared_yesno_s1_synthetic.py` | 有限规则/状态/概率任务 |
| `scripts/check_shared_yesno_labels.py` | S0 构建后的独立标签检查；不支持作为全 S1 标签检查器 |
| `scripts/train_qwen_masked_typed.py` | 继承 DDP trainer；明确指定 supervised/shared_yesno |
| `scripts/shared_yesno_supervised.py` | canonical loader、监督损失、内部评测 |
| `scripts/bench_diff_yesno.py` | 底层 DLLM 外部 batch 1/32 评测；通常由统一 runner 调用 |
| `scripts/bench_laya_aligned.py` | Laya 原生头、输入与温度对齐评测 |
| `scripts/run_full_benchmark.py` | 统一 DLLM/Laya 全量外部与内部评测；`--data-root` 使用当前源码，`--profile` 保留历史 bundle 路径；二者必选其一 |
| `scripts/bench_ar_matrix.py`、`scripts/bench_matrix.py`、`scripts/bench_local.py` | 必需的 suite/metrics/SDK 依赖闭包；各自 CLI 是 legacy optional，不是本研究主配方 |

数据解压入口为根目录的 `scripts/prepare_datasets.py`，不在 `research/scripts/` 下。当前监督入口读取 canonical JSONL；S1 不导出 test，内部测试使用 S0 test。

legacy 模块保留实际导入关系；Jev 引用分数不是本项目新测量，trainer 的 RLCD/identifier 模式也不是 Shared Yes/No 配方。`bench_local.py` 的模型目录可用 `LAYA_MODEL_ROOT` 或 `--model-root` 指定。

数据重建接口见 [S1_DATA_BUILDER](S1_DATA_BUILDER.md) 与 [raw/prepared 参考](../docs/data_and_models.md)。`requirements-research.txt` 兼容旧安装命令，依赖统一维护在根 `requirements.txt`。
