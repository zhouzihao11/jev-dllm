# 研究入口与继承模块

从仓库根目录执行；先读 [复现说明](../docs/reproduce.md)。这些脚本不是安装后自动提供的 console commands。

| 文件 | 用途 |
|---|---|
| `scripts/build_shared_yesno_data.py` | S0 grouped 数据构建 |
| `scripts/build_shared_yesno_s1.py`、`s1_data_config.json` | S1 扩充及外部文本排除审计 |
| `scripts/shared_yesno_sources.py` | S0 prepared train 转换 |
| `scripts/shared_yesno_synthetic.py`、`scripts/shared_yesno_s1_synthetic.py` | 有限规则/状态/概率任务 |
| `scripts/check_shared_yesno_labels.py` | S0 构建后的独立标签检查；不支持作为全 S1 标签检查器 |
| `scripts/train_qwen_masked_typed.py` | 继承 DDP trainer；明确指定 supervised/shared_yesno |
| `scripts/shared_yesno_supervised.py` | canonical loader、监督损失、内部评测 |
| `scripts/bench_diff_yesno.py` | 当前直接 DLLM batch 1/32 评测 |
| `scripts/bench_laya_aligned.py` | Laya 原生头、输入与温度对齐评测 |
| `scripts/run_full_benchmark.py` | 执行用户提供的 bundle/profile；不内置历史快照 |
| `scripts/bench_ar_matrix.py`、`scripts/bench_matrix.py`、`scripts/bench_local.py` | 必需的 suite/metrics/SDK 依赖闭包；各自 CLI 是 legacy optional，不是本研究主配方 |

保留 legacy 代码是为了维持实际导入关系和原语义；其中 Jev 引用分数不是本项目重新测量的结果。旧 trainer 的 RLCD/identifier 模式也保留，但不代表本文训练条件。没有复制旧下载器或依赖原机器缓存的 bundle 构建工具；prepared 来源由使用者按契约单独准备。

`bench_local.py` 原本已使用用户 HOME 下的 `~/laya_models`，没有固定用户名。本副本增加 `LAYA_MODEL_ROOT` 环境覆盖，仍可用已有 `--model-root` CLI。其他模型/数据/输出路径均通过现有参数提供。

S0 生成 data card 中的旧 Parquet trainer 限制是历史说明；当前监督入口读取 canonical JSONL，不读取 compatibility Parquet。S1 不导出 test，内部测试始终来自 S0。
