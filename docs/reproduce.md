# 复现入口：源码首版，不是一键环境

本文命令根据当前 CLI 静态检查整理，本次没有执行安装、模型加载、smoke、训练或评测。必须先完成[资源契约](data_and_models.md)并核实[第三方条款](../THIRD_PARTY.md)。缺少历史冻结 bundle、custom code/stub 和部分源版本会阻塞精确复现；不要把下列命令的存在当作干净机器验证。

## 1. 环境与源码导入

原研究记录为 Python 3.10、torch 2.3.1+cu121、Transformers 4.57.0、tokenizers 0.22.2、datasets 4.8.5、pandas 2.3.3、pyarrow 24.0.0。后三项来自原 full-test 环境记录；不代表本发布副本已在这些版本重新测试。NumPy、safetensors、huggingface_hub 的精确版本未记录，依赖文件只保留已知约束，不是完整 lockfile。原包的 Python >=3.10 声明不保证任意新版本兼容。

在执行机器上自行准备 Python 3.10 与合适驱动。若选择历史 CUDA 12.1 wheel，安装示意如下；其他硬件需独立验证，不要混用 CPU wheel 并称为同环境。

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r research/requirements-research.txt
python -m pip install --no-deps -e .
```

保留原包名 `laya`，不发布新 PyPI 包；已有同名安装可能冲突，建议使用独立执行环境。研究脚本从源码根运行，所需相对导入目录如下。先设置 `MODEL_DIR`、`DATA_ROOT`、`OUTPUT_ROOT`、`DLLM_STUB_ROOT` 为合法资源的绝对路径，均位于仓库外。DLLM_STUB_ROOT 应为含 `dllm/` 的目录，不是 `dllm/` 本身；此资源不随源码提供。

```bash
export PYTHONPATH="$PWD:$PWD/research/scripts:$DLLM_STUB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

所有模型加载信任外部 custom code，须先审阅代码。原 S1 训练是双 GPU NCCL DDP；CPU 不支持该 trainer。不要为了绕过缺资源修改训练循环、损失或模型代码。

## 2. S0 / S1 数据构建

MODEL_DIR 此时是原 DLLM tokenizer 所在模型目录。S0 使用实际记录的 8,000 原记录采样上限和 CLINC 分层，而非继承的 24,000 默认值。输入来源和导出顺序仍需一致。

```bash
python research/scripts/build_shared_yesno_data.py \
  --source-dir "$DATA_ROOT/s0_sources" --tokenizer-path "$MODEL_DIR" \
  --output-dir "$DATA_ROOT/s0" --seed 42 --max-length 4096 \
  --max-records-per-source 8000 --stratify-source clinc

python research/scripts/check_shared_yesno_labels.py \
  --dataset-dir "$DATA_ROOT/s0" --source-dir "$DATA_ROOT/s0_sources" \
  --output "$OUTPUT_ROOT/s0_label_check.json"

python research/scripts/build_shared_yesno_s1.py \
  --s0-dir "$DATA_ROOT/s0" --source-dir "$DATA_ROOT/s0_sources" \
  --new-source-dir "$DATA_ROOT/s1_sources" --tokenizer-path "$MODEL_DIR" \
  --heldout-profile "$DATA_ROOT/full_eval/profile.json" \
  --output-dir "$DATA_ROOT/s1" --config research/s1_data_config.json
```

两个 builder 都有 `--smoke`，应使用单独输出目录；smoke 不满足正式配额，S1 smoke 仍需要完整 S0 和外部 fixtures。OUTPUT_ROOT 及独立报告文件的父目录须预先存在。S1 输出目录必须全新；S0 拒绝非空目录。失败后保留报告，另选新目录重试，不覆盖原研究输入。

S0 预期 train/dev/test 为 10,000/1,000/1,000。S1 config 请求新增 train 30,000（公共 25,000 + 合成 5,000）、诊断 dev 2,000；原记录保留 S0 train 10,000，形成 40,000。实际保留数取决于碰撞/短缺报告。原 dev 不改，不代表没有碰撞；出现 `requires_parent_dev_collision_review` 必须人工审查，不能静默改 dev。标签检查器是 S0 专用，S1 使用构建器内的原标注映射和独立 solver 检查。

## 3. 监督训练

MODEL_DIR 必须是原 MDLM base，不是 S0 checkpoint。显式指定两个模式和原 dev；默认 `rlcd`/`identifier` 是不同实验。

```bash
torchrun --standalone --nproc_per_node=2 research/scripts/train_qwen_masked_typed.py \
  --model-path "$MODEL_DIR" --train-data "$DATA_ROOT/s1/canonical/train.jsonl" \
  --dev-data "$DATA_ROOT/s1/canonical/dev.jsonl" --output-dir "$OUTPUT_ROOT/s1_train" \
  --loss-mode supervised --scoring-mode shared_yesno \
  --seed 42 --epochs 3 --micro-batch 4 --grad-accum 4 \
  --learning-rate 2.5e-5 --min-lr 1e-6 --weight-decay 0.01 \
  --warmup-ratio 0.05 --rps-weight 0.25 --eval-every 100 --save-every 200 \
  --save-epoch-checkpoints --save-predictions --max-length 4096 --amp-dtype bfloat16
```

有效 batch=2×4×4=32，但继承实现逐个样本 forward，不是四样本张量 forward。40k/32×3 对应 3,750 更新。依据原 dev mean KL 选 `best/`；step 1,300 是原观察结果，不保证重新训练仍在同一步选优。epoch 1 末尾是 step 1,250。训练输出和预测可能含敏感来源内容，留在仓库外。

## 4. 全量质量 runner：需要独立供应 bundle

先在仓库外完成 `full_eval_profile.template.json` 的占位替换、fixtures、`code/` 和 `support/dllm_stub/`。模板不能直接运行。MODEL_DIR 改为待测 checkpoint。历史质量必须使用原冻结版本 bundle；当前发布代码不是该快照，新评测需不同 profile_id 和真实 code.commit。

```bash
python research/scripts/run_full_benchmark.py \
  --profile "$DATA_ROOT/full_eval/profile.json" --backend dllm \
  --model-path "$MODEL_DIR" --output-dir "$OUTPUT_ROOT/quality_smoke" \
  --device cuda:0 --dtype bfloat16 --smoke
```

正式全量运行去掉 `--smoke` 并换全新 output-dir。runner 拒绝覆盖已有目录；它调用 profile 的绝对 Python 和相对 code.path，不是调用当前 shell 的最新代码。smoke 每 suite 三个决策只检查执行，不能作质量比较。runner **没有 `--batch-size`**；默认 DLLM batch 1，dtype 仅作用于 DLLM。

Laya 使用相同 profile，改 `--backend laya`，MODEL_DIR 指向合法的 Laya 英文 checkpoint，并换输出目录；保留 FP32，默认最多 16 sequences / 8,192 tokens，报告 aligned_unit 主条件及三个预声明参考条件。不得据此与 DLLM BF16 batch 32 作同配置速度比较。

## 5. 当前批处理入口：独立于历史 runner

以下 fixtures 使用模板的相对布局，均由使用者准备。显式把全部 limit 设为 0，避免 400/600 前缀默认值。

```bash
python research/scripts/bench_diff_yesno.py \
  --model-path "$MODEL_DIR" \
  --typed-data "$DATA_ROOT/full_eval/data/typed_decisions_test.parquet" \
  --prompt-data "$DATA_ROOT/full_eval/data/prompt_injection_test.parquet" \
  --sst5-data "$DATA_ROOT/full_eval/data/sst5_test.jsonl" \
  --banking-test-csv "$DATA_ROOT/full_eval/data/banking77_test.csv" \
  --ag-news-test-jsonl "$DATA_ROOT/full_eval/data/ag_news_test.jsonl" \
  --emotion-test-jsonl "$DATA_ROOT/full_eval/data/emotion_test.jsonl" \
  --output "$OUTPUT_ROOT/b32_summary.json" --predictions "$OUTPUT_ROOT/b32_predictions.jsonl" \
  --device cuda:0 --dtype bfloat16 --max-length 4096 \
  --batch-size 32 --warmup-batches 3 --limit-per-suite 0 \
  --banking-test-limit 0 --ag-news-test-limit 0 --emotion-test-limit 0 --sst5-test-limit 0
```

batch 1 对照只改 batch-size=1 并使用两个新输出文件，仍用同代码、同设备、同精度和同样预热。远端 smoke 可用 `--limit-per-suite 3`，但仍需完整源文件；全量计数应为 17,006，零丢样本。内部测试另用：

```bash
python research/scripts/shared_yesno_supervised.py \
  --model-path "$MODEL_DIR" --data "$DATA_ROOT/s0/canonical/test.jsonl" \
  --output "$OUTPUT_ROOT/internal_summary.json" --predictions "$OUTPUT_ROOT/internal_predictions.jsonl" \
  --device cuda:0 --amp-dtype bfloat16 --max-length 4096 --limit 0
```

batch 32 使用原序右 padding 和四维布尔 valid-key mask，不改模型。BF16 并非逐位等价；已有 72 个 argmax 翻转，FP32 排查只有 384 条，不可推广为全量保证。性能只计同步 forward，不包括分词、传输、CPU softmax、I/O 或排队。

## 6. 本次验证范围

只做源码检查、AST/配置语法、导入闭包、相对文档链接及敏感模式检查，不执行被检查模块。未运行任何本地或远端测试，也未安装依赖。原研究结果与新源码副本的独立复现状态严格区分；外部许可、完整来源导出、冻结 bundle 和模型依赖仍待补齐。
