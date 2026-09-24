# 从来源重建数据：raw/prepared 参考

日常训练与评测见[根 README](../README.md)；本文供需要调整数据构建的人参考。完整上游 train 的获取与 raw-to-prepared 导出需自行准备。

重建时将 `$DATA_ROOT` 设为独立的新目录，沿用 README 中的 `$BASE_MODEL`。保留原始 train 文件、revision、标签元数据、顺序及 ID，导出后逐行对照字段和标签映射。

## S0 prepared contract

`$DATA_ROOT/s0_sources/source_manifest.json` 应含 `sources` 对象，四个条目 `clinc/snli/arc/sgd` 各有非空 `revision` 和 `license` 或 `license_declaration` 字段。记录 URL、config、train split 和稳定 ID 生成规则。

| 文件（相对 s0_sources） | 必需内容 |
|---|---|
| `prepared/clinc_train.jsonl` | `record_id`, `text`, `intent`, `domain`；全部官方 train，原 intent/domain 映射；由适配器排除 banking/credit_cards/oos |
| `prepared/snli_train.jsonl` | `record_id` 使用原 pairID，`premise`, `hypothesis`, 字符串 `label` 为 entailment/contradiction/neutral，`group_key=captionID.split('#',1)[0]`；排除无有效标签行；S1 要求图像组，不接受缺失组 |
| `prepared/arc_train.jsonl` | `record_id`, `config`, `question`, `choices=[{id,text},...]`, `answerKey`；原 Easy/Challenge train，不猜答案；ID/config 保持一致 |
| `prepared/sgd_train.jsonl` | 原对话对象含 `dialogue_id`, `services`, `turns`；turn 保留 speaker/utterance/frames，USER frame 保留 service/state.active_intent |
| `prepared/sgd_schema.json` | 原服务列表，含 service_name/description/intents，每个 intent 含 name/description |

原研究版本：CLINC `828f8093932c8fe6ca7936c3d2e52903b1c523de`；SNLI 1.0 原始 release；ARC `210d026faf9955653af8916fad021475a3f00453`；SGD `e852981ae34990f4358979625854259302feaa78`。

同一文件内稳定 ID 不应重复。可附 `split`/`original_split`，存在时必须为 train。没有 ID 的来源需固定源文件版本和行顺序后使用零基行号。

## S1 raw 到 prepared contract

使用官方原始 train，不使用六类外部来源任意 split、镜像或衍生数据。不得把 GoEmotions 转成 DAIR Emotion 的六类，也不得把 DBpedia 改成 AG News 四类。

| 来源 | 原研究 revision | 从 raw 转换的规则 |
|---|---|---|
| `fancyzhx/dbpedia_14` | `9abd46cf7fc8b4c64290f26993c540b92aa145ac` | Parquet 的 title/content/label；用该 revision 的 ClassLabel names 按整数下标映射为字符串标签；稳定 record_id 可为固定 train 文件零基行号 |
| `google-research-datasets/go_emotions`，simplified | `add492243ff905527e67aeb8b80c082af02207c3` | 原 comment id -> 字符串 record_id，text，labels 中每个整数按 ClassLabel names 映射；不丢失多标签，不取第一个标签 |
| `google/boolq` | `35b264d03638db9f4ce671b711558bf7ff0f80d5` | passage/question/answer，answer 必须为 JSON boolean，保留可用 title；无原 ID 时固定 train 文件零基行号 |

输出到 `$DATA_ROOT/s1_sources/prepared/`：

- `dbpedia_train.jsonl`：`record_id/title/content/label` 均为字符串；`dbpedia_labels.json` 是原 14 类有序名称列表。
- `goemotions_train.jsonl`：字符串 `record_id/text`，`labels` 为字符串列表；`goemotions_labels.json` 是原 27 情绪加 neutral 的有序名称列表。构建器只选单标签，原导出仍保留多标签。
- `boolq_train.jsonl`：字符串 `record_id/passage/question`、布尔 `answer`、可选字符串 `title`。禁止将字符串 `"false"` 用 Python truthiness 转成 True。
- `$DATA_ROOT/s1_sources/source_manifest.json`：`sources.dbpedia/goemotions/boolq` 各含非空 revision、upstream、license 或 license_declaration 字段。

S1 还需要完整的 S0 canonical train/dev/test、source_manifest、build_manifest；S0 的构建 seed 必须保留，S1 synthetic_seed 与之不同。复用的 SNLI/CLINC/SGD prepared revision 必须与 S0 相同。详细规则见 [S1_DATA_BUILDER](../research/S1_DATA_BUILDER.md)。

## 外部评测 fixtures 与 profile

仓库提供下表全部 fixtures，解压到 `bench/`（内部测试从 S0 test 复制）。保持全量、原始行顺序和重复行，不用 train 补数量。捕获版本见 [manifest](../datasets/manifest.json) 和[来源记录](../datasets/SOURCES.md)。

| fixture | 字段/标签契约 | 参考行数 / 决策数 |
|---|---|---|
| `typed_decisions_test.parquet` | workflow、state、questions、gold；questions/gold 为 JSON 字符串，choice label 匹配 criteria key，noul label 为 false/true，score label 为等级索引；保留原 probabilities/score | 400 / 2,000 |
| `ag_news_test.jsonl` | text 字符串，label 整数 0..3：world/sports/business/sci_tech | 7,600 / 7,600 |
| `emotion_test.jsonl` | text，label 0..5：sadness/joy/love/anger/fear/surprise | 2,000 / 2,000 |
| `banking77_test.csv` | text/category，原 77 类；候选按 category 排序后下划线替换为空格 | 3,080 / 3,080 |
| `prompt_injection_test.parquet` | text、label：0 非注入，1 注入；不能混入 546 行 train | 116 / 116 |
| `sst5_test.jsonl` | text、label 0..4：very negative/negative/neutral/positive/very positive | 2,210 / 2,210 |
| `internal_s0_test.jsonl` | S0 canonical test，不是 S1 new_dev | 1,000 / 1,000 |

S1 构建器读取六套外部 fixtures 的完整文本做排除检查，声明行数须与文件一致；外部来源各 split 均不用于监督训练。

S1 builder 的 `--heldout-profile` 使用 [profile 模板](full_eval_profile.template.json)中的 sources 路径读取这些输入；先按实际目录适配，JSON 不展开 shell 变量。可选模型 runner 的 code/runtime 配置见[高级用法](reproduce.md)。

## 构建命令

准备好上述来源和 profile 后，运行：

```bash
python research/scripts/build_shared_yesno_data.py \
  --source-dir "$DATA_ROOT/s0_sources" --tokenizer-path "$BASE_MODEL" \
  --output-dir "$DATA_ROOT/s0" --seed 42 --max-length 4096 \
  --max-records-per-source 8000 --stratify-source clinc

python research/scripts/check_shared_yesno_labels.py \
  --dataset-dir "$DATA_ROOT/s0" --source-dir "$DATA_ROOT/s0_sources" \
  --output "$DATA_ROOT/s0_label_check.json"

python research/scripts/build_shared_yesno_s1.py \
  --s0-dir "$DATA_ROOT/s0" --source-dir "$DATA_ROOT/s0_sources" \
  --new-source-dir "$DATA_ROOT/s1_sources" --tokenizer-path "$BASE_MODEL" \
  --heldout-profile "$DATA_ROOT/full_eval/profile.json" \
  --output-dir "$DATA_ROOT/s1" --config research/s1_data_config.json
```

两个 builder 都有 `--smoke`，需单独输出目录；S1 smoke 仍需要完整 S0 和外部 fixtures。S0 拒绝非空输出目录，S1 要求全新目录，报告父目录须存在。构建只加载 tokenizer。

S0 目标 train/dev/test 为 10,000/1,000/1,000；S1 请求新增 train 30,000（公共 25,000 + 合成 5,000）与诊断 dev 2,000。实际数量取决于碰撞/短缺报告；`requires_parent_dev_collision_review` 需要审查，不能静默修改原 dev。S0 标签检查器不适用于完整 S1；S1 使用构建器中的标注映射与独立 solver。
