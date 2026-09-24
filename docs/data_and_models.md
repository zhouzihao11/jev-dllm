# 数据与模型准备

**本首版提供 prepared 输入契约，不提供 raw 到 prepared 的完整获取/导出工具。** 旧下载器有原机器授权目录限制，因此没有复制；不绕过其限制，也不宣称已有一键下载。数据、模型另行发布的计划尚未交付任何文件，许可未确认的资产不捆绑。

所有 `$DATA_ROOT`、`$MODEL_DIR`、`$OUTPUT_ROOT` 由使用者设为仓库外合法路径。保留原始 train 文件、revision、标签元数据、原始顺序及 ID，导出后逐行对照原字段和标签映射。只有取得合法访问权后才准备资源。

## S0 prepared contract

`$DATA_ROOT/s0_sources/source_manifest.json` 应含 `sources` 对象，四个条目 `clinc/snli/arc/sgd` 各有非空 `revision` 和 `license` 或 `license_declaration`。建议同时记录官方 URL、config、train split 和稳定 ID 生成规则；这些声明不会被构建器当作独立法律批准。

| 文件（相对 s0_sources） | 必需内容 |
|---|---|
| `prepared/clinc_train.jsonl` | `record_id`, `text`, `intent`, `domain`；全部官方 train，原 intent/domain 映射；由适配器排除 banking/credit_cards/oos |
| `prepared/snli_train.jsonl` | `record_id` 使用原 pairID，`premise`, `hypothesis`, 字符串 `label` 为 entailment/contradiction/neutral，`group_key=captionID.split('#',1)[0]`；排除无有效标签行；S1 要求图像组，不接受缺失组 |
| `prepared/arc_train.jsonl` | `record_id`, `config`, `question`, `choices=[{id,text},...]`, `answerKey`；原 Easy/Challenge train，不猜答案；ID/config 保持一致 |
| `prepared/sgd_train.jsonl` | 原对话对象含 `dialogue_id`, `services`, `turns`；turn 保留 speaker/utterance/frames，USER frame 保留 service/state.active_intent |
| `prepared/sgd_schema.json` | 原服务列表，含 service_name/description/intents，每个 intent 含 name/description |

原研究版本记录：CLINC `828f8093932c8fe6ca7936c3d2e52903b1c523de`；SNLI 1.0 原始 release；ARC `210d026faf9955653af8916fad021475a3f00453`；SGD `e852981ae34990f4358979625854259302feaa78`。来源链接和许可见 [THIRD_PARTY](../THIRD_PARTY.md)。这些记录不是本次重新下载验证。

同一文件内稳定 ID 不应重复。可附 `split`/`original_split`，存在时必须为 train。没有 ID 的来源必须固定源文件版本和行顺序后使用零基行号，不能导出时随机生成。缺少 split 字段不会证明来源确为 train。

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
- `$DATA_ROOT/s1_sources/source_manifest.json`：`sources.dbpedia/goemotions/boolq` 各含非空 revision、upstream、license 或 license_declaration；不得用占位值冒充真实来源记录。

S1 还需要完整的 S0 canonical train/dev/test、source_manifest、build_manifest；S0 的构建 seed 必须保留，S1 synthetic_seed 与之不同。复用的 SNLI/CLINC/SGD prepared revision 必须与 S0 相同。详细规则见 [S1_DATA_BUILDER](../research/S1_DATA_BUILDER.md)。

## 外部评测 fixtures 与 profile

保持官方 test 全量、原始行顺序和重复行；不能用 train 补数量。各来源版本尚未全部对外锁定，这也是精确历史复现缺口。

| fixture | 字段/标签契约 | 参考行数 / 决策数 |
|---|---|---|
| `typed_decisions_test.parquet` | workflow、state、questions、gold；questions/gold 为 JSON 字符串，choice label 匹配 criteria key，noul label 为 false/true，score label 为等级索引；保留原 probabilities/score | 400 / 2,000 |
| `ag_news_test.jsonl` | text 字符串，label 整数 0..3：world/sports/business/sci_tech | 7,600 / 7,600 |
| `emotion_test.jsonl` | text，label 0..5：sadness/joy/love/anger/fear/surprise | 2,000 / 2,000 |
| `banking77_test.csv` | text/category，原 77 类；候选按 category 排序后下划线替换为空格 | 3,080 / 3,080 |
| `prompt_injection_test.parquet` | text、label：0 非注入，1 注入；不能混入 546 行 train | 116 / 116 |
| `sst5_test.jsonl` | text、label 0..4：very negative/negative/neutral/positive/very positive | 2,210 / 2,210 |
| `internal_s0_test.jsonl` | S0 canonical test，不是 S1 new_dev | 1,000 / 1,000 |

外部合计 17,006，内部另计。S1 exclusion audit 读取六来源完整文本而非评测前缀，声明行数必须与文件一致。来源策略是禁止六来源所有 split；文本审计只能覆盖实际供应的 fixtures，不证明预训练零污染。

`full_eval_profile.template.json` **不是 ready profile**：它只有字段结构、相对路径和协议参考数量，没有 fixtures/code/stub。JSON 不会展开 `$VAR` 或 `~`。在仓库外建立 bundle 并编辑本地 profile：`runtime.python`、`runtime.home`、`runtime.hf_home` 替换成执行机真实绝对路径；code.path、runtime.dllm_stub 和 sources.*.path 都是相对 profile 目录的路径，不得绝对或包含 `..`。

历史冻结质量需资源所有者单独提供 `fb672477594f29681b90d13ec679819ab8033c5c` 对应的合法源码 bundle 与原 fixtures；本仓库不含该快照。若改用本发布版源码，必须填写新的实际版本与不同 profile_id，称为新评测而非历史冻结复现。`code.commit` 只是记录字段，runner 不验证它与目录内容相符，使用者须如实填写。不要仅改字段冒充冻结版本。

## 模型准备与阻塞项

DLLM 路径应含兼容的本地 config/tokenizer/权重和审阅过的模型 custom code，模型须暴露 `base.model`、`lm_head`，tokenizer 的 Yes/No 必须单 token、互不相同，mask ID 必须可 round-trip。原记录 Yes=9454、No=2753、mask=151669；代码运行时检查而非任意模型通用保证。S0/S1 数据构建只加载 tokenizer，不加载权重，但仍需要相应 Python 依赖。

外部 `dllm_stub` 按历史布局位于 bundle 的 `support/dllm_stub`，其下应有 `dllm/__init__.py`。stub 的具体接口、来源、版本和许可没有足够可发布材料，不能凭名字编造替代实现；由合法资源所有者供应并核实。未得到匹配 custom code/stub 前，完整干净机器复现仍被阻塞。

runner 将 HOME/HF_HOME 设置为 profile 值，并移除 PYTHONUSERBASE/PYTHONNOUSERSITE；历史运行曾依赖 HOME 下的用户级包。选择明确的绝对 Python 路径，检查其实际依赖，不假定原机器缓存存在。runner 强制离线；直接命令也应使用明确 fixture 与本地模型。所有生成报告可能含文本、token IDs 或私有路径，不随源码分发。
