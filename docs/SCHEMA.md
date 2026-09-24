# Canonical 数据契约

依据 `build_shared_yesno_data.validate_record`、`shared_yesno_supervised.load_structured_items` 与 S1 构建器的当前实现整理；不是新的 schema 实现。推荐 UTF-8 JSONL，一行一个对象、一个 question，不含空行。保留 JSON 对象插入顺序，尤其不能重排 choice 的 criteria。

| 字段 | 类型/约束 |
|---|---|
| `schema_version` | 固定 `shared_yesno_data_v1` |
| `case_id`, `group_id`, `view_id` | 非空字符串；`(case_id, view_id)` 在文件内唯一；相关来源组不能跨 split |
| `split` | 单文件一致的 `train` / `dev` / `validation` / `test`；构建器输出前三种研究分区中的 train/dev/test；trainer 禁止 test |
| `source` | 对象：`name`, `record_id`, `original_split`；公共来源 original_split=train，合成来源 generated；revision 可附加 |
| `family`, `language` | 构建器要求非空 family、language=en |
| `state` | 字符串或对象，模型可见状态 |
| `questions` | 恰好一个 qid 对应对象，含 `type`, `instructions`, 可选/按类型要求的 `criteria` |
| `gold` | 恰好相同 qid，含 `kind` 与完整 `probabilities` 对象 |
| `option_ids` | 相同 qid 对应非空唯一字符串列表；与显示候选位置一一对应，不输入模型 |
| `metadata` | 构建器要求对象及非空 `sampling_stratum`；solver facts、源标签、精确分数等都不是模型输入 |

## 三类原语

- `choice`：criteria 为至少两项的有序字典，key 是自然显示标签，value 是描述字符串或 null；概率 key 必须恰好覆盖这些显示标签。候选顺序取 criteria 顺序，不取 gold 的任意排列。
- `score`：criteria 为至少两个字符串的有序等级列表；概率 key 为连续字符串 `"0"` 到 `"K-1"`，构建器要求升序。等级顺序不可打乱。
- `noul`：输出固定 `[false, true]`，概率 key 为 `"false"`, `"true"`；criteria 可省略或提供对应描述，但 DLLM 提示中不使用隐藏判断 rubric；只有一个 mask。

`kind=hard` 是精确 one-hot；`kind=known_distribution` 是已知机制分布，不是教师置信度。概率必须为有限非负数、总和为 1；构建器容差 1e-8，训练 loader 容差 1e-6，不会补齐缺失目标或静默归一化。JSON boolean 不能当概率数值。

## 输入与训练

只有 state 和 question 的 type/instructions/criteria 进入渲染器。禁止拼入 gold、option_ids、source、solver metadata。choice/score 的 K 个 mask 对应 `softmax(zYes-zNo)`；noul 的一个 mask 对应 `[P(No), P(Yes)]`。这不是 causal SFT，不对目标做 token shift。超长输入失败而非静默截断；特殊 mask/chat token 字面量在构建时拒收。

S0 的 `canonical/*.jsonl` 是监督入口；`rendered/` token 缓存和 `compatibility/*.parquet` 不是当前监督训练命令的输入。S1 的 `canonical/dev.jsonl` 保持原 S0 dev 字节不变，`new_dev.jsonl` 仅用于诊断，test 仍从 S0 读取。本文不提供未经运行验证的合成数据文件作为测试成功证据。
