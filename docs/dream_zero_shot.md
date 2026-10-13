# Dream-7B：共享 Yes/No zero-shot Core14

2026-10-09 完成的研究评测。使用已发布的 Dream Instruct 权重，将同一个共享 Yes/No 决策方法迁移到 Dream：**不做项目微调，不训练额外头，不拟合温度，不生成答案文本**。

**本页只报告已完成的 zero-shot 结果，不补填后续微调数字。** 原生 Dream CLI、训练代码和后续结果现见 [Dream-S1](dream_s1.md)；该发布迁移仅做静态检查，未重新训练或运行额外评测。现有 quickstart / HTTP 默认行为保持不变，本页历史验证不代替新入口的发布验证。

## 1. 固定模型与执行协议

| 项目 | 实际协议 |
|---|---|
| 模型 | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| 固定 revision | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` |
| 参数规模 | 7,615,616,512，模型发布名为7B |
| 加载 | 原生 AutoModel / DreamModel / DreamTokenizer，固定本地文件、信任已审阅自定义代码 |
| Zero-shot含义 | 无本项目决策训练或校准；并非没有上游预训练或Instruct SFT |
| 精度 / batch / 温度 | BF16 / 1 / T=1；无量化或精度fallback |
| Attention | 原生DreamSdpaAttention，attention_mask="full"，is_causal=False |
| 计算 | 每个支持的决策一次原生完整词表forward；无生成、cache或迭代去噪 |
| 预测位置 | 原始unshifted logits读取mask_position−1，只对齐一次 |
| Renderer | 历史system/user决策模板，Dream自身chat template，add_generation_prompt=False；旧generic noul不注入criteria |
| 输入 | Adapter只接收state/qdef；推理读manifest/inputs，评分独立join targets/provenance；不是安全sandbox |
| Context | 4096 token上限，完整tokenize，不截断 |
| 数据范围 | Core14：Core17排除MultiNLI、Aegis2、PAWS，保留三套JevBench；5945 attempted |

Dream使用shifted prediction。对输入mask位置 $m\geq1$，官方采样器把原始logits向右移一位后读取mask位置，等价于直接读取原始输出的 $m-1$ 位置。它仍是**双向注意力**，不代表只看左侧文本。官方实现见[固定revision的generation_utils.py](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/05334cb9faaf763692dcf9d8737c642be2b2a6ae/generation_utils.py)。

choice/score沿用原生共享分数与候选softmax：

$$s_i=z_{i,\mathrm{Yes}}(m_i-1)-z_{i,\mathrm{No}}(m_i-1),\qquad p_i=\operatorname{softmax}(s)_i.$$

noul使用 $P(true)=\sigma(z_{\mathrm{Yes}}(m-1)-z_{\mathrm{No}}(m-1))$，概率顺序为 `[false, true]`。本checkpoint token IDs为Yes9454、No2753、mask151666，不照搬Qwen3的mask ID。

本次采用原生完整词表forward后取Yes/No，不使用Qwen3的候选两行hidden投影shortcut。语义仍是同一种读出，但计算成本不同；不做公平吞吐或架构速度比值。

## 2. Core14共同成功ID比较

base/S0/S1/Laya/Kev/Dream六模型及S1校准条件对齐为**5690个相同成功ID**，其中ContractNLI1819，其余13套完整。Dream加入后没有减少原五模型共同集。

各套指标先计算再等权平均，沿用历史评分器的四位小数suite指标。ECE为15个等宽置信度bin，NLL自然对数，Brier为类别误差平方和。这不是pooled ECE、官方JevBench ECE10/sealed分数或综合排名。

| Model / condition | ACC ↑ % | ECE15 ↓ % | NLL ↓ | Brier(sum) ↓ |
|---|---:|---:|---:|---:|
| Original DLLM 0.6B，zero-shot T1 | 55.40 | 26.97 | 1.3567 | 0.6566 |
| S0 0.6B，T1 | 61.08 | 12.53 | 0.9461 | 0.4886 |
| S1 0.6B，T1 | 63.97 | 14.92 | 0.9890 | 0.4851 |
| S1 0.6B，全局T=1.87397 | 63.97 | 10.33 | 0.8808 | 0.4637 |
| Laya-base，FP32 T1 aligned_unit | 55.64 | 17.01 | 1.2423 | 0.5923 |
| Kev-0.8B，FP32 checkpoint T=2.35110 | 60.31 | 14.30 | 0.9160 | 0.4921 |
| Dream-7B，BF16 zero-shot T1 | 64.08 | 17.74 | 0.9186 | 0.4817 |

0.6B身份/固定revision见[模型与训练说明](training.md)；Kev来源与原生头协议见[Kev比较](kev_comparison.md)；S1温度仅由六套旧benchmark拟合，见[校准结果](temperature_calibration.md)。既有模型预测保持不变，没有因为Dream结果重跑、选温度或改renderer。

### 与0.6B base / S1的关系

- 相较0.6B base，Dream平均ACC高约8.69个百分点，ECE/NLL/Brier也明显更好。逐套ACC9胜/2负/3平；NLL、Brier各13/14套更低。
- 相较未校准S1，Dream平均ACC仅高约0.11个百分点；6胜/6负/2平。NLL更低、Brier略低，但ECE更差。**没有显著性检验，不能称显著提升或全面胜过S1。**
- 校准后的S1平均ECE、NLL、Brier均优于本次未校准Dream。不是“各自最优校准后”的比较，Dream未拟合温度。

| Dream与对照的逐套描述性比较 | ACC胜/负/平 | ECE更低 | NLL更低 | Brier更低 |
|---|---|---:|---:|---:|
| S1 T1 | 6/6/2 | 5/14 | 8/14 | 6/14 |
| Kev-0.8B | 9/4/1 | 5/14 | 7/14 | 7/14 |
| Laya-base T1 | 9/4/1 | 8/14 | 9/14 | 11/14 |

## 3. Dream完整逐套结果与覆盖

以下按**Dream自身成功样本**计概率指标，不混用比较分母；ContractNLI为1853而非共同1819。

| Suite | 成功 / attempted | ACC ↑ % | ECE15 ↓ % | NLL ↓ | Brier(sum) ↓ |
|---|---:|---:|---:|---:|---:|
| JevBench original | 72/72 | 75.00 | 10.68 | 0.6233 | 0.3517 |
| JevBench easy | 48/48 | 100.00 | 3.92 | 0.0409 | 0.0050 |
| JevBench hard | 111/111 | 40.54 | 27.77 | 1.3381 | 0.7624 |
| Jabr v2 | 866/866 | 78.41 | 8.26 | 0.5665 | 0.3096 |
| VitaminC | 599/599 | 50.92 | 18.15 | 0.9833 | 0.5812 |
| MASSIVE en-US | 348/348 | 81.03 | 23.73 | 0.9352 | 0.3531 |
| MASSIVE de-DE | 348/348 | 72.13 | 28.75 | 1.3098 | 0.4907 |
| SQuAD2 | 299/299 | 85.28 | 16.46 | 0.4483 | 0.2800 |
| Civil Comments | 300/300 | 72.67 | 6.60 | 0.5669 | 0.3842 |
| HelpSteer2 | 249/249 | 35.34 | 4.75 | 1.3976 | 0.7143 |
| SummEval relevance | 237/237 | 16.88 | 28.29 | 1.6579 | 0.8292 |
| SummEval consistency | 144/144 | 84.03 | 31.78 | 0.9741 | 0.4717 |
| PubMedQA | 250/250 | 53.20 | 25.49 | 1.1155 | 0.6671 |
| ContractNLI | 1,853/2,074 | 51.11 | 14.06 | 0.9054 | 0.5464 |

| Dream覆盖 / 聚合 | 值 |
|---|---|
| Attempted / successful / unsupported_length | 5945 / 5724 / 221；超长全部在ContractNLI |
| Unsupported处理 | 原样保留，无截断、静默丢弃或source-mask拒绝 |
| 自身成功，14套等权精确均值 | ACC64.0379%，ECE15 17.7640%，NLL0.91876838，Brier0.48190208 |
| 自身成功pooled ACC | 3466/5724=60.5521% |
| 全attempted ACC，unsupported计错 | 3466/5945=58.3011% |
| 覆盖率 | 5724/5945，约96.28% |

### 优势与短板

Dream在Jabr、德语MASSIVE和SQuAD2的ACC分别为78.41%、72.13%、85.28%，高于S1的65.94%、59.77%、67.56%。但VitaminC50.92%、Civil Comments72.67%、PubMedQA53.20%，低于S1的67.95%、82.67%、57.60%。

校准方向并不一致：MASSIVE en平均置信度57.30% vs ACC81.03%，de43.38% vs ACC72.13%，SummEval consistency52.24% vs ACC84.03%，偏低置信度；PubMedQA平均置信度76.41% vs ACC53.20%，偏高置信度。不能概括成全部过度自信，也不能把高ACC等同于概率可靠。

## 4. Score诊断

只对有定义的六套score指标等权平均：JevBench original/hard、Jabr、HelpSteer2、SummEval relevance/consistency；它们不替代hard分类指标。

| Model / condition | MAE ↓ | RPS ↓ |
|---|---:|---:|
| S1 T1 | 0.807897 | 0.589401 |
| S1校准 | 0.847684 | 0.545732 |
| Laya-base | 0.884390 | 0.594050 |
| Kev-0.8B | 0.768401 | 0.468446 |
| Dream zero-shot T1 | 0.792387 | 0.524521 |

Nimble9单独等子集诊断：Dream ACC61.28%、ECE20.44%、NLL1.0432、Brier0.5302。不能代替Core14均值。

## 5. 验证与结论边界

历史研究运行完成以下检查：

- 10个真实输入（noul3、choice4、score3，含ContractNLI）与官方右移后的native full logits对比，最大概率差0；旧模板完整tokenize与手工插入mask IDs一致。
- 42/42 smoke推理/评分通过；全5945条推理、评分和对齐分析通过，没有运行失败。
- 全5724条raw Yes/No重建概率的最大差约4.44e-16；mask→预测位置−1元数据核对通过。
- 独立标准库逐套重算ACC/ECE/NLL/Brier，复现历史评分器，仅有四位小数舍入差异。

这支持共享Yes/No读出可以正确迁移到更大的Dream模型，并在无项目微调时取得与0.6B S1附近的总体ACC。**不支持“纯扩参导致提升”、每套全面改善或官方sealed排行结论。**

需要保留的限制：

1. Dream已接受上游预训练和Instruct SFT；完整来源暴露未审计。Core14是既定比较范围，原按Kev来源排除三套，不是为Dream认证的无污染测试集。公开题已被观察，不是全新blind测试。
2. 与0.6B基座的基础模型、diffusion转换训练、SFT和chat template不同；S1还有项目40k微调。不是仅参数规模或匹配训练预算对照。
3. 模型间精度、温度和头计算路径不同；NLL/Brier同时受预测能力与校准影响，不把全部差异归因于校准或架构。
4. Core的JevBench行使用ECE15及旧renderer，不与HTTP ECE10、较新noul模板或sealed结果混合。
5. 本次没有checkpoint选择、温度拟合、显著性检验或多seed结论，也没有根据Core结果再改prompt。
6. 本页历史发布仅为结果文档；后续原生Dream推理/训练入口见 [Dream-S1](dream_s1.md)，新迁移仅做静态检查，未重新执行训练或评测，现有模型和HTTP默认行为不变。后续微调是独立实验，不补填本页zero-shot结果。

聚合结果同列于[统一实验表](experiment_results.md)。Core成员与来源说明见[数据卡](../benchmark/DATA_CARD.md)和[来源条款](benchmark_sources.md)。原始预测、私有日志和服务器路径不随本页公开。
