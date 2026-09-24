# Shared Yes/No：让 DLLM 直接评估选项

2026-09-24 · 技术博客

代码、训练/测试数据与使用说明：[zhouzihao11/jev-dllm](https://github.com/zhouzihao11/jev-dllm)。

把语言模型接入软件流程，我们经常需要它完成一个明确的判断：为请求选择工具、识别用户意图、判断条件是否成立，或者给事件评定等级。程序提供状态和候选，模型返回概率，后续逻辑据此完成排序、分支选择和人工转交。

这类任务有一个共同点：**候选范围已经给定，模型需要理解并比较这些候选。** 顺着这个需求，我们提出 Shared Yes/No：在每个选项旁设置一个决策 token，用预训练模型原有的 Yes/No 输出权重并行评分，再将分数组成结构化概率。

我们用一个约 0.6B 的 masked diffusion language model（MDLM）实现这一接口。它把语言模型的语义理解能力，接到了一个简单的输出契约上：**候选由自然语言描述，判断由共享权重完成，结果以概率交给程序。**

## 1. 为什么用 DLLM？

如果把语言模型用于决策，最常见的做法仍然沿用自回归模型的 next-token 接口：把状态、问题和候选都放进 prompt，在序列末尾预测一个答案 token。

```text
State: ...
Question: ...
A. option 1
B. option 2
C. option 3

Answer:
```

此时所有候选最终都汇聚到同一个答案位置 $h_{\mathrm{ans}}$，模型再用词表输出层比较：

$$
\ell_A=W_A^\top h_{\mathrm{ans}}+b_A,\qquad
\ell_B=W_B^\top h_{\mathrm{ans}}+b_B,\qquad
\ell_C=W_C^\top h_{\mathrm{ans}}+b_C.
$$

这个接口非常适合生成，但用于动态决策时会引入两个额外步骤。

第一，**候选本身没有独立的读出位置**。模型需要先在一个统一的答案 hidden state 中完成整组选项的比较，再把结果投影到 `A/B/C` 等答案 token。候选语义、候选间比较和最终类别读出都集中在同一个位置完成。候选数量和候选结构变化时，这个末端状态始终承担整道题的全部决策信息。

第二，**多个决定需要被序列化**。如果同一个 state 下同时存在多个输出变量，例如事件类型、风险等级、下一步动作和是否升级，next-token 生成会自然形成：

$$
p(y_1,\ldots,y_m\mid x)
=\prod_{i=1}^{m}p(y_i\mid x,y_{<i}).
$$

这会给原本并列的决策变量引入一个生成顺序：后面的决定依赖前面已经生成的结果。另一种做法是把每个问题拆开分别询问，但这样同一个 state 会被重复编码。对于“多个预设槽位同时判断”的任务，我们更希望这些输出从一开始就作为并列的未知变量存在。

Masked Diffusion Language Model 提供了一个更直接的表示。它的输入中可以同时存在多个 mask：

$$
[M_1], [M_2], \ldots, [M_m].
$$

每个 mask 都有自己的 contextual representation，并通过双向注意力读取完整的 state、question 和候选集合。于是决策接口可以从：

$$
\text{Context}\rightarrow\text{Next Token}
$$

改写成：

$$
\text{Context}+\text{Decision Slots}
\rightarrow
\text{Parallel Judgments}.
$$

这正是 Shared Yes/No 采用 DLLM 的核心原因：**我们希望把“待决策变量”直接表示成模型输入中的待恢复位置。**

这种表示同时复用了 DLLM 已有的三部分能力。

**双向上下文。** 每个 decision token 都能读取完整状态、题目和全部候选；不同 decision token 在同一个 Transformer 计算图中形成表示。

**原生 mask prediction。** DLLM 预训练本身就在学习如何根据上下文恢复 mask。Shared Yes/No 沿用同一输出接口，只把关注的词表行缩小到 Yes 和 No，因此可以直接从预训练模型出发测试 zero-shot 决策能力。

**多槽位并行。** 一道题中的多个候选可以同时拥有 decision token；进一步扩展到同一 state 下的多个问题时，也可以把多个输出变量同时放进一条序列，而不需要先人为指定生成顺序。

当前工作首先研究最简单的 $T=1$：所有 decision token 保持 mask，一次前向得到全部候选分数。DLLM 原生的 denoising 过程还提供了后续扩展空间，例如：

$$
[M,M,M,M]
\rightarrow
[\mathrm{Yes},M,\mathrm{No},M]
\rightarrow
[\mathrm{Yes},\mathrm{No},\mathrm{No},\mathrm{Yes}].
$$

这样，同一个接口可以从单步并行判断继续扩展到多步 refinement 和联合决策。

当前使用的 checkpoint 是 `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`。

## 2. 为什么每个选项一个决策 token？

确定使用 mask 表示决策变量之后，还有第二个选择：**只放一个答案 mask，预测 A/B/C；还是让每个候选拥有自己的 decision token？**

常见的单-mask 多选接口是：

```text
A. 直接安排发货
B. 因缺货取消订单
C. 请求补充商品数量

Answer: [MASK]
```

这种表示把候选语义先绑定到一组局部 identifier，再在答案位置预测 identifier。于是决策链条变成：

$$
\text{理解候选}
\rightarrow
\text{比较候选}
\rightarrow
\text{映射到 identifier}
\rightarrow
\text{预测 identifier token}.
$$

最后这层 `A/B/C` 映射本身并不承载候选语义。更重要的是，在语言模型里，`A`、`B`、`C` 并不是抽象的类别槽位，而是三个真实的 vocabulary token。它们拥有不同的输出向量和预训练统计。

在单个答案位置 $h_{\mathrm{ans}}$ 上，三个类别的分数分别是：

$$
\ell_A=W_A^\top h_{\mathrm{ans}}+b_A,\qquad
\ell_B=W_B^\top h_{\mathrm{ans}}+b_B,\qquad
\ell_C=W_C^\top h_{\mathrm{ans}}+b_C.
$$

因此，每个候选除了自己的语义之外，还绑定到了不同的输出方向 $W_A,W_B,W_C$。当候选顺序交换时，同一个语义候选会被重新绑定到另一组词表参数。例如：

```text
A. 直接安排发货
B. 因缺货取消订单
C. 请求补充商品数量
```

改成：

```text
A. 请求补充商品数量
B. 直接安排发货
C. 因缺货取消订单
```

“直接安排发货”这个语义没有变化，但它从 $W_A$ 对应的类别变成了 $W_B$ 对应的类别。**这就是 identifier bias 的来源之一：类别分数不仅取决于候选语义，也取决于该语义被分配到了哪个答案 token。**

候选数增加时，这个问题会继续放大。少量类别可以使用 `A/B/C/D`；几十个类别往往需要更多字母、数字甚至符号。不同 identifier 的词频、embedding、输出权重和 tokenizer 形式都可能不同，而这些差异与当前任务中的类别语义没有直接关系。

Shared Yes/No 的设计是把这层任意映射移到输入结构中：**每个候选拥有自己的 decision token，所有候选共享完全相同的输出语义。**

```text
State:
库存剩余 8 件，订单需要 3 件。

Question:
下一步应执行什么操作？

For each option, mark Yes if it answers the question, otherwise No.

直接安排发货: [MASK]
因缺货取消订单: [MASK]
请求补充商品数量: [MASK]
```

此时，第 $i$ 个候选首先得到自己的 contextual representation $h_i$。所有候选都读取同一对 Yes/No 词表权重：

$$
z_i^{Y}=W_Y^\top h_i+b_Y,\qquad
z_i^{N}=W_N^\top h_i+b_N.
$$

候选分数定义为：

$$
s_i=z_i^{Y}-z_i^{N}
=(W_Y-W_N)^\top h_i+(b_Y-b_N).
$$

这里最关键的变化是：**所有候选都沿着同一个输出方向 $W_Y-W_N$ 被评分。** 类别之间的区别放在 $h_i$ 中，由候选文本、state 和 question 共同决定；输出函数本身对每个候选完全共享。

对于互斥 `choice`，再在题目内部做 softmax：

$$
p_i=
\frac{\exp(s_i)}
{\sum_{j=1}^{K}\exp(s_j)}.
$$

由于 $(b_Y-b_N)$ 对所有候选相同，它在题内 softmax 中直接抵消。候选排名主要来自不同 contextual representation 在同一个 Yes-minus-No 判别方向上的投影，而不再为每个类别使用不同的 A/B/C 输出向量。

这个参数化带来三个直接好处。

1. **候选身份由语义表达。** 类别信息写在自然语言 option 中，而不是编码在任意 identifier token 里。
2. **判断函数跨候选共享。** 无论有 3 个、20 个还是 77 个候选，最终都使用同一个 Yes-minus-No 读出。
3. **每个候选都有自己的证据位置。** $h_i$ 专门服务于候选 $i$，同时又能通过双向注意力读取 state、question 和其他候选，从而完成候选级比较。

因此 Shared Yes/No 把常见的：

$$
\text{one answer state}+\text{many identifier-specific output rows}
$$

改写成：

$$
\text{many candidate states}+\text{one shared semantic scorer}.
$$

对 Banking77 这样的高候选数任务，77 个候选仍然使用同一对 Yes/No 权重；增加候选只会增加输入中的候选描述和 decision token，输出参数规模保持不变。

候选级评分也见于 Laya。Shared Yes/No 进一步把这种结构接到 MDLM 原生的 mask prediction 和预训练 Yes/No 词表权重上，使候选级决策可以从已有语言模型接口直接出发。

## 3. 一组分数，三种决策接口

有了共享读出，程序可以按任务含义组织输出。

| 接口 | 判断对象 | 概率与返回值 |
|---|---|---|
| `choice` | 动态候选集合 | 题内 softmax，返回各选项概率 |
| `noul` | 一个命题 | 单 mask 的 Yes/No 判断，返回 true/false 概率 |
| `score` | 有序等级集合 | 等级分布及其期望值 |

对于 `noul`，只设置一个决策 token：

$$
P(\mathrm{true})=\sigma\!\left(z^{\mathrm{Yes}}-z^{\mathrm{No}}\right).
$$

对于 `score`，保留等级值 $v_i$ 的顺序，通过题内 softmax 得到 $p_i$，再返回期望 $\hat v=\sum_i v_i p_i$。这样，系统既能读取最可能的等级，也能使用连续期望作排序。

三个接口共用 Yes/No 读出，同时保留各自的概率含义：choice 表达集合内的相对选择，noul 表达命题成立概率，score 表达有序等级分布。业务需要“证据不足”或“转交人工”时，可以把它们作为明确的候选，或交给单独的条件判断。

## 4. 用可迁移监督训练这个接口

Shared Yes/No 的监督直接作用于最终决策概率。对于 `choice`，设目标分布为 $q$，使用交叉熵：

$$
\mathcal L_{\mathrm{CE}}
=-\sum_{i=1}^{K} q_i\log p_i.
$$

`noul` 使用同样的 Yes/No log-odds，并对 true/false 概率计算二元交叉熵。

对于有序的 `score`，除了交叉熵，我们再加入 **normalized Ranked Probability Score（RPS）**。定义预测分布和目标分布的累计概率：

$$
F_p(k)=\sum_{i=0}^{k}p_i,
\qquad
F_q(k)=\sum_{i=0}^{k}q_i.
$$

对于 $K$ 个有序等级，normalized RPS 为：

$$
\mathrm{RPS}(p,q)
=\frac{1}{K-1}
\sum_{k=0}^{K-2}
\left(F_p(k)-F_q(k)\right)^2.
$$

最终的 score loss 写成：

$$
\mathcal L_{\mathrm{score}}
=
\mathcal L_{\mathrm{CE}}
+0.25\,\mathrm{RPS}(p,q)
$$

RPS 利用等级的顺序信息：预测偏离相邻等级和跨越多个等级会产生不同的累计误差，因此比单纯把等级看成无序类别更适合 ordinal decision。

公共数据的已有标签提供主要监督；规则和概率任务由程序给出可验证的答案或目标分布。所有损失按 decision 归一化，让不同候选数的题目保持清楚的训练权重。

我们从 10k 条决策的 S0 起步，扩充到 40k 条的 S1。S1 中 80% 来自已有公共标注的转换，涵盖 CLINC、SGD、SNLI、ARC、DBpedia、GoEmotions 和 BoolQ；20% 是程序构造的约束、状态、等级与已知概率机制任务。数据构建采用已有标注与程序验证。[^data]

这些来源共同训练的是自然语言与动态判断标准之间的匹配能力。每个来源保留自身标签语义，先按原始案例或对话分组，再生成决策视图。六个外部评测来源及其可追溯派生版本整体保留为评测数据，用于考察本项目新增监督的跨来源迁移。

## 5. 初步验证：从接口到可用的决策能力

下表展示完整测试集上的准确率。MDLM base 使用原始 checkpoint 的共享 Yes/No 读出；S1 使用完成 40k 监督适配后、按开发集 mean KL 选出的 step 1,300 checkpoint；Laya base 使用公开英文 checkpoint 与原生决策头。

| 测试集 | 决策数 | MDLM base | Shared Yes/No · S1 | Laya base |
|---|---:|---:|---:|---:|
| typed-decisions | 2,000 | 41.35% | **52.25%** | 35.85% |
| AG News† | 7,600 | 79.07% | 80.91% | **92.87%** |
| Emotion | 2,000 | 55.65% | 56.35% | **59.50%** |
| Banking77 | 3,080 | 24.09% | **60.49%** | 57.08% |
| Prompt Injection | 116 | **82.76%** | 70.69% | 69.83% |
| SST5 | 2,210 | 36.52% | **42.58%** | 35.02% |

† AG News 属于 Laya 官方说明中的训练混合来源。

适配后的模型在 Banking77 上达到 60.49%，较原始读出提高 36.40 个百分点；typed-decisions 从 41.35% 提高到 52.25%；SST5 从 36.52% 提高到 42.58%。这三个任务分别覆盖高候选数、混合决策原语和有序评分，展示了同一读出接口在多种任务结构中的适配能力。

### 概率质量与校准

决策接口返回的是概率，因此除了是否选对，还要看概率是否可靠。以下指标使用与准确率表相同的全量样本与 checkpoint，三个模型的温度均固定为 1，未额外做温度校准；MDLM 使用 BF16，Laya 使用 FP32。

- **ECE**：置信度与实际正确率的偏差，使用 15 个等宽置信度区间；下表以百分比显示。
- **NLL**：正确类别的平均负对数概率，对高置信度错误惩罚更大。
- **Brier**：预测概率与 gold 类别 one-hot 目标的平方误差之和，再对样本取均值；这里不混入 soft-target 版本。

三项指标都越低越好。ECE 描述校准偏差，NLL 和 Brier 则评价整体概率质量，不应与准确率相互替代。

**ECE（%）↓**

| 测试集 | MDLM base | Shared Yes/No · S1 | Laya base |
|---|---:|---:|---:|
| typed-decisions | 26.19% | **16.16%** | 30.90% |
| AG News† | 19.52% | 12.25% | **4.23%** |
| Emotion | **7.95%** | 23.02% | 30.58% |
| Banking77 | 19.06% | 19.57% | **16.48%** |
| Prompt Injection | **4.07%** | 19.98% | 29.13% |
| SST5 | **4.97%** | 20.77% | 33.58% |

**NLL ↓**

| 测试集 | MDLM base | Shared Yes/No · S1 | Laya base |
|---|---:|---:|---:|
| typed-decisions | 1.4582 | **1.0910** | 1.6282 |
| AG News† | 0.7015 | 0.8168 | **0.2539** |
| Emotion | **1.3282** | 1.4947 | 2.0146 |
| Banking77 | 3.7382 | **1.8990** | 2.2158 |
| Prompt Injection | **0.4150** | 0.6790 | 3.2636 |
| SST5 | 1.3969 | **1.3901** | 2.0661 |

**Brier ↓**

| 测试集 | MDLM base | Shared Yes/No · S1 | Laya base |
|---|---:|---:|---:|
| typed-decisions | 0.7946 | **0.6211** | 0.8653 |
| AG News† | 0.3601 | 0.3161 | **0.1181** |
| Emotion | **0.6226** | 0.6522 | 0.6963 |
| Banking77 | 0.9543 | **0.5928** | 0.6369 |
| Prompt Injection | **0.2623** | 0.4420 | 0.5701 |
| SST5 | **0.7147** | 0.7250 | 0.9037 |

这些结果展示了准确率之外的另一面：typed-decisions 上，S1 的准确率与三项概率指标同时改善；Emotion 上，准确率由 55.65% 小幅提高到 56.35%，但 ECE、NLL 和 Brier 都变差。Laya 在 Emotion 上准确率更高，却也有更大的概率误差。

低 ECE 也不一定意味着更强的分类能力。例如，原始 MDLM 在 SST5 上的 ECE 最低，但准确率只有 36.52%。因此，后续优化需要同时关注**答案选择与错误置信度**，不能仅凭一个指标判断决策模型是否更可靠。

### 候选并行与 batch 并行可以同时使用

单题内部，所有候选共享一次双向前向；多题之间，再通过 batch 提高吞吐。在同一张 RTX A6000、BF16 和同一 S1 checkpoint 上，17,006 条外部决策得到以下计时：

| 配置 | 均摊前向耗时 | 前向吞吐 |
|---|---:|---:|
| Batch 1 | 25.44 ms/decision | 39.30 decisions/s |
| Batch 32 | **6.61 ms/decision** | **151.26 decisions/s** |


## 6. Future Work

### 6.1 面向校准的训练 loss

决策模型最终交给程序的是概率，因此下一阶段会把**概率质量直接纳入训练目标**。当前 choice / noul 以交叉熵为主，score 额外使用 RPS；未来可以系统比较 log score、Brier score、RPS 以及轻量的分布保持项，让“选对答案”和“给出合适置信度”在同一个训练目标里协同优化。

一个直接的形式是：

$$
\mathcal L
=\mathcal L_{\mathrm{task}}
+\lambda_{B}\,\mathcal L_{\mathrm{Brier}}
+\lambda_{R}\,\mathcal L_{\mathrm{RPS}}.
$$

其中 RPS 只用于有序任务；Brier / log score 可以用于 choice 与 noul。已知概率机制、重复标注和可验证模拟器提供的 soft target 也可以用于研究概率学习。ECE、NLL、Brier 和 ordinal error 则继续作为统一的评测坐标。

另一个值得探索的方向，是在跨任务适配时加入原始模型分布的轻量保持项，让新增监督带来的能力扩展和原有概率尺度更平滑地衔接。

### 6.2 数据配比：从“更多数据”走向“更有效的能力覆盖”

现有结果已经体现出明显的任务差异。Banking77 与 typed-decisions 获得了较大的迁移收益，而 Emotion 的提升较小，Prompt Injection 的能力分布也发生了明显变化。这个现象说明，**训练数据与目标任务在“决策结构”上的对应关系，比单纯的数据行数更重要。**

例如，CLINC / SGD 与 Banking77 都强调细粒度意图和动态标签匹配，因此迁移路径较直接；GoEmotions 使用更细粒度的情绪体系，而且当前采样中 neutral 占比较高，与六类 Emotion 的有效覆盖并不均衡；BoolQ / SNLI 主要提供基于材料的真假与蕴含判断，而指令边界识别需要另一类语义证据。DBpedia 的实体本体分类同样与新闻主题判断存在不同的判别结构。

下一阶段会把数据 mixture 本身作为研究变量：

- 按 primitive、标签覆盖、候选数和难度做分层采样；
- 对不同来源做 source ablation，测量每个数据源带来的正迁移范围；
- 使用来源隔离 dev 选择 mixture weight；
- 增加 hard-negative 与边界样本，提高相近选项之间的分辨能力；
- 让数据预算围绕“能力覆盖”分配，而不是固定按来源平均分配。

这样可以进一步回答：哪些监督真正教会了模型一种可迁移的决策能力，哪些监督更偏向特定数据分布。

### 6.3 利用 DLLM 原生的迭代思考能力

当前主实验固定 $T=1$：所有决策 token 在一次双向前向中完成判断。这保留了极简的系统接口，也给后续研究留下了一个很有吸引力的方向——**让模型在决策空间里进行多步 refinement。**

一个自然的实现是为输入加入少量可迭代更新的 reasoning / workspace token，并让决策 token 与这些位置共同参与 denoising：

$$
X_T\rightarrow X_{T-1}\rightarrow\cdots\rightarrow X_0.
$$

第一步形成初始判断；后续步骤根据已经形成的局部结论、候选关系和中间工作区继续修正。可以进一步研究 confidence-based unmask / remask，让高置信决策先稳定下来，再为模糊候选提供新的上下文。

这条路线最值得比较的是 $T=1,2,4,8$ 的质量-计算曲线：DLLM 的价值从“并行 mask 分类器”逐步扩展到“可反复修正的决策推理器”，而最终接口仍然保持同一套 Shared Yes/No 概率读出。

### 6.4 联合问题决策：从一个问题扩展到多维决策

Shared Yes/No 当前以“一个问题、多个候选”为基本单元。更一般的实际系统经常面对同一个 state 下的多个相关决定，例如：事件类型、风险等级、下一步动作和升级路径；在游戏中，则可能同时决定多个单位的动作。

DLLM 很适合把这些输出组织成一组并行 decision tokens：

$$
[M_1],[M_2],\ldots,[M_m].
$$

所有 token 共享同一个 state，并在双向上下文中交换信息。进一步结合多步 refinement，可以直接研究：

$$
p(y_1,\ldots,y_m\mid x)
$$

并从各维度的边缘判断进一步扩展到联合分布。这样，模型可以让“动作”“风险”“资源分配”等多个决定在同一推理过程中相互协调。

这类任务会引入新的评价维度：joint exact match、约束满足率、组合动作有效率、环境 reward，以及联合概率质量。Overcooked、SMAC 一类协作环境，以及带明确规则的 workflow，都可以作为这一方向的测试场景。

## 7. 结语

Shared Yes/No 的核心思路很简单：**候选由自然语言描述，判断由共享权重完成，结果以结构化概率交给程序。** DLLM 提供了双向 mask 表示和原生词表读出，每个候选一个决策 token 则把动态选项直接映射到统一的判断接口。

当前实验已经展示了这一接口在高候选数、混合决策原语和有序评分上的可用性。接下来的重点，是让概率更可靠、数据覆盖更有效，并把 DLLM 的迭代推理与多维联合决策真正发挥出来。

语言理解负责适应场景，共享评分负责比较候选，结构化概率负责连接程序；在此基础上，diffusion 进一步为“并行、可修正、可联合”的决策过程提供了新的空间。

---

### 参考与代码

- 代码、训练/测试数据与复现命令：[zhouzihao11/jev-dllm](https://github.com/zhouzihao11/jev-dllm)。
- 原始模型：[Qwen3-0.6B MDLM](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)。
- 候选级决策模型与上游实现：[Laya](https://github.com/NandhaKishorM/laya)。

[^data]: 数据入口：[CLINC](https://github.com/clinc/oos-eval)、[SNLI](https://nlp.stanford.edu/projects/snli/)、[SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)；其余来源包括 `allenai/ai2_arc`、`fancyzhx/dbpedia_14`、`google-research-datasets/go_emotions`、`google/boolq`。
