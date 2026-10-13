# Dream-S1：原生共享 Yes/No 训练与 Core14 / Breadth-v1

本页发布已完成研究运行的**聚合结果、训练源码及原生评测入口**。Dream-S1 在本协议下提升 Core14 硬准确率和官方 Breadth 指数，但概率、校准和有序评分的迁移是混合的，不是全面质量改善，也不是架构因果结论。

**发布迁移状态：尚未完成远程 smoke 验证。** 下述训练、导出复载和完整评测数字来自既有研究运行，不是公开仓库重新训练所得。历史成功不等于本次公开迁移已验证；维护者将在独立远程验证后更新本段。本次不上传 Dream-S1 checkpoint、数据、原始标签/文本/预测、机器日志、服务器身份或私有路径，不承诺存在可下载的 Dream-S1 权重。既有 0.6B quickstart 和 HTTP 默认行为不变；不新增 Dream HTTP。

## 1. 来源、依赖与调用者输入

- 原模型：[Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)，固定 revision `05334cb9faaf763692dcf9d8737c642be2b2a6ae`。模型名为 7B，实际参数 7,615,616,512。请自行按来源许可获取该版本的完整本地模型并审阅自定义代码；加载会使用 `trust_remote_code=True`。
- 数据沿用原始 S1 的 **40,000 train 决策和原始 S0 1,000 dev**，不是 Kev-data 配方。已有公开构建入口、准备文件合同和使用边界见 [训练说明](training.md)、[S1_DATA_BUILDER](../research/S1_DATA_BUILDER.md)、[数据来源](data_and_models.md)。额外 2,000 dev 不用于选择。本页不新增数据转换器，不改变既有数据过滤或 criteria；重建必须检查实际计数及原始 dev 身份，不能用任意 40k 替代。
- 在仓库根目录使用 Python 3.10，安装既有 `requirements.txt`：已知研究 runtime 为 Torch `2.3.1+cu121`、Transformers `4.57.0`。这不是新的 baseline pin、完整锁文件或从零安装验证；CUDA 驱动、BF16 支持、显存和磁盘由调用者负责。官方 Kev 报告另用其兼容环境，不能覆盖主环境依赖。

```bash
python -m pip install -r requirements.txt
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export DREAM_MODEL=/path/to/reviewed/pinned/dream-model
export S1_TRAIN=/path/to/original/s1/canonical/train.jsonl
export S1_DEV=/path/to/original/s0/canonical/dev.jsonl
export TRAIN_OUT=/path/to/new/dream-training
export PREFLIGHT_OUT=/path/to/new/dream-preflight
export VERIFY_OUT=/path/to/new/dream-export-verification
export CORE_DATA=/path/to/prepared/core17
export CORE_OUT=/path/to/new/dream-core14
```

所有路径是调用者的模型、数据和输出，不是作者机器位置。训练路径固定为本地文件、离线加载。入口调用仓库已有 `_benchmark_bootstrap`，自动提供仓库根目录及必要的 `support/dllm_stub`，不需要私有 `PYTHONPATH`；真实 dllm 包存在时不替换它。

## 2. 正式训练、两更新预检和恢复

```bash
torchrun --standalone --nproc_per_node=2 research/scripts/train_dream_masked_typed.py \
  --model-path "$DREAM_MODEL" --train-data "$S1_TRAIN" --dev-data "$S1_DEV" \
  --output-dir "$TRAIN_OUT" --epochs 3 --global-batch 32 --seed 42 \
  --max-length 4096 --learning-rate 2.5e-5 --min-lr 1e-6 \
  --weight-decay 0.01 --warmup-ratio 0.05 --rps-weight 0.25 \
  --eval-every 100 --save-every 100 --log-every 10 \
  --train-limit 0 --dev-limit 0 --max-updates 0
```

两更新预检仍读取和验证**全 40k train / 1k dev**，保持正式 3 epoch 的学习率调度，不使用小数据替代，不强制最长输入；初始 dev、step1/step2 dev、best 导出和 sharded checkpoint 都可能写出，需要完整容量。每更新评测和保存，覆盖 step1/step2 的 latest/previous 发布。另用空目录：

```bash
torchrun --standalone --nproc_per_node=2 research/scripts/train_dream_masked_typed.py \
  --model-path "$DREAM_MODEL" --train-data "$S1_TRAIN" --dev-data "$S1_DEV" \
  --output-dir "$PREFLIGHT_OUT" --epochs 3 --global-batch 32 --seed 42 \
  --max-length 4096 --learning-rate 2.5e-5 --min-lr 1e-6 \
  --weight-decay 0.01 --warmup-ratio 0.05 --rps-weight 0.25 \
  --eval-every 1 --save-every 1 --log-every 1 \
  --train-limit 0 --dev-limit 0 --max-updates 2
```

正式训练从原模型新初始化，不把预检权重当作正式初始状态。FSDP `FULL_SHARD` 全参数训练，无 LoRA、量化或训练 CPU offload；原生 BF16 参数/梯度/Adam moments，损失 FP32，AdamW。microbatch=1，两 rank 各累积 16 决策得到 global32，每个 backward 同步；尾部 padding 决策零权重。loss 为 CE + 0.25 × normalized ordinal RPS（仅 score；除以 K−1）。cosine LR 2.5e-5→1e-6，warmup188 updates，WD0.01、clip1、seed42。全数据每 epoch1250 updates，共3750。

使用 `oldgeneric_noul_v1`，Dream chat template，不向 noul 注入 criteria；双向 `attention_mask="full"`，原始 unshifted hidden 取 mask−1，经既有公开 `yesno_projection` 投影 Yes/No 候选两行。choice/score 是 Yes−No 后 softmax，noul 顺序 `[No,Yes]`。无截断、静默丢行、生成、去噪或温度拟合。评测原生完整词表读出与训练候选投影路径不同，不宣称两者在任意硬件都 bitwise 等价。

恢复时重复原运行 CLI 的相同核心参数、模型/数据路径、输出目录和 world size，并添加 `--resume "$TRAIN_OUT/latest.json"`（预检使用其自身输出目录）。`--eval-every` 必须匹配原运行：正式为100，预检为1；resume 参数检查不允许将预检恢复为正式评测间隔。正式训练仍在独立空目录从原模型新初始化，使用上面的 `--eval-every 100`，不是接着预检训练。也可传同输出根下带 `complete.json` 的完整 generation 目录。保存模型、optimizer、scheduler、optimizer-boundary epoch/cursor 和 Python/NumPy/Torch/CUDA RNG；不从 best export 恢复 optimizer。`latest.json` 原子发布 latest 与 previous，保留最近两个完整 generations；未完成 `.tmp` 不作恢复点。latest 损坏时由调用者选择其 previous 完整目录；没有自动回退保证。`--max-updates` 是绝对停止 update，可从 capped 运行恢复为0；log/save 间隔可变，核心配方不能变化。BF16 重放不作跨设备逐位确定性保证。

best 用**原始 1,000 dev mean KL** 严格降低来选择（每100更新、epoch边界及停止点）；atomic HF export 保存模型、tokenizer 和本地自定义代码。export-scoped `DreamGenerationConfig.validate(strict=...)` 兼容修复只在保存时暂时修改类方法并恢复，不把补丁塞入可序列化实例。

## 3. 独立导出验证

从 `best.json` 读取所选路径及对应 dev reference 文件；输出应为空或不存在，reference 仅在调用者本地使用，不公开。

```bash
export DREAM_SELECTED="$(python -c 'import json,os; from pathlib import Path; r=Path(os.environ["TRAIN_OUT"]); print(r/json.loads((r/"best.json").read_text())["path"])')"
export DEV_REFERENCE="$(python -c 'import json,os; from pathlib import Path; r=Path(os.environ["TRAIN_OUT"]); print(r/json.loads((r/"best.json").read_text())["predictions_file"])')"
python research/scripts/verify_dream_export.py \
  --model-path "$DREAM_SELECTED" --dev-data "$S1_DEV" \
  --output-dir "$VERIFY_OUT" --reference-predictions "$DEV_REFERENCE" \
  --repeat-eval 2 --max-length 4096
```

这是独立进程 BF16 原生模型重载和候选投影 dev 评测，输出概率/KL/identity/shape/target/argmax 差异，**报告而非自动质量闸门**。历史所选模型两次完整1k dev复载的概率与 KL 差均精确0，所有身份/形状/目标不匹配及 argmax disagreement 为0。

## 4. Core14 原生评测

先按 [Core 来源与准备](benchmark_core.md) / [根 README](../README.md) 重建 Core17，保留成员和原顺序。Core14 是排除 MultiNLI、Aegis2、PAWS 的既定比较范围，不是为 Dream 认证的无污染集。直接对同一个现有 bundle 使用 `--suites` 选择下面14套；不复制或改写 inputs、targets、provenance、manifest 或任何 audit 数据。未传 `--suites` 时仍默认运行全部17套；选择按 bundle 顺序保留，run manifest 和 score report 明确标记 selected retained suite run。

```bash
python benchmark/evaluate.py --data-root "$CORE_DATA" --backend dream \
  --model-path "$DREAM_SELECTED" --output-dir "$CORE_OUT" \
  --device cuda:0 --dtype bfloat16 --batch-size 1 --max-length 4096 \
  --suites jevbench_public_original jevbench_public_easy jevbench_public_hard \
  jabr_v2 nimble_vitaminc-dev nimble_massive-en-US nimble_massive-de-DE \
  nimble_squad2 nimble_civil_comments nimble_helpsteer2 \
  nimble_summeval-relevance nimble_summeval-consistency nimble_pubmedqa contractnli
```

原生 `AutoModel` 一次完整词表 forward，双向 SDPA，mask位置−1取 `[Yes,No]`，T1、BF16、batch1、4096、不截断、不生成。可用另一个新输出目录添加 `--smoke --smoke-limit 3` 得到42条检查，不能当作全量结果。推理只读 manifest/inputs；Core scorer 独立连接 targets/provenance。自身成功5724/5945，超长221均在ContractNLI（1853/2074成功）。十条件比较共同成功5690，其中Contract1819；其余13套完整。共同成功均值和覆盖调整均值不可混用。

## 5. 全量 Breadth-v1：授权数据与官方报告

已有官方 [Kev 源码](https://github.com/jaredpalmer/kev/tree/fc4a17e194bca35a48c57211392ec49ea54a94ac) 与授权的 full frozen Breadth-v1 test，固定 integration revision `fc4a17e194bca35a48c57211392ec49ea54a94ac`。不要把 `scripts/prepare_kev.py` 的另一版本当作这个报告 revision。suite manifest 必须与所供 pinned checkout 完全一致，partition SHA、1990 records/3089 questions、14 datasets/5 areas 都保留；不排除训练来源重叠，不公开转存受限 partition 或 prepared bundle。获取和使用权限由调用者负责。详见 [Breadth bridge](../benchmark/BREADTH.md)。

```bash
export KEV_SOURCE=/path/to/reviewed/pinned/kev-source
export KEV_PYTHON=/path/to/existing/compatible/kev/python
export BREADTH_SUITE=/path/to/authorized/full/frozen/breadth-v1
export BREADTH_BUNDLE=/path/to/new/breadth-inputs
export BREADTH_OUT=/path/to/new/dream-breadth
export BREADTH_NATIVE=/path/to/new/dream-native-rows
export BREADTH_REPORT=/path/to/new/official-report
python benchmark/breadth.py prepare --kev-source "$KEV_SOURCE" \
  --suite "$BREADTH_SUITE" --split test --allow-test --out "$BREADTH_BUNDLE"
python benchmark/evaluate.py --data-root "$BREADTH_BUNDLE" --backend dream \
  --model-path "$DREAM_SELECTED" --output-dir "$BREADTH_OUT" \
  --device cuda:0 --dtype bfloat16 --batch-size 1 --max-length 4096 --predict-only
python benchmark/breadth.py export --kev-source "$KEV_SOURCE" \
  --suite "$BREADTH_SUITE" --split test --allow-test --data-root "$BREADTH_BUNDLE" \
  --predictions "$BREADTH_OUT/predictions.jsonl" --backend dream --out "$BREADTH_NATIVE"
"$KEV_PYTHON" benchmark/breadth.py report --kev-source "$KEV_SOURCE" \
  --suite "$BREADTH_SUITE" --split test --allow-test \
  --result Dream-S1="$BREADTH_NATIVE" --out "$BREADTH_REPORT"
```

**必须 `--predict-only`**，Core scorer 不是官方 Breadth scorer。prepare/export 是标准库 bridge，report 调用所供 checkout 的官方 `scripts/breadth_report.py`，保留官方 chance constants、index、coverage和bootstrap；没有复制或重写官方评分。若需要 paired comparison，按相同流程用原始 `$DREAM_MODEL` 另跑完整zero-shot至另一个native目录，再给report添加 `--result Dream-zero="$ZERO_NATIVE"`；已发布reference报告可按 bridge 文档添加，但没有本地行就不宣称paired CI。

## 6. 训练选择与既有独立检查

| 项目 | 既有正式结果 |
|---|---|
| 完成 | 3 epochs，3750 updates，completed=true，capped=false |
| 选定 | step2500，第二epoch终点（保存的零基epoch=1），不是最后权重 |
| 原始1000dev KL | 初始0.9039585416213376；selected0.1459667972254464；last3750为0.20896177570950744 |
| 时间 | 296974.386642秒，约82.49小时，含dev/save；硬件特定，不作速度排名 |
| 导出复载 | 两次完整1000dev概率/KL相对selected reference精确0差异 |
| Core独立检查 | 逐套四位小数ACC/ECE/NLL/Brier最大差0；logit重建概率最大差4.440892098500626e-16；全部成功行mask offset核对 |
| Breadth独立检查 | 官方score/skill/index/概率指标差≤7.105427357601002e-15 |

## 7. Core14：共同成功14套结果

每套指标先按历史四位小数口径再等权平均；下表百分数显示两位小数。Core ECE15为15个等宽 `(lo,hi]` bins，NLL floor1e-12，Brier为类别平方误差和。不是pooled ECE、官方sealed或Breadth ECE10。

| 条件，共同5690成功 | ACC% | ECE15% | NLL | Brier |
|---|---:|---:|---:|---:|
| Dream-S1 selected2500 | 70.72 | 17.53 | 0.8753 | 0.4320 |
| Dream zero-shot | 64.08 | 17.74 | 0.9186 | 0.4817 |
| Dream-S1 interim600，未选 | 68.07 | 12.08 | 0.7517 | 0.4080 |
| 0.6B S1 T1 | 63.97 | 14.92 | 0.9890 | 0.4851 |
| 0.6B S1 frozen calibration | 63.97 | 10.33 | 0.8808 | 0.4637 |
| Kev0.8B | 60.31 | 14.30 | 0.9160 | 0.4921 |
| Kev-data0.6B fixed endpoint（聚合背景） | 62.55 | 13.46 | 0.9781 | 0.4986 |

| Core suite | Common N | Final ACC% | Zero ACC% | Final ECE15% | Final NLL | Final Brier |
|---|---:|---:|---:|---:|---:|---:|
| Jev original | 72 | 97.22 | 75.00 | 3.83 | 0.0617 | 0.0349 |
| Jev easy | 48 | 100.00 | 100.00 | 0.18 | 0.0018 | 0.0001 |
| Jev hard | 111 | 43.24 | 40.54 | 40.97 | 1.7898 | 0.8820 |
| Jabr | 866 | 82.79 | 78.41 | 8.85 | 0.5168 | 0.2645 |
| VitaminC | 599 | 81.97 | 50.92 | 12.86 | 0.8070 | 0.3185 |
| MASSIVE en | 348 | 83.62 | 81.03 | 10.53 | 0.6803 | 0.2775 |
| MASSIVE de | 348 | 81.03 | 72.13 | 10.34 | 0.7456 | 0.2983 |
| SQuAD2 | 299 | 75.59 | 85.28 | 16.67 | 0.6731 | 0.3880 |
| Civil Comments | 300 | 91.00 | 72.67 | 4.67 | 0.2559 | 0.1447 |
| HelpSteer2 | 249 | 37.35 | 35.34 | 32.77 | 1.8759 | 0.8801 |
| SummEval relevance | 237 | 13.92 | 16.88 | 56.74 | 2.1045 | 1.1539 |
| SummEval consistency | 144 | 84.03 | 84.03 | 6.57 | 0.6620 | 0.2697 |
| PubMedQA | 250 | 51.20 | 53.20 | 26.94 | 1.2007 | 0.6668 |
| ContractNLI | 1819 | 67.12 | 51.73 | 13.44 | 0.8789 | 0.4691 |

Final相对zero ACC +6.64百分点，9胜/3负/2平；相对0.6B S1为10胜/3负/1平；相对Kev为12胜/0负/2平，仅描述性计数，无Core CI。自身成功宏均值ACC70.71357%、ECE17.53143%、NLL0.87555、Brier0.432157；**all-attempted宏ACC70.20399%**，不同于共同成功70.72%。自身Contract1242/1853=67.02644%；全attempted1242/2074=59.88428%。SummEval relevance ACC13.92% / ECE56.74%仍很差。interim600概率指标更好却未选：选择只用原始dev KL，不用test挑checkpoint。

Pooled score n841：final MAE0.72341388 vs zero0.85703560改善；未归一化RPS0.57144359 vs0.56081700反而变差。Pooled soft-target n310：final CE0.34271238/Brier0.07790246 vs zero0.59631268/0.28391100改善。这些不是suite macro，不支持概率普遍改善。

## 8. Breadth-v1：全14 datasets

官方index **36.2081603223**，95% record-bootstrap CI **[33.9340,39.1567]**；zero31.5401768589，delta **+4.6679834634**，paired95%CI **[1.3452,7.8761]**（2000resamples，seed0）。9胜/5负/0平。CI是固定测试records的采样不确定性，不是训练seed/架构因果不确定性，无多重比较校正。

全部3089questions保留，成功3069；20超长均在Contract，回答140/160。unsupported在官方main分母计错。SATA/BFCL为case-exact，其他12为question accuracy；main score和calibration的question分母不同。Breadth ECE10使用10个等宽 `[lo,hi)` bins，最后上端点包含，NLL直接用精确logits（无log(p) floor替代），Brier class sum。不能与Core合并。

| Breadth dataset | Main metric | Final score% | Zero score% | Coverage% | Final ECE10% | Final NLL | Final Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| MuSR | accuracy | 52.00 | 48.67 | 100 | 24.18 | 1.4262 | 0.7100 |
| SATA | case-exact | 28.67 | 14.00 | 100 | 11.60 | 0.5102 | 0.3046 |
| Chess | accuracy | 13.33 | 11.33 | 100 | 17.64 | 2.2971 | 0.9197 |
| ContractNLI | accuracy | 42.50 | 31.25 | 87.5 | 25.19 | 1.2462 | 0.6984 |
| HellaSwag | accuracy | 64.67 | 76.00 | 100 | 22.28 | 1.2847 | 0.5679 |
| CLINC | accuracy | 96.67 | 94.00 | 100 | 2.10 | 0.0911 | 0.0476 |
| SGD | accuracy | 89.33 | 64.00 | 100 | 5.49 | 0.3212 | 0.1716 |
| BRIGHT | accuracy | 56.67 | 42.00 | 100 | 12.82 | 1.2282 | 0.5751 |
| BFCL | case-exact | 86.00 | 83.33 | 100 | 2.71 | 0.2334 | 0.1435 |
| ToolRet | accuracy | 51.33 | 55.33 | 100 | 30.21 | 2.0699 | 0.7457 |
| API-Bank | accuracy | 88.00 | 89.33 | 100 | 6.15 | 0.4891 | 0.1920 |
| RouterBench | accuracy | 27.33 | 24.67 | 100 | 16.37 | 1.7761 | 0.8099 |
| Humicroedit | accuracy | 46.67 | 56.00 | 100 | 35.14 | 1.3957 | 0.7981 |
| cfcolor | accuracy | 52.67 | 48.67 | 100 | 19.47 | 0.8388 | 0.5985 |

Contract主score68/160=42.50%含20unsupported错误；answered68/140=48.5714%。Pooled answered final ECE13.54928% / NLL0.88802267 **差于**zero5.67067% /0.81912338；Brier0.44269084优于0.46552760。平均置信度81.1283% vs answered准确率67.5790%，过度自信；mean-dataset ECE16.52407%。不能以index改善掩盖校准/NLL退化。

## 9. 失败、边界与发布归属

历史预检在generation-config导出遇到unexpected `strict` 参数，export-scoped兼容修复后通过。一次独立复载在推理前因其他进程占用GPU而OOM，换可用设备，不改变精度、不终止他人任务。另一次Breadth启动错误调用Core scorer，保留预测，改为现有predict-only/官方report流程。独立Breadth校验器最初误用Core右闭bins，在边界值出现差异；改回官方左闭bins后复核一致，官方推理和分数未改变。不公开这些失败的私有任务标识或机器日志。

单一训练seed；公开test和interim600结果已观察，不是新blind test。Dream上游预训练/SFT暴露未知，S1的CLINC/SGD训练来源与Breadth已知重叠**未过滤**，其强表现不能称完全未见迁移。Core三套相关Jev拆分的等权macro、Breadth五area/chance clipping都是特定约定，不是普适utility。模型大小、预训练、SFT、数据、优化、模板和读出精度不同，不作架构因果或公平吞吐比较；没有全任务、所有概率指标或通用能力提升结论。

Dream原始读出位置依据 [固定generation_utils.py](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/05334cb9faaf763692dcf9d8737c642be2b2a6ae/generation_utils.py)。公开实现见 [trainer](../research/scripts/train_dream_masked_typed.py)、[小型training helper](../research/scripts/dream_training.py)、[独立verifier](../research/scripts/verify_dream_export.py)、[native adapter](../benchmark/adapters/dream.py) 和 [Breadth bridge](../benchmark/breadth.py)。继承来源为 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)；归属和许可见 [THIRD_PARTY](../THIRD_PARTY.md)。仅抽出原probe的CandidateWrapper和memory，不迁移整个诊断/审计probe；既有公共projection和loader默认行为保留。

发布仅源码和聚合文档；受限数据、checkpoint、自定义上游模型代码及逐题证据均由调用者另行授权获取和保存。本地完成的检查仅AST语法与diff/敏感路径审阅；尚未对本次公开迁移执行远程训练preflight、resume、export复载、Core42或Breadth bridge smoke，维护者验证后需明确更新结果及其覆盖范围，不把历史结果当成这些新检查。
