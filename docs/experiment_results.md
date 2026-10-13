| 结果口径 | 配置 / 单位 |
|---|---|
| Original DLLM | 原始 Qwen3-0.6B-diffusion-mdlm-v0.1；S0/S1 只列 dev 选定的 best |
| Legacy 六套主结果 | DLLM BF16、batch1；Laya FP32 eager、aligned_unit、原生长度排序 batch≤16；上限4096，不截断 |
| Audited Core | 17套、6744条；DLLM BF16 batch32，Laya FP32 batch1；上限4096，不截断 |
| ACC / 校准 | ACC↑、ECE↓单位%；Core/Legacy为ECE15，官方公开集为ECE10；NLL↓为自然对数，Brier↓为类别平方误差和 |
| Internal S0 test | 950个硬目标+50个分布目标；CE/KL/Brier按全部1000条原生目标计算 |
| MAE / RPS | MAE为预测期望评分误差；Core RPS未归一化，训练/internal RPS除以K−1；均越低越好 |
| Nimble12 平均 | 12个子集指标等权平均；不是 pooled ECE 或类别数归一化 Brier |
| 官方公开集 HTTP | jevbench::v1.4，harness v1.4.2，仅231条公开题；BF16 batch1、单次forward、温度1；无sealed综合分数/排名 |
| HTTP / forward 时间 | HTTP为预加载后的loopback请求耗时；forward为A6000同步前向耗时，每套3次不计时warmup；两者不可混用 |

| Primary training / selected checkpoint | S0 | S1 |
|---|---:|---:|
| Train decisions | 10,000 | 40,000 |
| Selection dev decisions | 1,000 | 1,000 (same S0 dev) |
| Completed epochs | 3 | 3 |
| Completed optimizer updates | 939 | 3,750 |
| Selected best update | 313 | 1,300 |
| Best dev mean KL | 0.29016964 | 0.26017353 |
| Training-loop seconds (including dev and saves) | 3,292.903 | 13,384.197 |

| Legacy full external: ACC / ECE15 (%) | Decisions | Original DLLM | S0 best | S1 best | Laya-base |
|---|---:|---:|---:|---:|---:|
| typed-decisions | 2,000 (400 cases) | 41.35 / 26.19 | 50.30 / 11.84 | 52.25 / 16.16 | 35.85 / 30.90 |
| AG News | 7,600 | 79.07 / 19.52 | 82.55 / 9.38 | 80.91 / 12.25 | 92.87 / 4.23 |
| Emotion | 2,000 | 55.65 / 7.95 | 53.80 / 24.82 | 56.35 / 23.02 | 59.50 / 30.58 |
| Banking77 | 3,080 (77 classes) | 24.09 / 19.06 | 61.27 / 12.79 | 60.49 / 19.57 | 57.08 / 16.48 |
| Prompt Injection | 116 | 82.76 / 4.07 | 71.55 / 19.90 | 70.69 / 19.98 | 69.83 / 29.13 |
| SST5 | 2,210 | 36.52 / 4.97 | 43.39 / 21.78 | 42.58 / 20.77 | 35.02 / 33.58 |

| Legacy full external: NLL / Brier (hard targets) | Original DLLM | S0 best | S1 best | Laya-base |
|---|---:|---:|---:|---:|
| typed-decisions | 1.4582 / 0.7946 | 1.1270 / 0.6401 | 1.0910 / 0.6211 | 1.6282 / 0.8653 |
| AG News | 0.7015 / 0.3601 | 0.6334 / 0.2791 | 0.8168 / 0.3161 | 0.2539 / 0.1181 |
| Emotion | 1.3282 / 0.6226 | 1.6734 / 0.6922 | 1.4947 / 0.6522 | 2.0146 / 0.6963 |
| Banking77 | 3.7382 / 0.9543 | 1.7359 / 0.5495 | 1.8990 / 0.5928 | 2.2158 / 0.6369 |
| Prompt Injection | 0.4150 / 0.2623 | 0.7515 / 0.4682 | 0.6790 / 0.4420 | 3.2636 / 0.5701 |
| SST5 | 1.3969 / 0.7147 | 1.4970 / 0.7448 | 1.3901 / 0.7250 | 2.0661 / 0.9037 |

| Internal S0 test: native-target metrics | Denominator | Original DLLM | S0 best | S1 best | Laya-base |
|---|---:|---:|---:|---:|---:|
| Hard correct / total | 950 | 419/950 | 827/950 | 850/950 | 573/950 |
| Hard ACC / ECE15 (%) | 950 | 44.11 / 21.61 | 87.05 / 2.00 | 89.47 / 2.55 | 60.32 / 13.11 |
| Mean CE | 1,000 | 1.498726 | 0.348250 | 0.314472 | 1.202802 |
| Mean KL(target to prediction) | 1,000 | 1.465985 | 0.315509 | 0.281731 | 1.170061 |
| Brier vs native targets | 1,000 | 0.715452 | 0.168714 | 0.141739 | 0.519331 |
| Normalized score RPS | 160 | 0.268223 | 0.036174 | 0.019463 | 0.254985 |
| Hard-score MAE | 150 | 1.153527 | 0.216951 | 0.139817 | 1.084838 |

| Audited Core: common-success ACC / ECE15 (%) | Common / retained | Original DLLM | S0 best | S1 best | Laya-base |
|---|---:|---:|---:|---:|---:|
| JevBench original | 72/72 | 54.17 / 18.32 | 81.94 / 8.98 | 86.11 / 12.11 | 68.06 / 19.97 |
| JevBench easy | 48/48 | 100.00 / 12.18 | 100.00 / 1.70 | 100.00 / 0.78 | 100.00 / 2.69 |
| JevBench hard | 111/111 | 42.34 / 31.15 | 31.53 / 28.80 | 40.54 / 29.81 | 32.43 / 33.62 |
| Jabr v2 | 866/866 | 50.81 / 15.51 | 65.82 / 10.74 | 65.94 / 13.40 | 58.66 / 19.35 |
| VitaminC | 599/599 | 50.25 / 39.54 | 69.62 / 7.71 | 67.95 / 15.95 | 78.96 / 12.35 |
| MASSIVE en-US | 348/348 | 69.83 / 40.30 | 75.29 / 10.43 | 77.87 / 10.53 | 57.47 / 9.32 |
| MASSIVE de-DE | 348/348 | 46.26 / 25.64 | 55.75 / 17.83 | 59.77 / 23.54 | 34.48 / 9.60 |
| SQuAD2 | 299/299 | 70.57 / 6.72 | 71.57 / 3.11 | 67.56 / 11.51 | 67.56 / 26.86 |
| PAWS | 250/250 | 71.60 / 3.51 | 79.20 / 13.58 | 76.40 / 7.04 | 54.00 / 42.01 |
| MultiNLI | 299/299 | 36.12 / 51.35 | 72.91 / 7.96 | 77.59 / 6.12 | 87.63 / 9.95 |
| Civil Comments | 300/300 | 62.33 / 15.00 | 74.67 / 5.88 | 82.67 / 4.21 | 93.33 / 6.65 |
| Aegis2 | 250/250 | 62.40 / 14.31 | 78.80 / 9.57 | 72.40 / 13.88 | 42.80 / 53.27 |
| HelpSteer2 | 249/249 | 42.17 / 28.86 | 34.14 / 5.96 | 34.14 / 26.25 | 42.57 / 11.07 |
| SummEval relevance | 237/237 | 3.80 / 21.29 | 3.80 / 22.53 | 15.61 / 14.19 | 21.10 / 13.72 |
| SummEval consistency | 144/144 | 84.03 / 30.98 | 83.33 / 35.50 | 84.72 / 13.08 | 22.92 / 13.31 |
| PubMedQA | 250/250 | 53.20 / 44.11 | 46.40 / 11.67 | 57.60 / 14.80 | 52.40 / 38.27 |
| ContractNLI | 1,819/2,074 | 45.79 / 47.92 | 61.19 / 4.64 | 55.09 / 18.65 | 48.98 / 21.36 |

| Audited Core: common-success NLL / Brier | Original DLLM | S0 best | S1 best | Laya-base |
|---|---:|---:|---:|---:|
| JevBench original | 1.1541 / 0.5546 | 0.4755 / 0.2793 | 0.4685 / 0.2260 | 0.9181 / 0.4967 |
| JevBench easy | 0.1478 / 0.0501 | 0.0182 / 0.0031 | 0.0081 / 0.0008 | 0.0296 / 0.0087 |
| JevBench hard | 1.5221 / 0.8573 | 1.4076 / 0.7933 | 1.5649 / 0.8422 | 1.6721 / 0.9122 |
| Jabr v2 | 1.2801 / 0.6534 | 0.7971 / 0.4513 | 0.8047 / 0.4535 | 1.1731 / 0.5908 |
| VitaminC | 1.4126 / 0.8051 | 0.7607 / 0.4305 | 0.9400 / 0.4709 | 0.7710 / 0.3675 |
| MASSIVE en-US | 1.5694 / 0.6175 | 0.9178 / 0.3660 | 0.9608 / 0.3330 | 1.3709 / 0.5421 |
| MASSIVE de-DE | 2.1690 / 0.7801 | 1.6144 / 0.5970 | 1.7153 / 0.6034 | 2.1582 / 0.7723 |
| SQuAD2 | 0.5673 / 0.3805 | 0.5418 / 0.3666 | 0.6478 / 0.4383 | 1.1885 / 0.5605 |
| PAWS | 0.5416 / 0.3676 | 0.5315 / 0.3493 | 0.4637 / 0.3004 | 1.6694 / 0.8308 |
| MultiNLI | 1.7038 / 0.9852 | 0.6051 / 0.3495 | 0.5560 / 0.3134 | 0.5247 / 0.2154 |
| Civil Comments | 0.7645 / 0.5209 | 0.5234 / 0.3447 | 0.4226 / 0.2650 | 0.1716 / 0.0948 |
| Aegis2 | 0.6371 / 0.4434 | 0.5564 / 0.3348 | 0.6325 / 0.3944 | 2.4620 / 1.0621 |
| HelpSteer2 | 1.5912 / 0.8190 | 1.5173 / 0.7566 | 1.7176 / 0.8193 | 1.3824 / 0.7175 |
| SummEval relevance | 1.7388 / 0.8508 | 1.7099 / 0.8414 | 1.6434 / 0.8170 | 1.5498 / 0.8012 |
| SummEval consistency | 0.8626 / 0.4106 | 0.9627 / 0.4580 | 0.7879 / 0.3175 | 1.7786 / 0.8705 |
| PubMedQA | 1.9746 / 0.8925 | 1.1471 / 0.6420 | 1.1227 / 0.5909 | 1.9853 / 0.8452 |
| ContractNLI | 2.2390 / 0.9995 | 0.8512 / 0.5103 | 1.0413 / 0.6131 | 1.2423 / 0.7122 |

| Nimble12: equal-subset means | ACC (%) | ECE15 (%) | NLL | Brier |
|---|---:|---:|---:|---:|
| Original DLLM | 54.38 | 26.80 | 1.2944 | 0.6561 |
| S0 best | 62.12 | 12.64 | 0.9490 | 0.4864 |
| S1 best | 64.52 | 13.43 | 0.9675 | 0.4720 |
| Laya-base | 54.60 | 20.53 | 1.4177 | 0.6400 |

| Scalar / native soft-target diagnostics | Metric / n | Original DLLM | S0 best | S1 best | Laya-base |
|---|---|---:|---:|---:|---:|
| Core HelpSteer2 | MAE / RPS; n=249 | 1.018557 / 0.771625 | 1.175459 / 0.748978 | 1.052851 / 0.781879 | 0.975935 / 0.639317 |
| Core SummEval relevance | MAE / RPS; n=237 | 0.959439 / 0.692516 | 0.963858 / 0.690411 | 0.843812 / 0.612250 | 0.864103 / 0.608253 |
| Core SummEval consistency | MAE / RPS; n=144 | 1.047714 / 0.467928 | 1.147673 / 0.541745 | 0.658880 / 0.345174 | 1.595597 / 1.055027 |
| Legacy typed-decisions score | MAE; n=800 | 0.6988 | 0.5185 | 0.5196 | 0.7051 |
| Legacy SST5 score | MAE; n=2,210 | 0.7903 | 0.6794 | 0.6757 | 0.9123 |
| Legacy typed-decisions soft targets | Mean dot(p,t) / Brier vs soft; n=2,000 | 0.3582 / 0.4066 | 0.3951 / 0.2745 | 0.4271 / 0.2734 | 0.3371 / 0.4356 |

| Coverage / probability denominators | Model(s) | Attempted | Successful | Unsupported length | Common-success n | Native probability n | All-attempted ACC (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Legacy external six suites | Each of Original DLLM, S0, S1, Laya-base | 17,006 | 17,006 | 0 | 17,006 | 17,006 | - |
| Internal S0 test | Each of Original DLLM, S0, S1, Laya-base | 1,000 | 1,000 | 0 | 1,000 | 1,000 (950 hard) | - |
| Core 17 suites | Original DLLM | 6,744 | 6,523 | 221 | 6,489 | 6,523 | - |
| Core 17 suites | S0 best | 6,744 | 6,523 | 221 | 6,489 | 6,523 | - |
| Core 17 suites | S1 best | 6,744 | 6,523 | 221 | 6,489 | 6,523 | - |
| Core 17 suites | Laya-base | 6,744 | 6,489 | 255 | 6,489 | 6,489 | - |
| ContractNLI (unsupported counted wrong) | Original DLLM | 2,074 | 1,853 | 221 | 1,819 | 1,853 | 40.45 |
| ContractNLI (unsupported counted wrong) | S0 best | 2,074 | 1,853 | 221 | 1,819 | 1,853 | 54.29 |
| ContractNLI (unsupported counted wrong) | S1 best | 2,074 | 1,853 | 221 | 1,819 | 1,853 | 48.70 |
| ContractNLI (unsupported counted wrong) | Laya-base | 2,074 | 1,819 | 255 | 1,819 | 1,819 | 42.96 |
| Official public HTTP (all HTTP 200, strict valid) | Each of Original DLLM, S0, S1 | 231 | 231 | 0 | 231 | 231 | - |

| Official JevBench public HTTP | Model | Correct / n | ACC (%) | ECE10 (%) | Brier (sum) | Supplemental NLL (aggregate) | HTTP mean / p50 / p95 ms (aggregate) |
|---|---|---:|---:|---:|---:|---:|---:|
| All public | Original DLLM | 135/231 | 58.44 | 12.97 | 0.595113 | 1.122264 | 49.80 / 38.19 / 100.69 |
| All public | S0 best | 142/231 | 61.47 | 13.19 | 0.468799 | 0.826951 | 49.47 / 38.02 / 99.66 |
| All public | S1 best | 155/231 | 67.10 | 14.25 | 0.476916 | 0.902037 | 48.56 / 36.66 / 98.70 |
| Original | Original DLLM | 40/72 | 55.56 | 15.88 | 0.5544 | - | - |
| Original | S0 best | 59/72 | 81.94 | 10.54 | 0.2786 | - | - |
| Original | S1 best | 62/72 | 86.11 | 5.45 | 0.2259 | - | - |
| Easy | Original DLLM | 48/48 | 100.00 | 12.15 | 0.0500 | - | - |
| Easy | S0 best | 48/48 | 100.00 | 1.67 | 0.0029 | - | - |
| Easy | S1 best | 48/48 | 100.00 | 0.74 | 0.0007 | - | - |
| Hard | Original DLLM | 47/111 | 42.34 | 30.29 | 0.8572 | - | - |
| Hard | S0 best | 35/111 | 31.53 | 28.84 | 0.7937 | - | - |
| Hard | S1 best | 45/111 | 40.54 | 29.73 | 0.8456 | - | - |

| 官方公开集：改写与评分 | 改写对数 | 改写预测一致率 (%) | 两题均正确率 (%) | Score MAE↓ | Strict-valid 请求 |
|---|---:|---:|---:|---:|---:|
| Original DLLM | 36 | 83.33 | 50.00 | 0.741449 | 231/231 |
| S0 best | 36 | 86.11 | 75.00 | 0.646475 | 231/231 |
| S1 best | 36 | 94.44 | 83.33 | 0.636221 | 231/231 |

| S1 best: legacy six-suite BF16 forward throughput (not request latency) | Decisions | Batch 1 ms/decision | Batch 32 amortized ms/decision | Speedup | Batch 32 decisions/s | Batch 1 / 32 ACC (%) | Changed argmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed-decisions | 2,000 | 25.31 | 9.13 | 2.77x | 109.49 | 52.2500 / 52.2000 | 15 |
| AG News | 7,600 | 25.45 | 4.23 | 6.02x | 236.68 | 80.9079 / 80.9737 | 15 |
| Emotion | 2,000 | 25.86 | 3.07 | 8.42x | 325.69 | 56.3500 / 56.3500 | 12 |
| Banking77 | 3,080 | 25.06 | 15.59 | 1.61x | 64.14 | 60.4870 / 60.5844 | 9 |
| Prompt Injection | 116 | 25.92 | 3.92 | 6.62x | 255.36 | 70.6897 / 70.6897 | 0 |
| SST5 | 2,210 | 25.67 | 3.36 | 7.64x | 297.44 | 42.5792 / 42.4887 | 21 |
| All six external suites | 17,006 | 25.4437 | 6.6109 | 3.8487x | 151.26 | - | 72 |

| Kev 比较口径（既有历史表不变） | 范围 / 限制 |
|---|---|
| 主范围 Core14 | 原 Core17 排除 Kev 梯度训练来源 MultiNLI/Aegis2、开发接触 PAWS；5,945 条尝试，五模型共同成功 5,690 条 |
| Legacy typed-decisions | 另列 2,000 条；不与 Core 合并为总体排名；已知训练/开发来源的其他 legacy 套件不加 Kev 值 |
| 来源保证 | 未确认训练或开发来源重叠，不保证无污染；预训练未知、私有基础模型梯度训练资料不完全可得 |
| 协议 | 其他四模型保留旧 renderer；S1 Core JevBench BF16 batch32 非官方 sealed；较新 T=1 HTTP 另列不混用 |
| 温度 / 运行 | Kev FP32 batch1，checkpoint T=2.3510958125672174；其余旧 T=1；无 Core 温度拟合 |
| 证据 | 历史 Core native parity 51 条，最大概率差 0；本次发布迁移未重新执行 Kev 推理 |
| 完整五模型表 | [ACC/ECE、NLL/Brier、MAE/RPS、固定 revisions 与覆盖](kev_comparison.md)；不作架构因果或匹配吞吐声明 |

| 来源筛选 Core14 + 另列 legacy：ACC / ECE15 (%) | Common / attempted | S0 best（旧） | S1 best（旧） | Kev-0.8B |
|---|---:|---:|---:|---:|
| JevBench original | 72/72 | 81.94 / 8.98 | 86.11 / 12.11 | 80.56 / 16.79 |
| JevBench easy | 48/48 | 100.00 / 1.70 | 100.00 / 0.78 | 100.00 / 6.87 |
| JevBench hard | 111/111 | 31.53 / 28.80 | 40.54 / 29.81 | 36.94 / 18.07 |
| Jabr v2 | 866/866 | 65.82 / 10.74 | 65.94 / 13.40 | 71.94 / 8.37 |
| VitaminC | 599/599 | 69.62 / 7.71 | 67.95 / 15.95 | 66.28 / 6.67 |
| MASSIVE en-US | 348/348 | 75.29 / 10.43 | 77.87 / 10.53 | 75.00 / 20.82 |
| MASSIVE de-DE | 348/348 | 55.75 / 17.83 | 59.77 / 23.54 | 66.09 / 17.33 |
| SQuAD2 | 299/299 | 71.57 / 3.11 | 67.56 / 11.51 | 66.22 / 3.54 |
| Civil Comments | 300/300 | 74.67 / 5.88 | 82.67 / 4.21 | 83.67 / 14.50 |
| HelpSteer2 | 249/249 | 34.14 / 5.96 | 34.14 / 26.25 | 28.92 / 9.37 |
| SummEval relevance | 237/237 | 3.80 / 22.53 | 15.61 / 14.19 | 13.92 / 27.24 |
| SummEval consistency | 144/144 | 83.33 / 35.50 | 84.72 / 13.08 | 54.86 / 22.56 |
| PubMedQA | 250/250 | 46.40 / 11.67 | 57.60 / 14.80 | 44.80 / 19.79 |
| ContractNLI | 1,819/2,074 | 61.19 / 4.64 | 55.09 / 18.65 | 55.20 / 8.32 |
| Legacy typed-decisions（另列） | 2,000/2,000 | 50.30 / 11.84 | 52.25 / 16.16 | 45.95 / 5.49 |

| Nimble9: equal-subset means（排除 MultiNLI/Aegis2/PAWS） | ACC (%) | ECE15 (%) | NLL | Brier(sum) |
|---|---:|---:|---:|---:|
| Original DLLM | 53.60 | 28.05 | 1.4056 | 0.6752 |
| S0 best | 57.17 | 13.40 | 1.0772 | 0.5336 |
| S1 best | 60.88 | 14.90 | 1.1065 | 0.5173 |
| Laya-base | 52.31 | 15.68 | 1.3729 | 0.6191 |
| Kev-0.8B | 55.53 | 15.76 | 1.0285 | 0.5372 |

| 新增 Kev 覆盖 / 概率分母 | Attempted | Successful | Unsupported length | Five-model common n | All-attempted ACC (%) |
|---|---:|---:|---:|---:|---:|
| 原 Core17（覆盖记录，非来源筛选质量总分） | 6,744 | 6,523 | 221 | 6,489 | - |
| Core14（质量比较范围） | 5,945 | 5,724 | 221 | 5,690 | - |
| ContractNLI | 2,074 | 1,853 | 221 | 1,819 | 49.23 |
| Legacy typed-decisions（另列） | 2,000 | 2,000 | 0 | 2,000 | 45.95 |

| S1 全局温度校准：固定协议 | 值 / 限制 |
|---|---|
| 全局温度 / 逆温度 | T=1.8739716972345604；beta=0.5336259888426866；choice/score/noul共用 |
| 校准数据 | 六套旧test明确改作校准：typed2000、AG7600、Emotion2000、Banking77 3080、Prompt Injection116、SST5 2210；共17006 |
| 拟合目标 | 六套等权unclipped hard-label NLL；不按样本量合并，不用Core标签或soft target拟合 |
| 模型 / logits | S1原dev选定best update1300；复用BF16 batch32、旧generic noul renderer、T1原始logits；不改权重或模板 |
| 校准集NLL | 1.2280664850 → 1.0249981928；校准后六套不是独立测试证据 |
| Core14 | 排除Kev已确认MultiNLI/Aegis2/PAWS来源；5945 attempted；同分母比较5690共同成功，ContractNLI1819 |
| 总体均值 | 各套指标等权平均，非pooled ECE；历史子集指标舍入，不作官方sealed综合排名 |
| 验证 | 校准/Core T1重构概率误差0；Core14 argmax变化0；221条S1超长记录原样保留；独立逐条重算复现评分 |
| 发布范围 | 仅结果与算法说明；默认评测/HTTP仍T1；未迁移研究用温度拟合/应用CLI |
| 完整说明 | [温度校准、覆盖、Score诊断与限制](temperature_calibration.md)；无显著性或架构因果结论 |

| Core14：校准后同分母比较（5690共同成功，14套等权） | 温度 | ACC (%) ↑ | ECE15 (%) ↓ | NLL ↓ | Brier(sum) ↓ |
|---|---|---:|---:|---:|---:|
| S1 best，旧T1 | 1 | 63.97 | 14.92 | 0.9890 | 0.4851 |
| S1 best，全局校准 | 1.87397 | 63.97 | 10.33 | 0.8808 | 0.4637 |
| Kev-0.8B，原生FP32 batch1 | 2.35110，checkpoint自带 | 60.31 | 14.30 | 0.9160 | 0.4921 |
| Laya-base，FP32 batch1 aligned_unit | 1，未重新拟合 | 55.64 | 17.01 | 1.2423 | 0.5923 |

| Core14：S1校准前→后（同分母） | Common / attempted | ACC，不变 (%) | NLL | ECE15 (%) | Brier(sum) |
|---|---:|---:|---:|---:|---:|
| JevBench original | 72/72 | 86.11 | 0.4685 → 0.4936 | 12.11 → 15.45 | 0.2260 → 0.2639 |
| JevBench easy | 48/48 | 100.00 | 0.0081 → 0.0517 | 0.78 → 4.76 | 0.0008 → 0.0110 |
| JevBench hard | 111/111 | 40.54 | 1.5649 → 1.2403 | 29.81 → 15.47 | 0.8422 → 0.7261 |
| Jabr v2 | 866/866 | 65.94 | 0.8047 → 0.7326 | 13.40 → 3.22 | 0.4535 → 0.4241 |
| VitaminC | 599/599 | 67.95 | 0.9400 → 0.7568 | 15.95 → 7.01 | 0.4709 → 0.4299 |
| MASSIVE en-US | 348/348 | 77.87 | 0.9608 → 0.7635 | 10.53 → 7.17 | 0.3330 → 0.3128 |
| MASSIVE de-DE | 348/348 | 59.77 | 1.7153 → 1.3617 | 23.54 → 6.31 | 0.6034 → 0.5208 |
| SQuAD2 | 299/299 | 67.56 | 0.6478 → 0.5964 | 11.51 → 3.91 | 0.4383 → 0.4103 |
| Civil Comments | 300/300 | 82.67 | 0.4226 → 0.4777 | 4.21 → 13.56 | 0.2650 → 0.3011 |
| HelpSteer2 | 249/249 | 34.14 | 1.7176 → 1.4472 | 26.25 → 11.54 | 0.8193 → 0.7377 |
| SummEval relevance | 237/237 | 15.61 | 1.6434 → 1.6140 | 14.19 → 9.38 | 0.8170 → 0.8026 |
| SummEval consistency | 144/144 | 84.72 | 0.7879 → 0.9246 | 13.08 → 34.12 | 0.3175 → 0.4421 |
| PubMedQA | 250/250 | 57.60 | 1.1227 → 0.9588 | 14.80 → 4.76 | 0.5909 → 0.5518 |
| ContractNLI | 1,819/2,074 | 55.09 | 1.0413 → 0.9123 | 18.65 → 7.96 | 0.6131 → 0.5576 |

| 校准后S1：逐套概率指标比较（非显著性检验） | ECE更低 | NLL更低 | Brier更低 |
|---|---:|---:|---:|
| 相对自身T1 | 10/14 | 10/14 | 10/14 |
| 相对Kev-0.8B | 10/14 | 9/14 | 10/14 |
| 相对Laya-base T1 | 10/14 | 10/14 | 9/14 |

| S1校准前→后：自身与共同成功口径（独立精确重算） | 概率n | 等套ACC，不变 (%) | 等套NLL | 等套ECE15 (%) | 等套Brier |
|---|---:|---:|---:|---:|---:|
| 自身成功，5945 attempted，221超长 | 5,724 | 63.9271 | 0.98984789 → 0.88125497 | 14.9645 → 10.3713 | 0.48576851 → 0.46411583 |
| 五模型共同成功，ContractNLI1819 | 5,690 | 63.9684 | 0.98895957 → 0.88079567 | 14.9159 → 10.3314 | 0.48507336 → 0.46371190 |

| S1校准：Score六套等权诊断（有定义项；非hard NLL拟合目标） | T1 | T=1.87397 | 方向 |
|---|---:|---:|---|
| MAE ↓ | 0.807897 | 0.847684 | 退化 |
| RPS ↓ | 0.589401 | 0.545732 | 改善 |

| Dream-7B zero-shot：固定协议 | 值 / 限制 |
|---|---|
| 模型 | Dream-org/Dream-v0-Instruct-7B；revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae；7,615,616,512参数 |
| Zero-shot | 已发布Instruct权重，无本项目训练/学习头/温度拟合；不是没有上游SFT或预训练暴露 |
| Native读出 | 一次完整词表full双向SDPA forward，原始logits读mask位置−1；共享Yes/No，choice/score softmax、noul sigmoid；不生成或迭代去噪 |
| 精度 / 温度 / Context | BF16 batch1，T1，4096上限，不截断；旧generic noul模板，Dream自身chat template |
| Core14 | 排除MultiNLI/Aegis2/PAWS、保留三套JevBench，5945 attempted；不是Dream无污染认证 |
| 覆盖 | 5724成功，221 ContractNLI超长；unsupported原样保留，0源mask拒绝，无静默丢样本 |
| 同分母比较 | 六模型+S1校准条件，共5690成功ID，ContractNLI1819；各套等权，非pooled ECE或官方sealed排名 |
| 验证 | 10条原生对齐最大概率差0，42/42 smoke，完整运行/评分/分析成功；5724条概率独立重建误差约4.44e-16 |
| 发布范围 | 仅zero-shot结果；不含后续Dream-S1微调，Dream推理/训练入口未迁移到公开仓库，默认行为不变 |
| 详细说明 | [Dream zero-shot：逐套、来源、优势/短板及限制](dream_zero_shot.md)；无显著性或纯扩参因果结论 |

| Core14：Dream zero-shot同分母比较（5690共同成功，14套等权） | 条件 | ACC (%) ↑ | ECE15 (%) ↓ | NLL ↓ | Brier(sum) ↓ |
|---|---|---:|---:|---:|---:|
| Original DLLM 0.6B | BF16 zero-shot T1 | 55.40 | 26.97 | 1.3567 | 0.6566 |
| S0 0.6B | BF16 T1 | 61.08 | 12.53 | 0.9461 | 0.4886 |
| S1 0.6B | BF16 T1 | 63.97 | 14.92 | 0.9890 | 0.4851 |
| S1 0.6B，全局校准 | BF16 T1.87397 | 63.97 | 10.33 | 0.8808 | 0.4637 |
| Laya-base | FP32 batch1 aligned_unit T1 | 55.64 | 17.01 | 1.2423 | 0.5923 |
| Kev-0.8B | FP32 batch1 checkpoint T2.35110 | 60.31 | 14.30 | 0.9160 | 0.4921 |
| Dream-7B | BF16 batch1 zero-shot T1 | 64.08 | 17.74 | 0.9186 | 0.4817 |

| Dream zero-shot：自身成功逐套（非共同分母） | 成功 / attempted | ACC (%) ↑ | ECE15 (%) ↓ | NLL ↓ | Brier(sum) ↓ |
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
| ContractNLI | 1853/2074 | 51.11 | 14.06 | 0.9054 | 0.5464 |

| Dream zero-shot：自身成功14套等权（独立精确重算） | 概率n | ACC (%) ↑ | ECE15 (%) ↓ | NLL ↓ | Brier(sum) ↓ |
|---|---:|---:|---:|---:|---:|
| 5945 attempted，221超长保留 | 5724 | 64.0379 | 17.7640 | 0.91876838 | 0.48190208 |

| Dream zero-shot：逐套描述性比较（非显著性检验） | ACC胜/负/平 | ECE更低 | NLL更低 | Brier更低 |
|---|---|---:|---:|---:|
| 相对S1 T1 | 6/6/2 | 5/14 | 8/14 | 6/14 |
| 相对Kev-0.8B | 9/4/1 | 5/14 | 7/14 | 7/14 |
| 相对Laya-base T1 | 9/4/1 | 8/14 | 9/14 | 11/14 |

| 有定义score六套等权诊断 | MAE ↓ | RPS ↓ |
|---|---:|---:|
| S1 T1 | 0.807897 | 0.589401 |
| S1校准 | 0.847684 | 0.545732 |
| Laya-base | 0.884390 | 0.594050 |
| Kev-0.8B | 0.768401 | 0.468446 |
| Dream zero-shot T1 | 0.792387 | 0.524521 |

| Dream-S1：正式训练与选择 | 结果 / 限制 |
|---|---|
| 原模型 | Dream-org/Dream-v0-Instruct-7B；revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae |
| 数据 / 配方 | 原始S1 train40000 / 原始dev1000；3epochs；global32 / micro1；FULL_SHARD BF16 AdamW；CE+0.25normalized ordinal RPS |
| 完成 / 选定 | 完成3750updates / 3epochs；选定2500为第二epoch终点，不是最后权重 |
| 原始dev mean KL | selected2500=0.1459667972254464；last3750=0.20896177570950744 |
| 训练时间 | 296974.386642秒，约82.49小时；含dev/save、硬件特定，不作速度比较 |
| 复载验证 | 两次完整1000dev概率/KL相对selected reference差精确0；identity/shape/target/argmax差0 |
| 独立指标复核 | Core四位小数metrics最大差0，logit概率差4.44e-16；Breadth差≤7.105e-15 |
| 新发布边界 | [Dream-S1代码、CLI与完整说明](dream_s1.md)；原生Dream训练/评测入口现已迁移，修正上方历史“未迁移”说明；本次迁移尚待远程验证 |
| 不发布 | Dream-S1权重、数据、原始标签/文本/预测、机器日志、私有路径或服务器身份 |

| Dream-S1：Core14共同5690成功、14套等权 | ACC (%) | ECE15 (%) | NLL | Brier(sum) |
|---|---:|---:|---:|---:|
| Dream-S1 selected2500 | 70.72 | 17.53 | 0.8753 | 0.4320 |
| Dream zero-shot | 64.08 | 17.74 | 0.9186 | 0.4817 |
| Dream-S1 interim600，未选，不用test选择 | 68.07 | 12.08 | 0.7517 | 0.4080 |
| 0.6B S1 T1 | 63.97 | 14.92 | 0.9890 | 0.4851 |
| 0.6B S1 frozen calibration | 63.97 | 10.33 | 0.8808 | 0.4637 |
| Kev0.8B | 60.31 | 14.30 | 0.9160 | 0.4921 |
| Kev-data0.6B fixed endpoint（聚合背景） | 62.55 | 13.46 | 0.9781 | 0.4986 |

| Dream-S1 Core14：共同成功逐套 | Common N | Final ACC% | Zero ACC% | Final ECE15% | Final NLL | Final Brier |
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

| Dream-S1 Breadth-v1：全14 datasets | Main metric | Final score% | Zero score% | Coverage% | Final ECE10% | Final NLL | Final Brier |
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

| Dream-S1覆盖与概率分母 | Attempted | Successful | Unsupported length | Common n | 说明 |
|---|---:|---:|---:|---:|---|
| Core14 | 5945 | 5724 | 221 | 5690 | 全超长在Contract；all-attempted宏ACC70.20399%，共同成功70.72% |
| Core ContractNLI | 2074 | 1853 | 221 | 1819 | 自身success1242/1853=67.02644%；all-attempted1242/2074=59.88428% |
| Breadth full test | 3089 questions / 1990 records | 3069 questions | 20 | - | 全unsupported计错；SATA/BFCL case-exact，其余12 question accuracy |
| Breadth ContractNLI | 160 | 140 | 20 | - | main68/160=42.50%；answered68/140=48.5714% |

| Dream-S1 Breadth pooled answered / index | Final | Zero | 限制 / 方向 |
|---|---:|---:|---|
| Official index | 36.2081603223 | 31.5401768589 | delta+4.6679834634；95% paired CI[1.3452,7.8761] |
| Final index 95% CI | [33.9340,39.1567] | - | 官方record-bootstrap2000 seed0；非训练seed不确定性 |
| ECE10 (%) | 13.54928 | 5.67067 | 变差 |
| Exact logit NLL | 0.88802267 | 0.81912338 | 变差 |
| Brier(sum) | 0.44269084 | 0.46552760 | 改善 |
| Final mean confidence / answered ACC (%) | 81.1283 / 67.5790 | - | 过度自信，不以index掩盖概率退化 |

| Dream-S1补充诊断与解释边界 | Final | Zero / 限制 |
|---|---|---|
| Pooled ordinal MAE，n841 | 0.72341388 | 0.85703560，改善 |
| Pooled unnormalized RPS，n841 | 0.57144359 | 0.56081700，变差 |
| Pooled soft CE / Brier，n310 | 0.34271238 / 0.07790246 | 0.59631268 / 0.28391100，改善 |
| ECE/NLL定义 | Core15 `(lo,hi]`，NLL floor1e-12 | Breadth10 `[lo,hi)` 最后端点包含，exact logit NLL；不可混合 |
| 暴露 / 因果 | 单seed，公开test已观察，Dream预训练未知 | S1 CLINC/SGD来源重叠未过滤；不作完全未见、普适改善或架构因果结论 |
