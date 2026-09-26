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
