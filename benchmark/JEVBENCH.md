# Native probability HTTP

该服务为继承的 DLLM evaluator 添加轻量 TypeSafe 传输层，协议标记 `jevbench::v1.4`。模型、renderer 和 shared Yes/No 读出不因 HTTP 而更换；`T=1`，不拟合温度，不从生成文本推算概率。

## 启动

按[根 README](../README.md)安装并下载模型，然后在仓库根目录执行：

```bash
python benchmark/serve.py \
  --model-path "$CHECKPOINT" --model-name Shared-YesNo-Qwen3-0.6B-S1 \
  --device cuda:0 --dtype bfloat16 --max-length 4096 --port 8000
```

仅绑定 `127.0.0.1`，标准库 `HTTPServer` 串行处理，不依赖 FastAPI。`GET /health` 返回 ready、实际模型名和 runtime。启动加载期间没有 ready 响应；看到 ready 再请求。它不是有认证、TLS、并发调度或公网防护的生产服务。

## 请求

`POST /v1/systemone` 使用 `Content-Type: application/json`。顶层必须恰好是 `state`、`model`、`questions`；`state` 可以是字符串、对象或数组。模型名必须与启动参数一致，也接受 `jev-latest` 别名，响应始终报告实际名称。

下面是协议示例，不是 benchmark 原始题：

```json
{
  "state": "The delivery arrived two days late.",
  "model": "Shared-YesNo-Qwen3-0.6B-S1",
  "questions": {
    "sentiment": {
      "type": "choice",
      "instructions": "Classify the customer sentiment.",
      "criteria": {
        "Positive": "The customer is satisfied.",
        "Negative": "The customer is dissatisfied."
      }
    },
    "late": {
      "type": "noul",
      "instructions": "Did the delivery arrive late?",
      "criteria": null
    },
    "severity": {
      "type": "score",
      "instructions": "Rate the delivery problem.",
      "criteria": ["No problem", "Minor problem", "Major problem"]
    }
  }
}
```

每题含 `type`、`instructions`、`criteria`；`noul` 可省略 `criteria`（补为 null）。`choice` 至少两个候选，示例用原始标签为键的对象；`score` 至少两个有序等级。继承实现的 noul prompt 不使用 criteria 描述，重要判断定义应写入 instructions。

## 返回的三种答案

以下数字仅解释结构，不是模型实测输出：

```json
{
  "sentiment": {
    "type": "choice",
    "choice": "Negative",
    "probabilities": {"Positive": 0.2, "Negative": 0.8}
  },
  "late": {"type": "noul", "noul": 0.9},
  "severity": {
    "type": "score",
    "score": 1.2,
    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}
  }
}
```

上述对象位于响应的 `answers` 中，键与请求问题名对应。响应还含 `model`、`usage`、`runtime`。

- `choice`：保留候选原始字符串和插入顺序，`choice` 为 native first-argmax 标签。
- `noul`：标量是 `P(true)`，不是 JSON 布尔值；`P(false)=1-P(true)`。
- `score`：分布键为零起始索引字符串，不是等级描述；标量为期望索引 `sum(i * p_i)`，不是四舍五入类别或任意外部评分尺度。
- 概率直接映射 native 输出，不在 HTTP 层舍入、重新校准或重归一化。`usage.input_tokens` 累计各独立问题的输入长度，`output_tokens=0`，不是总计算量或成本估算。

多个问题共享同一个 state，但逐题独立、串行推理。不能据此声称联合概率建模、多步推理或跨题条件推断。

## 拒绝与验证

非法 schema/model 返回 400；超过请求体上限返回 413；`unsupported_length` 或 `unsupported_source_mask` 返回 422；意外推理错误返回 500。单请求内任一题失败时不返回部分答案。默认请求体上限 16 MiB，长度上限按模型 tokenizer 和实际渲染结果计算，不截断输入。

[官方公开集流程](../docs/jevbench_public.md)逐题比较直接调用和 HTTP 映射，再启动全新服务运行官方 CLI。官方 argmax 使用自己的平局规则；与 native first-argmax 的精确平局差异须记录，不通过改概率掩盖。
