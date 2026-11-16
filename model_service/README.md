# 金融事件模型服务

使用 Hugging Face Transformers 与 PEFT 加载 Qwen3-0.6B 和 checkpoint-150 LoRA，提供聊天和金融事件抽取接口。模型服务与业务后端、前端分别运行，各自使用独立 uv 环境。

本地部署采用 CPU FP32，默认 6 个计算线程，监听 `127.0.0.1:8001`。PyTorch 使用 CPU 依赖源。模型服务只加载一份基座，不自动使用显卡。

## 运行

在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\model_service\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\model_service\scripts\start.ps1
```

停止命令为 `model_service/scripts/stop.ps1`。在 `model_service/` 中执行 `uv run --frozen model-service` 启动前台服务。服务使用一个 worker，日志写入 `run/`。

`setup.ps1` 安装 Python 3.11，按 `uv.lock` 创建 `.venv/`。Python 运行时与缓存分别保存在 `.python/`、`.cache/`。复制 `.env.example` 为 `.env` 设置端口、线程数、基座和适配器路径；相对模型路径按本服务目录解析。

| 接口 | 用途 |
| --- | --- |
| `/docs` | API 文档 |
| `/health/live` | 进程存活 |
| `/health/ready` | 模型就绪 |
| `/v1/model` | 模型配置、适配器模式与生成锁状态 |
| `POST /v1/chat` | 原始聊天推理 |
| `POST /v1/extract` | 固定协议的金融事件抽取 |

## 聊天推理

```json
{
  "messages": [{"role": "user", "content": "你好！"}],
  "adapter_mode": "base",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.8
}
```

`adapter_mode` 接受 `base` 和 `finetuned`，默认 `finetuned`。返回的 `model` 与 `lora_enabled` 标明实际模式。业务后端的普通聊天和意图分类显式使用 `base`；本接口不判断意图。

基座模式通过 PEFT 的 `disable_adapter()` 临时禁用 LoRA。切换、生成和适配器恢复都在同一把锁内执行，异常结束同样恢复。

回复文本位于 `message.content`。默认 system 提示为“你是一个乐于助人的助手。请根据对话上下文回答用户的问题。”，没有 JSON 限制。模型自行生成的 JSON 或其他文本原样返回，不进行抽取校验。

可选的 system 消息只能位于第一条，之后 user/assistant 交替，以 user 结束。最多 32 条消息，总内容不超过 24000 字符。服务不保存会话，也不删除传入历史。

`max_new_tokens` 为 1–512，默认 256。默认关闭 thinking，采用 temperature=0.7、top_p=0.8、top_k=20；temperature=0 时使用贪心解码。

## 金融事件抽取

```json
{"text": "甲公司公告，拟以1亿元收购乙公司100%股权。"}
```

抽取固定启用 LoRA，使用与实验相同的提示词，关闭 thinking，采用贪心解码，最多生成 256 Token。协议要求每篇至多一个目标事件。

响应示例：

```json
{
  "request_id": "请求标识",
  "status": "ok",
  "valid_json_schema": true,
  "events": [{
    "event_type": "企业收购",
    "arguments": {
      "收购方": ["甲公司"],
      "被收购方": ["乙公司"],
      "交易金额": ["1亿元"]
    }
  }],
  "raw_output": "模型原始输出文本",
  "validation_error": null,
  "warnings": [],
  "usage": {
    "input_tokens": 292,
    "output_tokens": 33,
    "finish_reason": "eos",
    "latency_ms": 5326.17
  }
}
```

示例公告为虚构内容。`events: []` 表示本次模型判断为非目标事件。

| 状态 | 含义 |
| --- | --- |
| `ok` | 通过输出协议校验，保留字段预测 |
| `invalid_output` | JSON 或字段结构校验失败，events 为 null |
| `length_limit` | 生成达到 Token 上限，按截断结果处理 |
| `warnings` 非空 | 字段与原文的检查发现问题，保留预测值并附警告 |

合法 JSON 不等于字段正确。服务不合并旧适配器，不修补 JSON，不改写字段值。

## 资源与输入限制

聊天和抽取共享生成锁，同一时刻只执行一个生成请求。忙碌时返回 503，附 `Retry-After: 1`。健康检查不参与生成锁。

输入与预留输出合计上限为 2048 Token。抽取预留 256 Token，因此包含模板的输入上限为 1792 Token；请求原文字符上限为 12000。超限返回 422，不截断文本。模型未就绪返回 503，推理异常返回 500。

启动时校验基座和适配器 SHA-256，再逐张量核对已加载 LoRA 与磁盘权重。历史云端基座路径仅在内存中覆盖，原始模型配置与权重不修改。推理代码不读取训练集、验证集或测试集。

云端正式评估使用 GPU BF16、batch 4，本服务使用 CPU FP32、单请求。本地部署检查与正式测试指标分别记录。

## 检查

在 `model_service/` 执行：

```powershell
uv run --frozen pytest
uv run --frozen ruff check src tests scripts
```

2026-11-16，42 项自动检查通过，覆盖协议校验、消息顺序、输入预算、接口响应、并发锁和基座模式结束后的适配器恢复。

`scripts/smoke_test.py` 在运行中的服务上检查收购、回购和非目标文章各一条，同时验证忙碌返回 503、健康检查响应和输入超限返回 422。结果写入 `run/smoke_result.json`。该检查只覆盖少量样例，不替代 125 条正式评估。

基座与 LoRA 的身份指代对照见 [chat_identity_comparison.md](docs/chat_identity_comparison.md)。它记录固定提示下的 40 次诊断生成，不作为当前业务后端的聊天评估。
