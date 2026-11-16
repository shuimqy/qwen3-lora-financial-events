# 金融助手业务后端

FastAPI 提供统一聊天入口，LangGraph 编排意图识别、基座聊天、LoRA 抽取和澄清分支。后端通过 HTTP 调用 `model_service/`，不加载权重，不安装 PyTorch。

## 运行

以下命令从项目根目录执行，模型服务先于后端启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\backend\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\backend\scripts\start.ps1
```

API 文档：http://127.0.0.1:8000/docs。停止命令为 `backend/scripts/stop.ps1`。前台运行命令为 `uv run --frozen financial-backend`，执行目录为 `backend/`。

本目录单独保存 `.venv/`、`.python/`、`.cache/` 和 `uv.lock`。配置从环境变量与本目录 `.env` 读取，示例见 `.env.example`。

| 配置项 | 默认值 |
| --- | --- |
| `BACKEND_HOST` | `127.0.0.1` |
| `BACKEND_PORT` | `8000` |
| `BACKEND_MODEL_SERVICE_URL` | `http://127.0.0.1:8001` |
| `BACKEND_MODEL_TIMEOUT_SECONDS` | `120` |

后端部署到服务器时，模型地址必须是该服务器可访问的地址。反向隧道入口配置在 `BACKEND_MODEL_SERVICE_URL`。

## 统一聊天接口

`POST /v1/chat`：

```json
{
  "messages": [{"role": "user", "content": "请提取这篇公告的事件字段。"}],
  "document": "甲公司公告，拟以1亿元收购乙公司100%股权。"
}
```

普通问答只传 `messages`。原文还支持放在当前消息的“公告原文：”之后。请求中的消息从 user 开始，user/assistant 交替，以 user 结束。

| 响应字段 | 内容 |
| --- | --- |
| `message` | 助手回复，包含 `role` 与 `content` |
| `intent` | `CHAT`、`EXTRACT`、`CLARIFY` |
| `source` | `rule`、`llm`、`fallback` |
| `route` | `base_chat`、`lora_extract`、`clarification` |
| `status` | `ok`、`needs_document`、`clarification`、`invalid_output`、`length_limit` |
| `context` | 当前原文与待补原文状态 |
| `extraction` | 抽取状态、结构化字段、原始输出、警告与错误 |
| `usage` | Token 数量、结束原因、推理耗时 |

抽取结果同时保留文本回复和结构化字段。前端直接使用 `extraction.events` 展示卡片。无效输出、输出截断和空事件列表分别处理，不修补预测字段。

`messages` 上限为 24 条、合计 24000 字符；单条消息与 `document` 上限均为 12000 字符。`max_new_tokens` 默认 256、范围 1–512，仅作用于普通聊天。抽取固定为 256 Token。

模型的输入与预留输出合计限制为 2048 Token。超限返回 422，后端不删除历史或截断原文。

## 路由与上下文

```text
当前请求 → 意图识别子图 → 对话分支
                         ├─ CHAT    → 模型 /v1/chat，adapter_mode=base
                         ├─ EXTRACT → 模型 /v1/extract，固定启用 LoRA
                         └─ CLARIFY → 直接返回澄清问题
```

规则优先处理问候、概念解释、自我介绍、明确抽取命令和不支持的抽取。没有抽取候选表达的当前消息默认进入普通聊天。存在抽取、金融字段或材料指代，但规则未确定意图时，调用基座分类。

分类请求采用贪心解码，最多生成 8 Token，接受 `CHAT`、`EXTRACT`、`CLARIFY` 三个完整标签。标签无效或输出未正常结束时返回澄清。分类没有置信分数，也没有独立训练的意图模型。

分类器仅接收最近四条历史摘要、当前消息摘要和材料摘要。当前消息摘要上限为 500 字符，长文中间的指令不在完整解析范围内。

`context` 由调用方保存并在下一轮传回，后端不保存会话或写入数据库：

```json
{"document": "当前公告原文", "awaiting_document": false}
```

查找原文时，当前消息优先，其次为显式 `document`、上下文原文与历史用户材料。助手回复不作为原文。

抽取请求缺少原文时返回 `needs_document`，并设置 `awaiting_document=true`。下一轮补充公告后直接执行抽取。首次只发送公告时先澄清，回复“抽取”后执行。新对话清空历史与 `context`。

## 当前边界

当前上下文只保存原文与待补状态，未保存上次结构化结果用于字段追问。“交易金额是多少”等追问仍会触发完整抽取，返回整套字段。字段追问统一进入普通聊天的方案尚未实现。

开发测试复现了历史错误抽取回复干扰基座聊天：路由已经转为 `base_chat`，回复仍重复此前字段。规则修正解决了普通新话题进入抽取分支的问题，未解决全部历史干扰。当前验收范围为独立聊天与单次公告抽取。

## 错误与健康检查

| HTTP 状态 | 含义 |
| --- | --- |
| 422 | 请求无效或模型输入过长 |
| 502 | 上游响应格式错误，或未按要求使用基座 |
| 503 | 模型繁忙、未就绪、连接失败或超时 |

失败响应使用 `error` 结构，不作为助手消息加入历史。后端不自动重试。

`/health/live` 检查后端进程；`/health/ready` 检查模型服务就绪且支持基座模式。`POST /v1/intent` 保留为意图调试接口，接收 `messages` 和可选 `document`，不执行聊天或抽取。

## 检查

在 `backend/` 执行：

```powershell
uv run --frozen pytest
uv run --frozen ruff check src tests scripts
uv run --frozen python scripts/check_intent.py
```

前两项不加载模型。最后一项调用运行中的服务，使用人工开发样例验证意图分支，结果写入 `run/intent_check.json`。这些样例参与了规则与提示调试，不作为独立分类准确率。

2026-11-16，90 项自动检查通过。覆盖历史校验、分支选择、缺原文后补充、当前材料替换、普通话题切换、无效/空/截断输出和上游错误。自动检查使用替身模型与 HTTP 响应，不计为真实模型的回答正确率。
