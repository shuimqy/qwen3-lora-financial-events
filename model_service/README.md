# 金融事件模型推理服务

独立运行的模型服务，使用uv维护自己的Python环境、依赖声明和锁文件。它从上级`models/`加载Qwen3-0.6B基座和checkpoint-150单LoRA，供业务后端通过HTTP调用；业务后端和前端不在此目录实现。

当前版本为Windows本机CPU部署，采用FP32，默认6个计算线程。没有自动占用GTX1060 3GB显卡。依赖明确使用PyTorch CPU源，避免下载CUDA运行库。默认只监听`127.0.0.1:8001`。

## 安装与启动

在PowerShell执行：

```powershell
cd D:\Projects\LLM_SFT\model_service
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

`setup.ps1`使用uv安装Python3.11，并按`uv.lock`创建`.venv/`。Python运行时和下载缓存也放在本目录的`.python/`、`.cache/`，不与后续业务后端共用环境。模型权重已在本地，不会重新下载。

`start.ps1`在后台启动一个模型进程。模型加载一次，日志在`run/stdout.log`和`run/stderr.log`。加载完成后：

- API文档：http://127.0.0.1:8001/docs
- 存活检查：http://127.0.0.1:8001/health/live
- 就绪检查：http://127.0.0.1:8001/health/ready
- 模型配置：http://127.0.0.1:8001/v1/model

停止：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop.ps1
```

需要在终端查看日志时，也可以在本目录前台运行`uv run --frozen model-service`，按Ctrl+C停止。只启动一个worker，避免重复加载模型。启动前可将`.env.example`复制为`.env`修改端口、线程数或模型路径；相对模型路径固定相对本服务目录解析。

## 推理接口

`POST /v1/extract`只接收原文：

```json
{"text":"甲公司公告，拟以1亿元收购乙公司100%股权。"}
```

PowerShell调用示例：

```powershell
$body = @{ text = '甲公司公告，拟以1亿元收购乙公司100%股权。' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/v1/extract `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

成功响应结构如下，仅作字段示例，具体抽取值由模型生成：

```json
{
  "request_id": "一次请求的标识",
  "status": "ok",
  "valid_json_schema": true,
  "events": [{
    "event_type": "企业收购",
    "arguments": {"收购方": ["甲公司"], "被收购方": ["乙公司"], "交易金额": ["1亿元"]}
  }],
  "raw_output": "模型原始输出文本",
  "validation_error": null,
  "warnings": [],
  "usage": {"input_tokens": 300, "output_tokens": 70, "finish_reason": "eos", "latency_ms": 1000.0}
}
```

`events: []`表示模型判断为非目标事件。业务后端应检查`status`和`valid_json_schema`，不能只根据HTTP200认定抽取成功：

| 状态或HTTP码 | 含义 |
| --- | --- |
| `status=ok` | 输出符合约定协议，events可用于后续展示 |
| `status=invalid_output` | 输出违反JSON协议；events为null，保留raw_output及错误原因 |
| `status=length_limit` | 生成达到256 Token上限，可能不完整，应标记或人工复核 |
| `warnings`非空 | 例如字段值未在原文出现；保留原始预测，不悄悄删除或修复 |
| HTTP422 | 请求无效，或含提示词的输入超过1792 Token；不截断原文 |
| HTTP503 / `model_busy` | 当前正处理另一请求；附Retry-After:1，业务后端可稍后重试 |
| HTTP503 / `model_not_ready` | 模型尚未就绪 |
| HTTP500 | 推理执行异常，按request_id查询服务日志 |

## 与实验设置的关系

system/user提示与最终实验一致，非thinking、贪心解码、最多256个新Token。服务预留生成预算，将输入限制为1792 Token，使总长度不超过2048；请求最多12000个字符，先做基础校验，再按实际tokenizer计数。

启动时核对基座和适配器SHA-256，并逐张量检查加载的LoRA等于磁盘权重。`src/model_service/task.json`只包含固定提示、模型身份和生成上限，不包含测试答案。推理不读取训练集、验证集或测试集。

实验正式评估使用GPU BF16、batch4；本机服务使用CPU FP32、单请求。部署后的输出和延迟要以本机实测为准，不能直接把云端评估值视为本机复测结果。服务不修复JSON、不合并旧LoRA、不使用真实事件标签提示。

## 文件组织与检查

```text
model_service/
├── pyproject.toml / uv.lock       # 独立依赖与锁定版本
├── .python-version / .env.example
├── src/model_service/            # API、模型加载、协议校验与固定任务
├── tests/                        # 协议、输入限制、接口与并发保护检查
├── scripts/                      # 安装、后台启动、停止
└── run/                          # 运行日志与本机检查记录，自动生成
```

开发检查：

```powershell
uv run --frozen pytest
uv run --frozen ruff check .
```

## 本机验收记录

2026-11-15在本机i5-14400F、16GB内存环境完成：

- 23项自动检查及Ruff静态检查通过。
- 真实HTTP检查收购、回购、非目标文章各一条，均返回合法协议；并发请求返回503，推理期间健康检查可响应，超长输入返回422。
- 初次三条请求的模型处理耗时约8.6–9.2秒；重启后首条复核约17.5秒，原始输出与重启前一致。仅为本机少量样例观察，不是完整性能压测。
- 回购样例与参考字段一致；收购样例存在实体边界差异，非目标样例误报收购。协议合法不等于抽取内容正确。

本机具体响应及耗时保存在`run/smoke_result.json`，运行文件不提交Git。可在服务启动后运行`uv run --frozen python scripts/smoke_test.py`重新生成。该检查只运行少量样例，没有重跑125条基准、训练模型或改写原实验报告。
