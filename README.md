# Qwen3-0.6B 金融事件抽取 LoRA 实践

使用 Qwen3-0.6B 和一套 LoRA，从中文金融公告中识别企业收购、股份回购，并抽取各自的三个核心字段。仓库包含本次使用的数据、冻结划分、原始基座、最终适配器、评估记录和独立模型服务。

## 数据与结果

数据来自 [DuEE-Fin 的 Hugging Face 镜像](https://huggingface.co/datasets/hccngu/Viscacha-Chinese-IE/tree/main/DuEE-fin)，本次只选两类事件的核心字段和非目标事件负样本。

| 集合 | 企业收购 | 股份回购 | 非目标事件 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 | 320 | 320 | 160 | 800 |
| 验证集 | 40 | 40 | 20 | 100 |
| 测试集 | 50 | 50 | 25 | 125 |

全800条训练3轮，共150个优化步，验证集选择checkpoint-150。原始模型与微调模型使用同样的提示词、125条测试和贪心解码配置。

| 指标 | 原始 Qwen3-0.6B | 加载 LoRA |
| --- | ---: | ---: |
| 严格字段 micro-F1 | 0 | 0.6944 |
| 文档完全正确率 | 0% | 45.6% |
| JSON 协议合法率 | 0% | 98.4% |

原模型的0分涉及Markdown围栏、arguments结构等协议失败，不能解释为完全没有抽取知识。微调模型仍有字段边界和负样本误报问题：25条负样本中15条被误识别成目标事件。这是固定子集上的一次实验，不是完整DuEE-Fin官方基准结果。

## 文件位置

```text
data/raw/DuEE-fin/                       原始数据及来源哈希
data/processed/two_events_v1/            train/validation/test.jsonl及分组清单
models/Qwen3-0.6B/                       原始基座和分词器
models/Qwen3-0.6B-financial-LoRA/        最终LoRA权重和配置
model_service/                           uv管理的独立CPU模型服务
完整实验报告.md                          数据处理、训练与评估说明
测试集评估明细.json                      125条测试的原文、参考答案、输出及指标
docs/qwen3-lora-practice.md              本次实践文章
```

模型采用基座加LoRA分别加载。LoRA权重约4.6MB，不能脱离基座单独推理。原训练代码和中间优化器状态未保留，一次性实验的参数、训练loss、验证选择和测试结果已记录在报告与JSON中。

## 获取仓库

模型权重通过Git LFS存储，完整下载约1.5GB以上。先安装[Git LFS](https://git-lfs.com/)，再克隆：

```bash
git lfs install
git clone https://github.com/shuimqy/qwen3-lora-financial-events.git
cd qwen3-lora-financial-events
git lfs pull
```

只查看代码和报告、暂不下载权重时，可设置`GIT_LFS_SKIP_SMUDGE=1`后克隆；模型目录里会先保留LFS指针，推理前仍需`git lfs pull`下载真实文件。

## 本地模型服务

当前实现为Windows CPU FP32推理，服务、业务后端和前端分别组织。本仓库已有模型服务，尚未实现业务后端、前端和自动聊天路由。

安装[uv](https://docs.astral.sh/uv/)后，在PowerShell中执行：

```powershell
cd model_service
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

就绪后访问 http://127.0.0.1:8001/docs，抽取接口为`POST /v1/extract`，请求示例：

```json
{"text":"甲公司公告，拟以1亿元收购乙公司100%股权。"}
```

具体协议、运行方式和本机检查结果见[模型服务说明](model_service/README.md)。环境、缓存、运行日志和真实`.env`配置不提交；`.env.example`仅提供本地路径、端口等示例配置，不含令牌。

## 模型和数据来源

基座来源为[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，原模型许可文件保留在[模型目录](models/Qwen3-0.6B/LICENSE)。本项目使用的是官方已后训练的Qwen3-0.6B，不是Qwen3-0.6B-Base。

数据来源、镜像格式和原始文件SHA-256见[data/raw/DuEE-fin/source_manifest.json](data/raw/DuEE-fin/source_manifest.json)。模型和数据分别遵循各自来源的许可与使用条件，仓库不将镜像代码许可视为原数据许可。
