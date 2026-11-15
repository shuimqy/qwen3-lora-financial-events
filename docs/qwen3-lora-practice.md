# 基于 Qwen3-0.6B 的 LoRA 微调实践：金融公告事件抽取

前面介绍过 [LoRA 的基本原理](https://shuimqy.art/archives/wei-ming-ming-wen-zhang-buU38wF5)，这次用 Qwen3-0.6B 做一个具体的微调实验：给模型一篇金融公告，让它判断是否发生了企业收购或股份回购，并提取相关字段。

项目地址：[shuimqy/qwen3-lora-financial-events](https://github.com/shuimqy/qwen3-lora-financial-events)。数据集、训练和测试划分、最终 LoRA、基座权重、实验报告，以及本地模型服务都放在仓库中。

## 一、任务范围与输出格式

最初考虑的是中文通用问答。不过，开放式回答很难只用一个自动指标判断好坏。同一个问题可以有很多种说法，回答相似也不一定意味着事实正确。

后来把任务收窄到金融公告事件抽取。输入原文，输出事件类型和字段，结果比较容易检查，也方便后面接一个页面展示。

第一版只选两种事件：

| 事件类型 | 提取字段 |
| --- | --- |
| 企业收购 | 收购方、被收购方、交易金额 |
| 股份回购 | 回购方、回购股份数量、交易金额 |

例如，下面是一条用于说明输出格式的简化公告：

```text
甲公司公告，拟以1亿元收购乙公司100%股权。
```

期望输出：

```json
{
  "events": [
    {
      "event_type": "企业收购",
      "arguments": {
        "收购方": ["甲公司"],
        "被收购方": ["乙公司"],
        "交易金额": ["1亿元"]
      }
    }
  ]
}
```

字段统一使用字符串数组，缺失值记为 `[]`。实体名称、金额和数量沿用原文写法，例如“1亿元”直接作为字符串保存。

不包含这两类事件的公告，对应的输出为空数组：

```json
{"events": []}
```

如果训练数据里全是收购和回购，模型可能会默认每篇文章都需要提取一个事件。因此，训练集里也放了非目标事件。

## 二、数据筛选与数据集划分

使用的数据是 DuEE-Fin，下载自 [Hugging Face 上的镜像目录](https://huggingface.co/datasets/hccngu/Viscacha-Chinese-IE/tree/main/DuEE-fin)。这个镜像中的数据是 JSON 数组，包含公告原文和事件标注。

原始 train 有 7015 条，dev 有 1171 条。这次没有直接把全部数据拿来训练，而是先筛出适合当前任务的样本：

- 正样本只保留原标注恰好有一个事件的完整短文档。
- 只使用两类事件的三个核心字段。
- 标注值必须能在原文中连续找到，不能靠拼接或改写得到。
- 排除过长、标注无法对应原文等记录。
- 非目标事件作为负样本。

划分之前还做了文档去重。除了完全相同的文本，也按字符三元组的 Jaccard 相似度处理近重复文档，阈值设为 0.85。同一组文档不跨训练、验证和测试集。

最终划分如下：

| 集合 | 企业收购 | 股份回购 | 非目标事件 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 | 320 | 320 | 160 | 800 |
| 验证集 | 40 | 40 | 20 | 100 |
| 测试集 | 50 | 50 | 25 | 125 |

训练和验证来自原始 train，测试来自原始 dev。筛选和去重后按类别分层抽样，各集合的正负样本比例统一为 4:1。

仓库中的对应文件是：

```text
data/processed/two_events_v1/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── manifest.json
├── groups.json
└── excluded.json
```

前三个是训练、验证、测试数据，后面几个记录划分、去重和排除信息。

## 三、Qwen3-0.6B 和 LoRA 配置

基座使用官方已完成后训练的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，在原有指令能力上继续进行任务微调。

事件识别和字段抽取使用同一套 LoRA，在一次生成中完成。

LoRA 配置如下：

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
```

可训练参数为 1,146,880 个，最终保存的 LoRA 权重约 4.6 MB。

LoRA 参数是单独保存的。对于原来计算 `y = Wx` 的线性层，加入 LoRA 后会多出一个分支：

```text
y = Wx + (alpha / r) × B(Ax)
```

基座的 W 保持不变，训练更新的是小矩阵 A 和 B。本次 `alpha / r = 16 / 8 = 2`。

PEFT 根据配置将 A、B 挂载到对应线性层，推理时计算原权重分支与 LoRA 分支的输出，再将两者相加。

## 四、训练样本和结束符

每条训练序列由三部分组成：

```text
system：事件类型、字段定义、输出格式等要求
user：公告原文
assistant：标准JSON答案
```

训练采用自回归语言模型的交叉熵损失。提示词和输入原文对应的标签设为 `-100`，只对助手答案和自然结束符计算 loss，再通过反向传播更新 LoRA 参数。

训练之前，检查了完整序列长度和结束符。序列上限设为 2048 Token，本次 800 条样本中最长的一条是 1156 Token，没有截断答案。结束标记使用官方 chat template 生成的自然 `<|im_end|>`，并参与监督。

长度统计以套用 chat template 后的完整 Token 序列为准，包含 system 提示、公告原文、标准答案和结束符。

## 五、训练过程与验证选模

训练使用一张 RTX 3080 10 GB，主要配置如下：

| 配置 | 值 |
| --- | --- |
| 训练数据 | 全部 800 条，包含 160 条负样本 |
| Epoch | 3 |
| 学习率 | 2e-5 |
| 调度方式 | constant_with_warmup |
| Warmup | 3 步 |
| 优化器 | AdamW |
| Weight decay | 0.01 |
| Micro batch | 1 |
| 梯度累积 | 16 |
| 训练精度 | BF16 |
| 梯度检查点 | 开启 |
| 随机种子 | 42 |

有效 batch 为 16。800 条数据训练一轮对应 50 个优化步，3 轮一共 150 步。

训练记录按样本 ID 保存，每条训练样本都参与了 3 次反向传播，共 2400 次。

正式训练耗时约 594 秒，即 9.9 分钟，验证和测试另行执行。

每轮结束保存一次，得到 `checkpoint-50`、`checkpoint-100`、`checkpoint-150`。然后分别从磁盘加载，在 100 条验证样本上评估：

| 检查点 | 验证字段 F1 |
| --- | ---: |
| checkpoint-50 | 0.4478 |
| checkpoint-100 | 0.5992 |
| checkpoint-150 | 0.6771 |

最终按验证字段 F1 选择 `checkpoint-150`，独立重载后生成结果一致，随后用这个检查点运行测试。

## 六、微调前后的评估方法

比较的是原始 Qwen3-0.6B 和加载最终 LoRA 的 Qwen3-0.6B。

评估沿用项目已有的 125 条固定测试划分。两组使用完全相同的 system/user 提示和生成配置：

```text
enable_thinking = False
do_sample = False
num_beams = 1
max_new_tokens = 256
batch_size = 4
```

推理时只给模型原文，不给真实事件类型或参考字段。

字段评估采用精确匹配，把结果展开成：

```text
(事件类型, 字段名, 字段值)
```

例如：

```text
(企业收购, 收购方, 甲公司)
(企业收购, 被收购方, 乙公司)
(企业收购, 交易金额, 1亿元)
```

累计整个测试集的 TP、FP、FN 后计算 micro-F1。另外记录文档全对率和 JSON 协议合法率。

“文档全对”要求这一篇的事件类型和所有字段都正确。对于负样本，只有正确返回空事件才算全对。

“JSON 合法”也不只是语法正确，还要符合约定结构。例如，`arguments` 必须是按角色命名的对象，各字段值必须是字符串数组。

评分直接读取模型原始输出，保留 Markdown 围栏、JSON 结构和实体名称。协议无效的输出记一个 FP，同时累计未匹配参考字段的 FN；负样本输出无效也记错。

## 七、测试结果与零分原因分析

| 指标 | 原始模型 | 加载 LoRA |
| --- | ---: | ---: |
| 字段精确率 | 0 | 0.6472 |
| 字段召回率 | 0 | 0.7491 |
| 字段 micro-F1 | 0 | 0.6944 |
| 文档全对率 | 0% | 45.6% |
| 事件类型准确率 | 0% | 84.8% |
| JSON 协议合法率 | 0% | 98.4% |

原始模型的 0 分主要来自输出格式。125 条输出均未通过严格协议，常见问题是带有 Markdown 代码围栏，或者将 `arguments` 写成实体列表。

比如测试集第一条公告，原模型确实输出了“领益智造”、被收购公司名称和“2991.54万美元”，但 `arguments` 的形式是：

```json
"arguments": [
  "领益智造",
  "新加坡LITE-ON-MOBILE INDIA PRIVATE LIMITED",
  "2991.54万美元"
]
```

这个片段还包在 Markdown 围栏中。原模型已经找出了部分实体和金额，但尚未按字段组织结果。这一轮微调的收益同时包含输出格式和抽取能力的变化。

微调模型的字段 TP、FP、FN 分别是 200、109、67。125 篇中有 57 篇完全正确，2 条输出不符合协议，没有达到生成长度上限的样本，也没有记录到严重重复生成。

逐条查看输出后，错误主要集中在字段边界和非目标事件识别。

### 1. 字段边界错误

测试集第一条的标准被收购方是：

```text
LITE-ON-MOBILE INDIA PRIVATE LIMITED
```

微调后输出的是：

```text
新加坡LITE-ON-MOBILE INDIA PRIVATE LIMITED
```

两者都能在原文中找到，但实体边界不一致，精确匹配时产生一个 FP 和一个 FN。这条样本的 JSON 结构合法，被收购方字段却匹配失败。

### 2. 非目标事件误报

25 条负样本中，只有 10 条正确返回空事件。剩下 13 条被判断成企业收购，2 条被判断成股份回购。

例如，融资新闻里同样会出现公司名称、投资方和金额。模型可能找到了这些实体，但没有分清“融资”和“收购”的事件关系。

负样本误报占到了 60%，说明模型对几类相近金融事件的区分还比较弱。后续的数据整理会重点关注融资、收购和股东交易之间的边界。

## 八、本地加载与部署

最终保存的是两部分：

```text
models/
├── Qwen3-0.6B/
└── Qwen3-0.6B-financial-LoRA/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

使用 Transformers 加载基座，再用 PEFT 加载 LoRA。下面是简化的 CPU 加载代码：

```python
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "models/Qwen3-0.6B"
adapter_path = "models/Qwen3-0.6B-financial-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    base_path,
    local_files_only=True,
    torch_dtype=torch.float32,
    attn_implementation="sdpa",
)
config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
config.base_model_name_or_path = base_path
model = PeftModel.from_pretrained(
    base, adapter_path, config=config, local_files_only=True
)
model.eval()
```

加载时，将适配器配置中的 `base_model_name_or_path` 指向当前基座目录。

推理模块放在独立的 `model_service/` 目录，用 uv 管理环境。模型计算使用 PyTorch，加载和生成使用 Transformers、PEFT，HTTP 接口使用 FastAPI。

```text
HTTP 请求 → FastAPI → Transformers + PEFT → 基座 + LoRA
```

`POST /v1/extract` 接收公告文本，响应包含抽取字段、原始输出、协议校验状态和推理耗时。

本地使用 i5-14400F 和 16 GB 内存，以 CPU FP32 运行。收购、回购和非目标文章三条样例的处理时间约为 8.6～9.2 秒，重启后的首条请求约为 17.5 秒。

## 九、项目文件与实验记录

数据、模型和服务代码放在[项目仓库](https://github.com/shuimqy/qwen3-lora-financial-events)，详细实验记录包括：

- [完整实验报告](https://github.com/shuimqy/qwen3-lora-financial-events/blob/main/%E5%AE%8C%E6%95%B4%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md)：数据处理、训练参数、loss、验证和评估说明。
- [测试集评估明细](https://github.com/shuimqy/qwen3-lora-financial-events/blob/main/%E6%B5%8B%E8%AF%95%E9%9B%86%E8%AF%84%E4%BC%B0%E6%98%8E%E7%BB%86.json)：125 条原文、标准答案、两组原始输出和逐条评分。
