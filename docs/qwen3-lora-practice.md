# 基于 Qwen3-0.6B 的 LoRA 微调实践：金融公告事件抽取

前面介绍过 [LoRA 的基本原理](https://shuimqy.art/archives/wei-ming-ming-wen-zhang-buU38wF5)，这次用 Qwen3-0.6B 做一个具体的微调实验：给模型一篇金融公告，让它判断是否发生了企业收购或股份回购，并提取相关字段。

项目地址：[shuimqy/qwen3-lora-financial-events](https://github.com/shuimqy/qwen3-lora-financial-events)。数据集、训练和测试划分、最终 LoRA、基座权重、实验报告，以及本地模型服务都放在仓库中。

这篇主要记录数据怎么处理、训练怎么进行，以及微调前后到底有什么变化。

## 一、先确定模型要做什么

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

字段统一使用字符串数组。原文没有给出金额，就返回 `[]`，不要根据常识补一个。提取的值保留原文写法，也不把“1亿元”自行换算成数字。

还有一种情况不能漏掉：公告中没有这两类事件。这时应该返回：

```json
{"events": []}
```

如果训练数据里全是收购和回购，模型可能会默认每篇文章都需要提取一个事件。因此，训练集里也放了非目标事件。

## 二、数据集怎么选、怎么划分

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

训练和验证来自原始 train，测试来自原始 dev。这里的 800、100、125 是筛选、去重和分层抽样后的固定数量，不是对全部原始数据直接做一个 90/5/5 的划分。

20% 的负样本是这次人为设置的比例，也不代表实际业务中这两类事件出现的频率。

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

基座使用官方的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。这里说的“基座”是还没做本项目微调的原模型，它本身已经经过官方后训练，不是 Qwen3-0.6B-Base。

这次只训练一套 LoRA，模型在一次生成中同时判断事件类型、抽取字段。没有先用一个模型分类，再交给另一个模型抽取。

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

使用 PEFT 加载时，它会根据配置把这些矩阵放到对应层中，推理时自动计算两部分的结果。并不需要手动把两个权重文件相加。

## 四、训练样本和结束符

每条训练序列由三部分组成：

```text
system：事件类型、字段定义、输出格式等要求
user：公告原文
assistant：标准JSON答案
```

模型仍然是在做下一个 Token 的预测。计算 loss 时，提示词和输入原文对应的标签设为 `-100`，只监督助手答案和自然结束符。

这里没有用文本向量的相似度训练。反向传播比较的是预测 Token 的概率和标准答案 Token，而不是先把模型完整回答转成向量。

训练之前，检查了完整序列长度和结束符。序列上限设为 2048 Token，本次 800 条样本中最长的一条是 1156 Token，没有截断答案。结束标记使用官方 chat template 生成的自然 `<|im_end|>`，并参与监督。

这个检查很有必要。只看 JSONL 中的一条记录不长，并不能保证加上 system 提示、原文和完整答案后仍然放得下。如果预处理把答案或结束符截掉，后面就很难判断生成异常到底来自哪里。

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

这里的“使用全部训练集”指的是这 800 条，不是原始数据集的全部 7015 条。训练后按样本 ID 检查了实际参与反向传播的记录，每条都出现 3 次，共 2400 次样本暴露。

正式训练约 594 秒，也就是 9.9 分钟。这个时间只包含正式训练，不包含前面的检查，以及后续验证和测试生成。

每轮结束保存一次，得到 `checkpoint-50`、`checkpoint-100`、`checkpoint-150`。然后分别从磁盘加载，在 100 条验证样本上评估：

| 检查点 | 验证字段 F1 |
| --- | ---: |
| checkpoint-50 | 0.4478 |
| checkpoint-100 | 0.5992 |
| checkpoint-150 | 0.6771 |

按验证字段 F1 选择了 `checkpoint-150`。再独立加载一次，确认生成结果一致，锁定模型后才运行本轮测试。测试集没有参与梯度更新，也没有用来重新选择检查点。

## 六、微调前后怎么比较

比较的是原始 Qwen3-0.6B 和加载最终 LoRA 的 Qwen3-0.6B。

两组使用完全相同的 system/user 提示、125 条测试数据和生成配置：

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

这次没有剥离 Markdown 围栏，也没有修复 JSON 或替换实体名称。输出无效时记一个 FP，并保留应有的 FN；不能因为一条负样本没输出有效内容，就把它算对。

## 七、测试结果，以及这个 0 分该怎么看

| 指标 | 原始模型 | 加载 LoRA |
| --- | ---: | ---: |
| 字段精确率 | 0 | 0.6472 |
| 字段召回率 | 0 | 0.7491 |
| 字段 micro-F1 | 0 | 0.6944 |
| 文档全对率 | 0% | 45.6% |
| 事件类型准确率 | 0% | 84.8% |
| JSON 协议合法率 | 0% | 98.4% |

先解释原始模型为什么是 0。

125 条原始输出都没有通过这套严格协议。有的带了 Markdown 代码围栏，有的把 `arguments` 写成实体列表，没有按字段名组织。

比如测试集第一条公告，原模型确实输出了“领益智造”、被收购公司名称和“2991.54万美元”，但 `arguments` 的形式是：

```json
"arguments": [
  "领益智造",
  "新加坡LITE-ON-MOBILE INDIA PRIVATE LIMITED",
  "2991.54万美元"
]
```

这个片段还包在 Markdown 围栏中。因此，0 分不能理解成它完全不知道该提取什么，也不能把微调后的提升全部解释成学到了金融知识。模型学会按约定组织答案，本身就占了这次变化的一部分。

微调模型的字段 TP、FP、FN 分别是 200、109、67。125 篇中有 57 篇完全正确，2 条输出不符合协议，没有达到生成长度上限的样本，也没有记录到严重重复生成。

对这个规模的模型来说，我认为这个结果已经能说明任务微调是有作用的。不过，打开具体输出后，问题也很明显。

### 1. JSON 对了，字段仍然可能错

测试集第一条的标准被收购方是：

```text
LITE-ON-MOBILE INDIA PRIVATE LIMITED
```

微调后输出的是：

```text
新加坡LITE-ON-MOBILE INDIA PRIVATE LIMITED
```

两者都能在原文中找到，但精确匹配时边界不一样，就会产生一个 FP 和一个 FN。这也是为什么 JSON 合法率已经有 98.4%，文档全对率却只有 45.6%。

### 2. 非目标事件误报比较多

25 条负样本中，只有 10 条正确返回空事件。剩下 13 条被判断成企业收购，2 条被判断成股份回购。

例如，融资新闻里同样会出现公司名称、投资方和金额。模型可能找到了这些实体，但没有分清“融资”和“收购”的事件关系。

接下来我会优先看这部分，而不是直接把 epoch 加大。当前已经训练了 3 轮，再多训练几轮是否有帮助，需要验证，不能只看训练 loss 还在下降。

这 125 条也是项目中沿用的固定测试基准，历史上做过评估和样例查看。这里记录的是当前固定子集上的结果，不把它当作一个全新的外部盲测成绩。

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

适配器配置里保存过原云主机的基座路径，服务加载时会在内存中改成本地路径，原权重文件不需要改。

目前已经写了独立的 `model_service/`，用 uv 管理自己的环境。推理使用 PyTorch、Transformers 和 PEFT，再用 FastAPI 提供 HTTP 接口。业务后端和前端还没有实现，之后会与模型服务分开。

```text
前端 → 业务后端 → 模型服务 → 基座 + LoRA
```

当前接口是 `POST /v1/extract`。它接收公告文本，返回结构化结果、原始输出、协议检查信息和耗时。输出不符合协议时，会明确标记，不会悄悄修成一个空结果。

本地电脑是 i5-14400F、16 GB 内存、GTX 1060 3 GB。现在先用 CPU FP32 部署。检查了收购、回购和非目标文章各一条，三次推理约 8.6～9.2 秒；重启后首条复核约 17.5 秒。这个时间只是少量样例的观察，不是完整压测。

云端评估用的是 GPU BF16、batch4，本地是 CPU FP32、单请求，所以不能直接把云端的评估结果当成本地复测结果。

关于“正常聊天，遇到抽取需求再抽取”，目前还在设计。可以共享同一份基座：聊天时禁用适配器，抽取时启用 LoRA，但还需要处理意图识别和对话上下文，当前服务还没有实现这部分。

## 九、仓库里保留了什么

仓库保留了金融原始数据、800/100/125 的固定划分、完整基座、最终 LoRA、模型服务，以及两份记录：

- [完整实验报告](https://github.com/shuimqy/qwen3-lora-financial-events/blob/main/%E5%AE%8C%E6%95%B4%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md)：数据处理、训练参数、loss、验证和评估说明。
- [测试集评估明细](https://github.com/shuimqy/qwen3-lora-financial-events/blob/main/%E6%B5%8B%E8%AF%95%E9%9B%86%E8%AF%84%E4%BC%B0%E6%98%8E%E7%BB%86.json)：125 条原文、标准答案、两组原始输出和逐条评分。

中间检查点、优化器状态和最初的一次性训练脚本已经在整理时删除，训练配置和关键记录保留在报告中。仓库现在没有一键重跑训练的入口。

模型通过 Git LFS 保存。克隆后如果看到权重文件只有几行文字，需要先执行 `git lfs pull` 下载真实权重，再启动服务。
