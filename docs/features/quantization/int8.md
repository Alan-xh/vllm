---
title: INT8 W8A8
---
[](){ #int8 }

vLLM 支持将权重和激活量化为 INT8，以节省内存并加速推理。
这种量化方法特别适用于在保持良好性能的同时减小模型大小。

请访问 HF 集合，获取 [适用于 vLLM 的流行 LLM 的量化 INT8 检查点](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415)。

!!! note
    INT8 计算在计算能力 > 7.5 的 NVIDIA GPU 上受支持（Turing、Ampere、Ada Lovelace、Hopper、Blackwell）。

## 前提条件

要在 vLLM 中使用 INT8 量化，您需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```console
pip install llmcompressor
```

此外，安装 `vllm` 和 `lm-evaluation-harness` 用于评估：

```console
pip install vllm lm-eval==0.4.4
```

## 量化过程

量化过程包括四个主要步骤：

1. 加载模型
2. 准备校准数据
3. 应用量化
4. 在 vLLM 中评估准确性

### 1. 加载模型

使用标准的 `transformers` AutoModel 类加载模型和分词器：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. 准备校准数据

在将激活量化为 INT8 时，需要样本数据来估计激活尺度。
最好使用与部署数据密切匹配的校准数据。
对于通用指令调优模型，可以使用像 `ultrachat` 这样的数据集：

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# 加载并预处理数据集
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 3. 应用量化

现在，应用量化算法：

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# 配置量化算法
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# 应用量化
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 保存压缩模型：Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

此过程创建一个权重和激活量化为 8 位整数的 W8A8 模型。

### 4. 评估准确性

量化后，您可以在 vLLM 中加载并运行模型：

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

要评估准确性，可以使用 `lm_eval`：

```console
$ lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量化模型可能对 `bos` 标记的存在敏感。确保在运行评估时包含 `add_bos_token=True` 参数。

## 最佳实践

- 从 512 个校准数据样本开始（如果准确性下降则增加）
- 以 2048 的序列长度为起点
- 使用模型训练时使用的聊天模板或指令模板
- 如果您对模型进行了微调，考虑使用您的训练数据样本进行校准

## 故障排除和支持

如果您遇到任何问题或有功能请求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 存储库上提交问题。