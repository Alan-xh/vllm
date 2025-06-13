---
title: INT4 W4A16
---
[](){ #int4 }

vLLM 支持将权重量化为 INT4，以节省内存并加速推理。这种量化方法特别适用于减少模型大小并在低每秒查询数（QPS）的工作负载中保持低延迟。

请访问 Hugging Face 上的 [适用于 vLLM 的流行 LLM 的 INT4 量化检查点集合](https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c)。

!!! note
    INT4 计算在 NVIDIA GPU 上支持，需计算能力 > 8.0（Ampere、Ada Lovelace、Hopper、Blackwell）。

## 前置条件

要在 vLLM 中使用 INT4 量化，您需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```console
pip install llmcompressor
```

此外，安装 `vllm` 和 `lm-evaluation-harness` 以进行评估：

```console
pip install vllm lm-eval==0.4.4
```

## 量化过程

量化过程包括以下四个主要步骤：

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

在将权重量化为 INT4 时，需要样本数据来估计权重更新和校准尺度。最好使用与部署数据密切匹配的校准数据。对于通用指令调整模型，可以使用像 `ultrachat` 这样的数据集：

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# 加载和预处理数据集
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
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# 应用量化
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 保存压缩模型：Meta-Llama-3-8B-Instruct-W4A16-G128
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

此过程创建了一个权重量化为 4 位整数的 W4A16 模型。

### 4. 评估准确性

量化后，您可以在 vLLM 中加载并运行模型：

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
```

要评估准确性，可以使用 `lm_eval`：

```console
$ lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量化模型对 `bos` 标记的存在可能很敏感。运行评估时，请确保包含 `add_bos_token=True` 参数。

## 最佳实践

- 从 512 个校准样本开始，如果准确性下降则增加样本数量
- 确保校准数据包含多样化的样本，以防止对特定用例的过拟合
- 使用 2048 的序列长度作为起点
- 使用模型训练时使用的聊天模板或指令模板
- 如果您对模型进行了微调，可以考虑使用训练数据的样本进行校准
- 调整量化算法的关键超参数：
  - `dampening_frac` 设置 GPTQ 算法的影响程度。较低的值可以提高准确性，但可能导致数值不稳定性，从而导致算法失败。
  - `actorder` 设置激活顺序。在压缩层权重时，量化通道的顺序很重要。设置 `actorder="weight"` 可以在不增加延迟的情况下提高准确性。

以下是一个扩展的量化配方示例，您可以根据自己的用例进行调整：

```python
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
) 
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
                symmetric=True,
                dynamic=False,
                actorder="weight",
            ),
        ),
    },
    ignore=["lm_head"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=0.01
)
```

## 故障排除和支持

如果您遇到任何问题或有功能请求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 仓库上提交 issue。`llm-compressor` 中的完整 INT4 量化示例可在 [此处](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_example.py) 找到。