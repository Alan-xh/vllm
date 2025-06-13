---
title: FP8 W8A8
---
[](){ #fp8 }

vLLM 支持使用 GPU（如 Nvidia H100 和 AMD MI300x）上的硬件加速进行 FP8（8 位浮点）权重和激活量化。
目前，仅 Hopper 和 Ada Lovelace GPU 官方支持 W8A8。
Ampere GPU 支持 W8A16（仅权重 FP8），使用 Marlin 内核。
使用 FP8 量化模型可以将模型内存需求减少 2 倍，并将吞吐量提高高达 1.6 倍，同时对精度的影响极小。

请访问 HF 上的 [适用于 vLLM 的流行 LLM 的量化 FP8 检查点集合](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127)。

硬件中通常支持的 FP8 类型有两种不同的表示方式，分别适用于不同场景：

- **E4M3**：由 1 位符号位、4 位指数位和 3 位尾数组成。可以存储高达 +/-448 的值以及 `nan`。
- **E5M2**：由 1 位符号位、5 位指数位和 2 位尾数组成。可以存储高达 +/-57344 的值、+/- `inf` 和 `nan`。增加动态范围的代价是存储值的精度较低。

!!! note
    FP8 计算在 NVIDIA GPU 上支持，需计算能力 > 8.9（Ada Lovelace、Hopper）。
    FP8 模型可在计算能力 > 8.0（Ampere）的 GPU 上以仅权重 W8A16 运行，使用 FP8 Marlin。

## 安装

要生成高性能的 FP8 量化模型与 vLLM 一起使用，您需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```console
pip install llmcompressor
```

## 量化过程

量化过程包括三个主要步骤：

1. 加载模型
2. 应用量化
3. 在 vLLM 中评估精度

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

### 2. 应用量化

对于 FP8 量化，我们可以使用简单的 RTN 量化来恢复精度。我们建议使用 `FP8_DYNAMIC` 方案针对所有 `Linear` 层，该方案使用：

- 权重上的静态、按通道量化
- 激活上的动态、按令牌量化

由于简单的 RTN 不需要数据进行权重量化，且激活是动态量化的，因此此量化流程不需要任何校准数据。

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 配置简单的 PTQ 量化
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# 应用量化算法
oneshot(model=model, recipe=recipe)

# 保存模型：Meta-Llama-3-8B-Instruct-FP8-Dynamic
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3. 评估精度

安装 `vllm` 和 `lm-evaluation-harness` 进行评估：

```console
pip install vllm lm-eval==0.4.4
```

在 `vllm` 中加载并运行模型：

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = model.generate("Hello my name is")
print(result[0].outputs[0].text)
```

使用 `lm_eval` 评估精度（例如在 `gsm8k` 的 250 个样本上）：

!!! note
    量化模型可能对 `bos` 令牌的存在敏感。`lm_eval` 默认不添加 `bos` 令牌，因此在运行评估时请确保包含 `add_bos_token=True` 参数。

```console
$ MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
$ lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

以下是结果分数的示例：

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## 故障排除和支持

如果您遇到任何问题或有功能请求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 仓库上提交问题。

## 在线动态量化

无需任何校准数据即可使用 vLLM 将原始精度 BF16/FP16 模型动态量化为 FP8。您可以通过在命令行中指定 `--quantization="fp8"` 或在 LLM 构造函数中设置 `quantization="fp8"` 来启用此功能。

在此模式下，所有 Linear 模块（除最终的 `lm_head` 外）的权重都将量化为 FP8_E4M3 精度，并使用每张量尺度。激活在每次前向传播期间计算其最小值和最大值，以提供高精度的动态每张量尺度。因此，此模式下的延迟改进有限。

```python
from vllm import LLM
model = LLM("facebook/opt-125m", quantization="fp8")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = model.generate("Hello, my name is")
print(result[0].outputs[0].text)
```

!!! warning
    当前，我们会先以原始精度加载模型，然后再量化为 8 位，因此您需要足够的内存来加载整个模型。