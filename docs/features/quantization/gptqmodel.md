---
title: GPTQModel
---
[](){ #gptqmodel }

要创建新的4位或8位GPTQ量化模型，您可以利用来自ModelCloud.AI的[GPTQModel](https://github.com/ModelCloud/GPTQModel)。

量化将模型的精度从BF16/FP16（16位）降低到INT4（4位）或INT8（8位），这显著减少了模型的总内存占用，同时提高了推理性能。

兼容的GPTQModel量化模型可以利用`Marlin`和`Machete` vLLM自定义内核，最大化Ampere（A100+）和Hopper（H100+）Nvidia GPU的批量处理事务每秒（`tps`）和令牌延迟性能。这两个内核由vLLM和NeuralMagic（现为Redhat的一部分）高度优化，以实现量化的GPTQ模型的世界级推理性能。

GPTQModel是全球少数支持`动态`每模块量化的工具包之一，允许对大型语言模型的不同层和/或模块使用自定义量化参数进行进一步优化。`动态`量化已完全集成到vLLM中，并得到ModelCloud.AI团队的支持。请参阅[GPTQModel readme](https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override)以获取更多详细信息和其他高级功能。

## 安装

您可以通过安装[GPTQModel](https://github.com/ModelCloud/GPTQModel)或从[Huggingface上的5000多个模型](https://huggingface.co/models?search=gptq)中选择一个来量化您自己的模型。

```console
pip install -U gptqmodel --no-build-isolation -v
```

## 量化模型

安装GPTQModel后，您可以开始量化模型。请参阅[GPTQModel readme](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization)以获取更多详细信息。

以下是如何量化`meta-llama/Llama-3.2-1B-Instruct`的示例：

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# 增加`batch_size`以匹配GPU/VRAM规格以加速量化
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)
```

## 使用vLLM运行量化模型

要使用vLLM运行GPTQModel量化模型，您可以使用[DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2)，命令如下：

```console
python examples/offline_inference/llm_engine_example.py \
    --model ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

## 使用vLLM的Python API运行GPTQModel

GPTQModel量化模型也直接通过LLM入口点支持：

```python
from vllm import LLM, SamplingParams

# 示例提示
prompts = [
    "你好，我的名字是",
    "美国总统是",
    "法国的首都是",
    "人工智能的未来是",
]

# 创建采样参数对象
sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

# 创建LLM
llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

# 从提示生成文本。输出是包含提示、生成文本和其他信息的RequestOutput对象列表
outputs = llm.generate(prompts, sampling_params)

# 打印输出
print("-"*50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}\n生成文本: {generated_text!r}")
    print("-"*50)
```