---
title: AutoAWQ
---
[](){ #auto-awq }

要创建一个新的4位量化模型，您可以利用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。  
量化将模型的精度从 BF16/FP16 降低到 INT4，从而有效减少模型的总内存占用。  
主要优势包括更低的延迟和内存使用量。

您可以通过安装 AutoAWQ 来量化自己的模型，或者选择 [Huggingface 上超过 6500 个模型](https://huggingface.co/models?search=awq)。

```console
pip install autoawq
```

安装 AutoAWQ 后，您即可开始量化模型。请参阅 [AutoAWQ 文档](https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization) 获取更多详细信息。以下是如何量化 `mistralai/Mistral-7B-Instruct-v0.2` 的示例：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化为4位并保存至 "{quant_path}"')
```

要使用 vLLM 运行 AWQ 模型，您可以使用 [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ) 并运行以下命令：

```console
python examples/offline_inference/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq
```

AWQ 模型也通过 LLM 入口直接支持：

```python
from vllm import LLM, SamplingParams

# 示例提示。
prompts = [
    "你好，我的名字是",
    "美国总统是",
    "法国的首都是",
    "人工智能的未来是",
]
# 创建采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建一个 LLM。
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
# 从提示生成文本。输出是一个包含提示、生成文本和其他信息的 RequestOutput 对象列表。
outputs = llm.generate(prompts, sampling_params)
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```