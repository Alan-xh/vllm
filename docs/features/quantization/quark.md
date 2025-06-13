---
title: AMD QUARK
---
[](){ #quark }

量化可以有效减少内存和带宽使用量，加速计算并提高吞吐量，同时精度损失极小。vLLM 可以利用 [Quark](https://quark.docs.amd.com/latest/)，一个灵活且强大的量化工具包，生成高性能的量化模型以在 AMD GPU 上运行。Quark 专门支持大型语言模型的量化，包括权重、激活和 KV 缓存量化，以及 AWQ、GPTQ、Rotation 和 SmoothQuant 等前沿量化算法。

## Quark 安装

在量化模型之前，您需要先安装 Quark。可以使用 pip 安装最新版本的 Quark：

```console
pip install amd-quark
```

您可以参考 [Quark 安装指南](https://quark.docs.amd.com/latest/install.html) 获取更多安装详情。

此外，为了评估，还需安装 `vllm` 和 `lm-evaluation-harness`：

```console
pip install vllm lm-eval==0.4.4
```

## 量化过程

安装 Quark 后，我们将通过一个示例说明如何使用 Quark。  
Quark 量化过程可以分为以下 5 个步骤：

1. 加载模型
2. 准备校准数据加载器
3. 设置量化配置
4. 量化模型并导出
5. 在 vLLM 中评估

### 1. 加载模型

Quark 使用 [Transformers](https://huggingface.co/docs/transformers/en/index) 来获取模型和分词器。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
MAX_SEQ_LEN = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token
```

### 2. 准备校准数据加载器

Quark 使用 [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 加载校准数据。有关如何高效使用校准数据集的更多详情，请参阅 [添加校准数据集](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html)。

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1
NUM_CALIBRATION_DATA = 512

# 加载数据集并获取校准数据
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION_DATA]

tokenized_outputs = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader(tokenized_outputs['input_ids'],
    batch_size=BATCH_SIZE, drop_last=True)
```

### 3. 设置量化配置

我们需要设置量化配置，您可以查看 [Quark 配置指南](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html) 获取更多详情。此处我们使用 FP8 每张量量化应用于权重、激活和 KV 缓存，量化算法为 AutoSmoothQuant。

!!! note
    请注意，量化算法需要一个 JSON 配置文件，配置文件位于 [Quark PyTorch 示例](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html) 的 `examples/torch/language_modeling/llm_ptq/models` 目录下。例如，Llama 的 AutoSmoothQuant 配置文件为 `examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json`。

```python
from quark.torch.quantization import (Config, QuantizationConfig,
                                     FP8E4M3PerTensorSpec,
                                     load_quant_algo_config_from_file)

# 定义 FP8/每张量/静态规格
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
    is_dynamic=False).to_quantization_spec()

# 定义全局量化配置，输入张量和权重应用 FP8_PER_TENSOR_SPEC
global_quant_config = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
    weight=FP8_PER_TENSOR_SPEC)

# 定义 KV 缓存层的量化配置，输出张量应用 FP8_PER_TENSOR_SPEC
KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
kv_cache_quant_config = {name :
    QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                       weight=global_quant_config.weight,
                       output_tensors=KV_CACHE_SPEC)
    for name in kv_cache_layer_names_for_llama}
layer_quant_config = kv_cache_quant_config.copy()

# 通过配置文件定义算法配置
LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE =
    'examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json'
algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

EXCLUDE_LAYERS = ["lm_head"]
quant_config = Config(
    global_quant_config=global_quant_config,
    layer_quant_config=layer_quant_config,
    kv_cache_quant_config=kv_cache_quant_config,
    exclude=EXCLUDE_LAYERS,
    algo_config=algo_config)
```

### 4. 量化模型并导出

接下来，我们可以应用量化。在量化后，我们需要先冻结量化模型，然后再导出。请注意，我们需要以 HuggingFace 的 `safetensors` 格式导出模型，更多导出格式详情请参阅 [HuggingFace 格式导出](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html)。

```python
import torch
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.export import ExporterConfig, JsonExporterConfig

# 应用量化
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# 冻结量化模型以便导出
freezed_model = quantizer.freeze(model)

# 定义导出配置
LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
export_config = ExporterConfig(json_export_config=JsonExporterConfig())
export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP

# 模型：Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant
EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
with torch.no_grad():
    exporter.export_safetensors_model(freezed_model,
        quant_config=quant_config, tokenizer=tokenizer)
```

### 5. 在 vLLM 中评估

现在，您可以通过 LLM 入口点直接加载和运行 Quark 量化模型：

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
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建 LLM
llm = LLM(model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
          kv_cache_dtype='fp8', quantization='quark')
# 从提示生成文本，输出是一个包含提示、生成文本和其他信息的 RequestOutput 对象列表
outputs = llm.generate(prompts, sampling_params)
# 打印输出
print("\n生成输出：\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示：    {prompt!r}")
    print(f"输出：    {generated_text!r}")
    print("-" * 60)
```

或者，您可以使用 `lm_eval` 来评估精度：

```console
$ lm_eval --model vllm \
  --model_args pretrained=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```

## Quark 量化脚本
除了上述 Python API 示例外，Quark 还提供了一个 [量化脚本](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)，可以更方便地量化大型语言模型。它支持使用多种不同的量化方案和优化算法来量化模型。它可以导出量化模型并即时运行评估任务。使用该脚本，上述示例可以简化为：

```console
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```