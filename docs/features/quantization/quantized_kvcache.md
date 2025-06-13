---
title: 量化键值缓存
---
[](){ #quantized-kvcache }

## FP8 键值缓存

将键值缓存量化为 FP8 可以减少其内存占用。这增加了缓存中可以存储的令牌数量，从而提高吞吐量。

### FP8 格式

[OCP（开放计算项目）](https://www.opencompute.org) 指定了两种常见的 8 位浮点数据格式：

- E5M2（5 位指数和 2 位尾数）
- E4M3FN（4 位指数和 3 位尾数，通常简称为 E4M3）

E4M3 格式相比 E5M2 提供更高的精度。然而，由于其动态范围较小（±240.0），E4M3 通常需要为每个量化张量配备一个更高精度的（FP32）缩放因子。

### 当前限制

目前仅支持按张量（标量）的缩放因子。正在开发更细粒度的缩放因子支持（例如按通道）。

### 性能影响

当前的 FP8 键值缓存实现主要通过允许大约两倍的键值缓存分配空间来提高吞吐量。这可以实现：

- 处理单个请求的更长上下文长度，或
- 处理更多并发请求批次

然而，由于当前实现尚未包括融合的反量化和注意力操作，因此暂无延迟改进。未来的版本将支持硬件加速的量化注意力操作，预计将提供额外的性能优势。虽然最新的硅芯片（如 AMD MI300、NVIDIA Hopper 或更高版本）支持 FP8 与其他格式（FP32、FP16、BF16）之间的原生硬件转换，但这一优势尚未完全实现。

研究表明，FP8 E4M3 量化通常仅对推理精度产生最小的影响，使其成为吞吐量优化的实用选择。

## 使用示例

以下是如何启用 FP8 量化的示例：

```python
# 要动态计算键值缓存缩放因子，请启用 calculate_kv_scales 参数

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
          kv_cache_dtype="fp8",
          calculate_kv_scales=True)
prompt = "伦敦是以下国家的首都"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

`kv_cache_dtype` 参数指定键值缓存存储的数据类型：
- `"auto"`：使用模型默认的“未量化”数据类型
- `"fp8"` 或 `"fp8_e4m3"`：在 CUDA 11.8+ 和 ROCm（AMD GPU）上支持
- `"fp8_e5m2"`：在 CUDA 11.8+ 上支持

## 校准缩放因子以提高精度

为了在使用 FP8 键值缓存时获得最佳模型质量，我们建议使用针对代表性推理数据调优的校准缩放因子。[LLM Compressor](https://github.com/vllm-project/llm-compressor/) 是推荐的工具。

### 安装

首先，安装所需的依赖项：

```console
pip install llmcompressor
```

### 使用示例

以下是使用 `meta-llama/Llama-3.1-8B-Instruct` 的完整示例（大多数模型可以使用相同的模式）：

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot

# 选择模型并加载
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 选择校准数据集
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# 配置校准参数
NUM_CALIBRATION_SAMPLES = 512  # 512 个样本是一个不错的起点
MAX_SEQUENCE_LENGTH = 2048

# 加载并预处理数据集
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# 配置量化设置
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

# 应用量化
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 保存量化模型：Llama-3.1-8B-Instruct-FP8-KV
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

上述脚本将在当前目录中创建一个包含量化模型的文件夹（例如 `Llama-3.1-8B-Instruct-FP8-KV`），其中包含校准的缩放因子。

运行模型时，必须指定 `kv_cache_dtype="fp8"` 以启用键值缓存量化和使用缩放因子。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="Llama-3.1-8B-Instruct-FP8-KV", kv_cache_dtype="fp8")
prompt = "伦敦是以下国家的首都"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```