# TorchAO

TorchAO 是用于 PyTorch 的架构优化库，它为推理和训练提供了高性能数据类型、优化技术和内核，具备与原生 PyTorch 功能（如 torch.compile、FSDP 等）的可组合性。一些基准测试数据可以在[这里](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)找到。

我们推荐安装最新的 torchao 夜间构建版本：

```console
# 安装最新的 TorchAO 夜间构建版本
# 选择与您的系统匹配的 CUDA 版本（cu126、cu128 等）
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

## 量化 HuggingFace 模型
您可以使用 torchao 量化自己的 HuggingFace 模型，例如 [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) 和 [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao)，并将检查点保存到 HuggingFace Hub，例如 [这样](https://huggingface.co/jerryzh168/llama3-8b-int8wo)，使用以下示例代码：

```Python
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "我们晚餐吃什么？"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

hub_repo = # 您的 Hub 仓库 ID
tokenizer.push_to_hub(hub_repo)
quantized_model.push_to_hub(hub_repo, safe_serialization=False)
```

或者，您可以使用 [TorchAO 量化空间](https://huggingface.co/spaces/medmekk/TorchAO_Quantization) 通过简单的用户界面进行模型量化。