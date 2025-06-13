# NVIDIA TensorRT 模型优化器

[NVIDIA TensorRT 模型优化器](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 是一个专为 NVIDIA GPU 优化推理模型的库。它包括用于大型语言模型（LLM）、视觉语言模型（VLM）和扩散模型的后训练量化（PTQ）和量化感知训练（QAT）的工具。

我们建议使用以下命令安装该库：

```console
pip install nvidia-modelopt
```

## 使用 PTQ 量化 HuggingFace 模型

您可以使用 TensorRT 模型优化器仓库中提供的示例脚本来量化 HuggingFace 模型。用于 LLM PTQ 的主要脚本通常位于 `examples/llm_ptq` 目录中。

以下是一个展示如何使用 modelopt 的 PTQ API 量化模型的示例：

```python
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM

# 从 HuggingFace 加载模型
model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

# 选择量化配置，例如 FP8
config = mtq.FP8_DEFAULT_CFG

# 定义用于校准的前向循环函数
def forward_loop(model):
    for data in calib_set:
        model(data)

# 使用原地替换量化模块进行 PTQ
model = mtq.quantize(model, config, forward_loop)
```

在模型量化完成后，您可以使用导出 API 将其导出为量化检查点：

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # 量化后的模型
        export_dir,  # 导出文件存储的目录
    )
```

随后，量化检查点可以与 vLLM 一起部署。作为示例，以下代码展示了如何使用 vLLM 部署 `nvidia/Llama-3.1-8B-Instruct-FP8`，这是从 `meta-llama/Llama-3.1-8B-Instruct` 派生的 FP8 量化检查点：

```python
from vllm import LLM, SamplingParams

def main():

    model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"
    # 确保在加载 modelopt 检查点时指定 quantization='modelopt'
    llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

    prompts = [
        "你好，我的名字是",
        "美国总统是",
        "法国的首都是",
        "人工智能的未来是",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")

if __name__ == "__main__":
    main()
```