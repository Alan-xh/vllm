---
title: 架构概览
---
[](){ #arch-overview }

本文档提供了 vLLM 架构的概览。

[TOC]

## 入口点

vLLM 提供了多种与系统交互的入口点。以下图表展示了它们之间的关系。

![入口点图表](../assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM 类

`LLM` 类提供了主要的 Python 接口，用于进行离线推理，即在不使用独立模型推理服务器的情况下与模型交互。

以下是 `LLM` 类使用的示例：

```python
from vllm import LLM, SamplingParams

# 定义输入提示列表
prompts = [
    "你好，我的名字是",
    "法国的首都是",
    "最大的海洋是",
]

# 定义采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 使用 OPT-125M 模型初始化 LLM 引擎
llm = LLM(model="facebook/opt-125m")

# 为输入提示生成输出
outputs = llm.generate(prompts, sampling_params)

# 打印生成的输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

更多 API 详情可在 API 文档的[离线推理](#offline-inference-api)部分找到。

`LLM` 类的代码可在 <gh-file:vllm/entrypoints/llm.py> 中找到。

### OpenAI 兼容的 API 服务器

vLLM 的第二个主要接口是通过其 OpenAI 兼容的 API 服务器。可以使用 `vllm serve` 命令启动该服务器。

```bash
vllm serve <model>
```

`vllm` CLI 的代码可在 <gh-file:vllm/entrypoints/cli/main.py> 中找到。

有时，你可能会看到直接使用 API 服务器入口点，而不是通过 `vllm` CLI 命令。例如：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

该代码可在 <gh-file:vllm/entrypoints/openai/api_server.py> 中找到。

有关 API 服务器的更多详情，请参阅 [OpenAI 兼容服务器][openai-compatible-server] 文档。

## LLM 引擎

`LLMEngine` 和 `AsyncLLMEngine` 类是 vLLM 系统功能的核心，负责处理模型推理和异步请求处理。

![LLMEngine 图表](../assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine

`LLMEngine` 类是 vLLM 引擎的核心组件。它负责接收客户端的请求并从模型生成输出。`LLMEngine` 包括输入处理、模型执行（可能分布在多个主机和/或 GPU 上）、调度和输出处理。

- **输入处理**：使用指定的分词器处理输入文本的分词。
- **调度**：选择在每个步骤中处理哪些请求。
- **模型执行**：管理语言模型的执行，包括跨多个 GPU 的分布式执行。
- **输出处理**：处理模型生成的输出，将语言模型的 token ID 解码为人类可读的文本。

`LLMEngine` 的代码可在 <gh-file:vllm/engine/llm_engine.py> 中找到。

### AsyncLLMEngine

`AsyncLLMEngine` 类是 `LLMEngine` 类的异步包装器。它使用 `asyncio` 创建一个后台循环，持续处理传入的请求。`AsyncLLMEngine` 设计用于在线服务，可以处理多个并发请求并将输出流式传输给客户端。

OpenAI 兼容的 API 服务器使用 `AsyncLLMEngine`。还有一个更简单的示例 API 服务器，代码在 <gh-file:vllm/entrypoints/api_server.py> 中。

`AsyncLLMEngine` 的代码可在 <gh-file:vllm/engine/async_llm_engine.py> 中找到。

## 工作进程

工作进程是一个运行模型推理的进程。vLLM 遵循使用一个进程控制一个加速器设备（例如 GPU）的常见做法。例如，如果我们使用大小为 2 的张量并行和大小为 2 的流水线并行，总共会有 4 个工作进程。工作进程通过其 `rank` 和 `local_rank` 进行标识。`rank` 用于全局协调，而 `local_rank` 主要用于分配加速器设备和访问本地资源，如文件系统和共享内存。

## 模型运行器

每个工作进程有一个模型运行器对象，负责加载和运行模型。模型执行逻辑的大部分都在这里，例如准备输入张量和捕获 cudagraph。

## 模型

每个模型运行器对象有一个模型对象，这是实际的 `torch.nn.Module` 实例。有关各种配置如何影响最终获取的类，请参阅 [huggingface_integration][huggingface-integration]。

## 类层次结构

下图展示了 vLLM 的类层次结构：

![](../assets/design/hierarchy.png)


这种类层次结构背后有几个重要的设计选择：

1. **扩展性**：层次结构中的所有类都接受一个包含所有必要信息的配置对象。[VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036) 类是传递的主要配置对象。类层次结构相当深，每个类都需要读取其感兴趣的配置。通过将所有配置封装在一个对象中，我们可以轻松地传递配置对象并访问所需的配置。假设我们想添加一个新功能（考虑到 LLM 推理领域的快速发展，这很常见），仅涉及模型运行器。我们只需要在 `VllmConfig` 类中添加新的配置选项。由于我们传递整个配置对象，只需在 `VllmConfig` 类中添加配置选项，模型运行器即可直接访问。我们无需更改引擎、工作进程或模型类的构造函数来传递新的配置选项。

2. **统一性**：模型运行器需要一个统一的接口来创建和初始化模型。vLLM 支持超过 50 种流行的开源模型。每种模型都有自己的初始化逻辑。如果构造函数签名因模型而异，模型运行器将不知道如何根据模型调用构造函数，而无需复杂且容易出错的检查逻辑。通过使模型类的构造函数统一，模型运行器可以轻松创建和初始化模型，而无需知道具体模型类型。这对于组合模型也很有用。视觉-语言模型通常由视觉模型和语言模型组成。通过使构造函数统一，我们可以轻松创建视觉模型和语言模型，并将它们组合成视觉-语言模型。

!!! note
    为支持这一变化，所有 vLLM 模型的签名已更新为：

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    为避免意外传递错误的参数，构造函数现在仅限关键字参数。这确保如果传递了旧配置，构造函数会报错。vLLM 开发者已为 vLLM 内的所有模型完成了这一更改。对于外部注册的模型，开发者需要更新其模型，例如通过添加适配代码以将旧构造函数签名适配到新签名：

    ```python
    class MyOldModel(nn.Module):
        def __init__(
            self,
            config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
            prefix: str = "",
        ) -> None:
            ...

    from vllm.config import VllmConfig
    class MyNewModel(MyOldModel):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            config = vllm_config.model_config.hf_config
            cache_config = vllm_config.cache_config
            quant_config = vllm_config.quant_config
            lora_config = vllm_config.lora_config
            super().__init__(config, cache_config, quant_config, lora_config, prefix)

    if __version__ >= "0.6.4":
        MyModel = MyNewModel
    else:
        MyModel = MyOldModel
    ```

    这样，模型可以兼容 vLLM 的旧版本和新版本。

3. **初始化时的分片和量化**：某些功能需要更改模型权重。例如，张量并行需要对模型权重进行分片，量化需要对模型权重进行量化。有两种实现此功能的方法。一种是在模型初始化后更改模型权重。另一种是在模型初始化期间更改模型权重。vLLM 选择了后者。第一种方法对大型模型不可扩展。假设我们要运行一个 405B 模型（权重约为 810GB），使用 16 个 H100 80GB GPU。理想情况下，每个 GPU 应仅加载 50GB 权重。如果在模型初始化后更改模型权重，我们需要将完整的 810GB 权重加载到每个 GPU 上，然后再进行分片，这会导致巨大的内存开销。相反，如果在模型初始化期间分片权重，每一层只会创建所需的权重分片，从而大大减少内存开销。量化的情况也是如此。请注意，我们还在模型的构造函数中添加了一个额外的参数 `prefix`，以便模型可以根据前缀以不同方式初始化自己。这对于非均匀量化很有用，其中模型的不同部分以不同方式量化。`prefix` 对于顶级模型通常是空字符串，对于子模型则是像 `"vision"` 或 `"language"` 这样的字符串。通常，它与检查点文件中模块状态字典的名称相匹配。

这种设计的一个缺点是很难为 vLLM 中的单个组件编写单元测试，因为每个组件都需要由完整的配置对象初始化。我们通过提供一个默认初始化函数来解决这个问题，该函数创建一个所有字段都设置为 `None` 的默认配置对象。如果我们要测试的组件只关心配置对象中的几个字段，我们可以创建一个默认配置对象并设置我们关心的字段。这样，我们可以隔离地测试组件。请注意，vLLM 中的许多测试是端到端测试，测试整个系统，因此这不是一个大问题。

总之，完整的配置对象 `VllmConfig` 可以被视为引擎级别的全局状态，在所有 vLLM 类中共享。