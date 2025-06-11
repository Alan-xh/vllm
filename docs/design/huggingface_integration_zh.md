---
title: 与HuggingFace的集成
---
[](){ #huggingface-integration }

本文档描述了vLLM如何与HuggingFace库集成。我们将逐步解释运行`vllm serve`时幕后发生的事情。

假设我们想通过运行`vllm serve Qwen/Qwen2-7B`来服务流行的QWen模型。

1. `model`参数为`Qwen/Qwen2-7B`。vLLM通过检查对应的配置文件`config.json`来确定该模型是否存在。具体实现见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)。在此过程中：
    - 如果`model`参数对应一个存在的本地路径，vLLM将直接从该路径加载配置文件。
    - 如果`model`参数是一个由用户名和模型名组成的HuggingFace模型ID，vLLM将首先尝试从HuggingFace本地缓存中加载配置文件，使用`model`参数作为模型名，`--revision`参数作为修订版本。有关HuggingFace缓存工作原理的更多信息，请参见[其网站](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)。
    - 如果`model`参数是HuggingFace模型ID但在缓存中未找到，vLLM将从HuggingFace模型中心下载配置文件。具体实现参见[此函数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)。输入参数包括`model`参数作为模型名，`--revision`参数作为修订版本，以及环境变量`HF_TOKEN`作为访问模型中心的令牌。在我们的例子中，vLLM将下载[config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json)文件。

2. 在确认模型存在后，vLLM加载其配置文件并将其转换为字典。具体实现见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)。

3. 接下来，vLLM[检查](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)配置字典中的`model_type`字段，以[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)要使用的配置对象。vLLM直接支持一些`model_type`值；支持的列表见[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)。如果`model_type`不在列表中，vLLM将使用[AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained)加载配置类，参数包括`model`、 `--revision`和`--trust_remote_code`。请注意：
    - HuggingFace也有自己的逻辑来确定要使用的配置类。它会再次使用`model_type`字段在transformers库中搜索类名；支持的模型列表见[此处](https://github.com/huggingface/transformers/tree/main/src/transformers/models)。如果未找到`model_type`，HuggingFace将使用配置文件中的`auto_map`字段来确定类名。具体来说，是`auto_map`下的`AutoConfig`字段。参见[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json)示例。
    - `auto_map`下的`AutoConfig`字段指向模型仓库中的模块路径。为了创建配置类，HuggingFace会导入该模块并使用`from_pretrained`方法加载配置类。这通常可能导致任意代码执行，因此只有在启用`--trust_remote_code`时才会执行。

4. 随后，vLLM对配置对象应用一些历史补丁。这些补丁主要与RoPE配置相关；具体实现见[此处](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)。

5. 最后，vLLM可以确定要初始化的模型类。vLLM使用配置对象中的`architectures`字段来确定要初始化的模型类，因为它在[其注册表](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)中维护了从架构名称到模型类的映射。如果在注册表中未找到架构名称，则意味着vLLM不支持该模型架构。对于`Qwen/Qwen2-7B`，`architectures`字段为`["Qwen2ForCausalLM"]`，对应于[vLLM代码](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364)中的`Qwen2ForCausalLM`类。该类将根据各种配置进行自我初始化。

除此之外，vLLM还依赖HuggingFace的两个方面：

1. **分词器**：vLLM使用HuggingFace的分词器对输入文本进行分词。分词器使用[AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)加载，参数包括`model`作为模型名和`--revision`作为修订版本。还可以通过在`vllm serve`命令中指定`--tokenizer`参数来使用另一个模型的分词器。其他相关参数包括`--tokenizer-revision`和`--tokenizer-mode`。请查看HuggingFace文档以了解这些参数的含义。这部分逻辑可在[get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87)函数中找到。值得注意的是，vLLM会在[get_cached_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L24)中缓存分词器的一些昂贵属性。

2. **模型权重**：vLLM使用`model`参数作为模型名和`--revision`参数作为修订版本，从HuggingFace模型中心下载模型权重。vLLM提供`--load-format`参数来控制从模型中心下载哪些文件。默认情况下，它会尝试加载safetensors格式的权重，如果safetensors格式不可用，则回退到PyTorch bin格式。还可以通过传递`--load-format dummy`跳过权重下载。
    - 建议使用safetensors格式，因为它在分布式推理中加载效率高，且对任意代码执行是安全的。有关safetensors格式的更多信息，请参见[文档](https://huggingface.co/docs/safetensors/en/index)。这部分逻辑可在[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)找到。请注意：

这完成了vLLM与HuggingFace的集成。

总之，vLLM从HuggingFace模型中心或本地目录读取配置文件`config.json`、分词器和模型权重。它使用来自vLLM、HuggingFace transformers或从模型仓库加载的配置类。