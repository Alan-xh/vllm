---
title: 多模态数据处理
---
[](){ #mm-processing }

为了在 vLLM 中启用各种优化，例如 [分块预填充][chunked-prefill] 和 [前缀缓存][automatic-prefix-caching]，我们使用 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor]，根据 HF 处理器的输出，提供占位符特征标记（例如 `<image>`）与多模态输入（例如原始输入图像）之间的对应关系。

以下是 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 的主要功能：

## 提示更新检测

HF 处理器的主要职责之一是使用占位符标记更新提示。例如：

- 在字符串开头插入特征占位符标记（例如 `<image><image>...<image>`，其数量等于特征大小）。
- 将现有的输入占位符标记（例如单个图像的 `<image>`）替换为特征占位符标记（例如 `<image><image>...<image>`，其数量等于特征大小）。

关于哪些标记已被更新的信息是找到占位符特征标记与多模态输入之间对应关系的关键。

在 vLLM 中，此信息通过 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 中的 [PromptUpdate][vllm.multimodal.processing.PromptUpdate] 指定。我们可以通过检查更新标记的存在来自动检测 HF 是否更新了提示。

## 标记化的提示输入

为了在单独的进程中启用标记化，我们支持将输入标记 ID 与多模态数据一起传递。

### 问题

考虑 HF 处理器遵循以下主要步骤：

1. 标记化文本
2. 处理多模态输入
3. 执行提示更新

我们要求：

- 对于文本 + 多模态输入，应用所有步骤 1--3。
- 对于标记化 + 多模态输入，仅应用步骤 2--3。

如何在不重写 HF 处理器的情况下实现这一点？我们可以尝试在不同输入上多次调用 HF 处理器：

- 对于文本 + 多模态输入，直接调用 HF 处理器。
- 对于标记化 + 多模态输入，仅对多模态输入调用处理器。

虽然 HF 处理器原生支持文本 + 多模态输入，但对于标记化 + 多模态输入则不然：如果输入占位符标记的数量与多模态输入的数量不对应，会抛出错误。

此外，由于标记化的文本未经过 HF 处理器，我们必须自行应用步骤 3，以保持输出标记和多模态数据的一致性。

[](){ #mm-dummy-text }

### 虚拟文本

我们通过要求每个模型定义如何根据多模态输入的数量生成虚拟文本来解决第一个问题，通过 [get_dummy_text][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text] 实现。这使我们可以生成与多模态输入对应的虚拟文本，并将它们一起输入以获得处理后的多模态数据。

[](){ #mm-automatic-prompt-updating }

### 自动提示更新

我们通过在 [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates] 中实现与模型无关的代码来解决第二个问题，根据 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 输出的规格，自动使用特征占位符标记更新提示。

### 总结

借助虚拟文本和自动提示更新，我们的多模态处理器最终可以接受带有文本和标记提示的多模态数据。详细逻辑在 [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main] 中展示。

## 处理器输出缓存

一些 HF 处理器，例如 Qwen2-VL 的处理器，[非常慢](gh-issue:9238)。为了缓解这个问题，我们缓存 HF 处理器的多模态输出，以避免再次处理相同的多模态输入（例如图像）。

当新数据传入时，我们首先检查哪些项目在缓存中，哪些缺失。缺失的项目被批量传递到 HF 处理器中进行处理并缓存，然后与缓存中现有的项目合并。

由于我们仅处理缺失的多模态数据项目，输入占位符标记的数量不再与多模态输入的数量对应，因此它们不能与文本提示一起传递给 HF 处理器。因此，我们分别处理文本和多模态输入，使用 [虚拟文本][mm-dummy-text] 来避免 HF 错误。由于这跳过了 HF 的提示更新代码，我们随后应用 [自动提示更新][mm-automatic-prompt-updating] 以保持输出标记和多模态数据的一致性。