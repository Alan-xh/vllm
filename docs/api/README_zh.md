# 概述

[](){ #configuration }

## 配置

vLLM 配置类的 API 文档。

- [vllm.config.ModelConfig][]
- [vllm.config.CacheConfig][]
- [vllm.config.TokenizerPoolConfig][]
- [vllm.config.LoadConfig][]
- [vllm.config.ParallelConfig][]
- [vllm.config.SchedulerConfig][]
- [vllm.config.DeviceConfig][]
- [vllm.config.SpeculativeConfig][]
- [vllm.config.LoRAConfig][]
- [vllm.config.PromptAdapterConfig][]
- [vllm.config.MultiModalConfig][]
- [vllm.config.PoolerConfig][]
- [vllm.config.DecodingConfig][]
- [vllm.config.ObservabilityConfig][]
- [vllm.config.KVTransferConfig][]
- [vllm.config.CompilationConfig][]
- [vllm.config.VllmConfig][]

[](){ #offline-inference-api }

## 离线推理

LLM 类。

- [vllm.LLM][]

LLM 输入。

- [vllm.inputs.PromptType][]
- [vllm.inputs.TextPrompt][]
- [vllm.inputs.TokensPrompt][]

## vLLM 引擎

用于离线和在线推理的引擎类。

- [vllm.LLMEngine][]
- [vllm.AsyncLLMEngine][]

## 推理参数

vLLM API 的推理参数。

[](){ #sampling-params }
[](){ #pooling-params }

- [vllm.SamplingParams][]
- [vllm.PoolingParams][]

[](){ #multi-modality }

## 多模态

vLLM 通过 [vllm.multimodal][] 包提供对多模态模型的实验性支持。

多模态输入可以通过 [vllm.inputs.PromptType][] 中的 `multi_modal_data` 字段与文本和令牌提示一起传递给 [支持的模型][supported-mm-models]。

想添加自己的多模态模型？请遵循 [此处][supports-multimodal] 列出的说明。

- [vllm.multimodal.MULTIMODAL_REGISTRY][]

### 输入

面向用户的输入。

- [vllm.multimodal.inputs.MultiModalDataDict][]

内部数据结构。

- [vllm.multimodal.inputs.PlaceholderRange][]
- [vllm.multimodal.inputs.NestedTensors][]
- [vllm.multimodal.inputs.MultiModalFieldElem][]
- [vllm.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.multimodal.inputs.MultiModalKwargs][]
- [vllm.multimodal.inputs.MultiModalInputs][]

### 数据解析

- [vllm.multimodal.parse][]

### 数据处理

- [vllm.multimodal.processing][]

### 内存分析

- [vllm.multimodal.profiling][]

### 注册表

- [vllm.multimodal.registry][]

## 模型开发

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]