---
title：引擎参数
---
[](){ #engine-args }

引擎参数控制 vLLM 引擎的行为。

- 对于 [offline inference][offline-inference]，它们是 [LLM][vllm.LLM] 类参数的一部分。
- 对于 [online serving][openai-compatible-server]，它们是 `vllm serve` 参数的一部分。

您可以查看 [EngineArgs][vllm.engine.arg_utils.EngineArgs] 和 [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs] 来了解可用的引擎参数。

但是，这些类是 [vllm.config][] 中定义的配置类的组合。因此，我们建议您阅读它们文档最齐全的文档。

对于离线推理，您可以访问这些配置类；对于在线服务，您可以使用“vllm serve --help”交叉引用配置，其参数按配置分组。

!!! 注意
[AsyncLLMEngine][vllm.engine.async_llm_engine.AsyncLLMEngine] 还提供了用于在线服务的额外参数。您可以通过运行“vllm serve --help”找到这些参数。