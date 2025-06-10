---
title: 常见问题解答
---
[](){ #faq }

> 问：如何使用 OpenAI API 在单个端口上服务多个模型？

答：假设您指的是使用 OpenAI 兼容服务器同时服务多个模型，目前这是不支持的。您可以同时运行多个服务器实例（每个实例服务于一个不同的模型），并使用另一层来将传入的请求路由到正确的服务器。

---

> 问：用于离线推理嵌入的模型有哪些推荐？

答：您可以尝试 [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 和 [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)；
更多模型列于[此处][supported-models]。

通过提取隐藏状态，vLLM 可以将文本生成模型（如 [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)、
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)）自动转换为嵌入模型，
但这些模型的性能预计不如专门为嵌入任务训练的模型。

---

> 问：vLLM 中同一提示的输出在多次运行中会有变化吗？

答：是的，可能会发生变化。vLLM 无法保证输出 token 的对数概率（logprobs）稳定。由于 Torch 操作的数值不稳定性或批处理 Torch 操作的非确定性行为（当批处理发生变化时），logprobs 可能会出现变化。更多详情，请参阅 [数值精度部分](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。

在 vLLM 中，由于其他并发请求、批处理大小的变化或推测性解码中的批处理扩展等因素，相同的请求可能会以不同方式进行批处理。这些批处理差异结合 Torch 操作的数值不稳定性，可能导致每一步的 logit/logprob 值略有不同。这种差异可能会累积，最终导致采样的 token 不同。一旦采样到不同的 token，进一步的差异可能会更明显。

## 缓解策略

- 为提高稳定性和减少差异，可以使用 `float32`。请注意，这将需要更多内存。
- 如果使用 `bfloat16`，切换到 `float16` 也会有所帮助。
- 使用请求种子可以帮助在温度 > 0 时实现更稳定的生成，但由于精度差异，仍然可能出现不一致。