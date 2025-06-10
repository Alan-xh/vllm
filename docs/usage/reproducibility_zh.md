# 再现性

vLLM 默认情况下为了性能考虑，不保证结果的再现性。要实现可再现的结果，需要进行以下设置：

- 对于 V1：关闭多进程以使调度确定性，设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
- 对于 V0：设置全局种子（见下文）。

示例：<gh-file:examples/offline_inference/reproducibility.py>

!!! warning

    应用上述设置会[改变用户代码中的随机状态](#locality-of-random-state)。

!!! note

    即使应用了上述设置，vLLM 仅在运行于相同硬件和相同 vLLM 版本时提供再现性。
    此外，在线服务 API（`vllm serve`）不支持再现性，因为在在线环境中几乎不可能使调度具有确定性。

## 设置全局种子

vLLM 中的 `seed` 参数用于控制各种随机数生成器的随机状态。

如果指定了一个特定的种子值，将相应设置 `random`、`np.random` 和 `torch.manual_seed` 的随机状态。

然而，在某些情况下，设置种子也会[改变用户代码中的随机状态](#locality-of-random-state)。

### 默认行为

在 V0 中，`seed` 参数默认为 `None`。当 `seed` 参数为 `None` 时，不会设置 `random`、`np.random` 和 `torch.manual_seed` 的随机状态。这意味着如果 `temperature > 0`，每次运行 vLLM 将产生不同的结果，这是符合预期的。

在 V1 中，`seed` 参数默认为 `0`，这会为每个工作进程设置随机状态，因此即使 `temperature > 0`，每次 vLLM 运行的结果也将保持一致。

!!! note

    在 V1 中无法取消指定种子，因为不同的工作进程需要采样相同的输出，以支持如推测解码之类的工作流程。
    
    更多信息，请参见：<gh-pr:17929>

### 随机状态的局部性

在以下情况下，vLLM 会更新用户代码（即构造 [LLM][vllm.LLM] 类的代码）中的随机状态：

- 对于 V0：指定了种子。
- 对于 V1：工作进程与用户代码运行在同一进程中，即：`VLLM_ENABLE_V1_MULTIPROCESSING=0`。

默认情况下，这些条件不生效，因此您可以放心使用 vLLM，而无需担心意外使依赖随机状态的后续操作变得确定性。