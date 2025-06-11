# Python 多进程

## 调试

请参阅[故障排除][troubleshooting-python-multiprocessing]页面，了解已知问题及解决方法。

## 引言

!!! warning
    源代码引用基于撰写本文时的代码状态，即2024年12月。

vLLM 中使用 Python 多进程的复杂性源于：

- vLLM 作为库使用，且无法控制使用 vLLM 的代码
- 多进程方法与 vLLM 依赖项之间的不同程度不兼容性

本文档描述了 vLLM 如何应对这些挑战。

## 多进程方法

[Python 多进程方法](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)包括：

- `spawn` - 启动一个新的 Python 进程。在 Windows 和 macOS 上为默认方法。

- `fork` - 使用 `os.fork()` 分叉 Python 解释器。在 Python 3.14 之前的 Linux 上为默认方法。

- `forkserver` - 启动一个服务器进程，按需分叉新进程。在 Python 3.14 及更高版本的 Linux 上为默认方法。

### 权衡

`fork` 是最快的方法，但与使用线程的依赖项不兼容。如果在 macOS 上使用 `fork`，可能会导致进程崩溃。

`spawn` 与依赖项的兼容性更高，但当 vLLM 作为库使用时可能出现问题。如果使用 vLLM 的代码没有使用 `__main__` 保护（`if __name__ == "__main__":`），vLLM 启动新进程时会无意中重新执行代码。这可能导致无限递归等问题。

`forkserver` 会启动一个服务器进程，按需分叉新进程。但当 vLLM 作为库使用时，它与 `spawn` 面临相同的问题。服务器进程作为新启动的进程创建，会重新执行未受 `__main__` 保护的代码。

对于 `spawn` 和 `forkserver`，进程不得依赖于继承 `fork` 提供的任何全局状态。

## 与依赖项的兼容性

多个 vLLM 依赖项表明偏好或要求使用 `spawn`：

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

更准确地说，在初始化这些依赖项后使用 `fork` 会出现已知问题。

## 当前状态 (v0)

环境变量 `VLLM_WORKER_MULTIPROC_METHOD` 可用于控制 vLLM 使用的多进程方法。当前默认值为 `fork`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

当我们知道自己控制进程（即使用了 `vllm` 命令）时，我们使用 `spawn`，因为它具有最广泛的兼容性。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor` 强制使用 `spawn`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

还有其他一些地方硬编码使用 `spawn`：

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

相关 PR：

- <gh-pr:8823>

## v1 中的先前状态

有一个环境变量用于控制 v1 引擎核心是否使用多进程，`VLLM_ENABLE_V1_MULTIPROCESSING`，默认关闭。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

当启用时，v1 的 `LLMEngine` 会创建一个新进程来运行引擎核心。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

默认关闭的原因是上述提到的所有问题——与依赖项的兼容性以及作为库使用的代码。

### v1 中的更改

没有一个 Python `multiprocessing` 的简单解决方案能够通用于所有场景。作为第一步，我们可以将 v1 设置为“尽力而为”的多进程方法选择，以最大化兼容性。

- 默认使用 `fork`。
- 当我们知道控制主进程（执行了 `vllm`）时，使用 `spawn`。
- 如果检测到 `cuda` 已初始化，强制使用 `spawn` 并发出警告。我们知道 `fork` 会失败，因此这是最佳选择。

已知仍会失败的情况是：作为库使用的代码在调用 vLLM 之前初始化了 `cuda`。我们发出的警告应指导用户添加 `__main__` 保护或禁用多进程。

如果发生已知失败情况，用户将看到两条消息解释情况。首先，来自 vLLM 的日志消息：

```console
警告 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA 已初始化。
    必须使用 `spawn` 多进程启动方法。将 VLLM_WORKER_MULTIPROC_METHOD 设置为 'spawn'。请参阅
    https://docs.vllm.ai/en/latest/usage/debugging.html#python-multiprocessing
    了解更多信息。
```

其次，Python 本身会抛出带有详细说明的异常：

```console
RuntimeError:
        在当前进程完成其引导阶段之前，尝试启动一个新进程。

        这通常意味着您没有使用 fork 来启动子进程，并且您忘记在主模块中使用正确的习惯用法：

            if __name__ == '__main__':
                freeze_support()
                ...

        如果程序不会被冻结以生成可执行文件，则可以省略 "freeze_support()" 行。

        要解决此问题，请参阅 https://docs.python.org/3/library/multiprocessing.html 中的“安全导入主模块”部分
```

## 考虑的替代方案

### 检测是否存在 `__main__` 保护

有人建议如果能检测到使用 vLLM 的代码是否具有 `__main__` 保护，我们可以表现得更好。这个 [StackOverflow 帖子](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard) 来自面临同样问题的库作者。

可以检测我们是否在原始的 `__main__` 进程中，还是在后续启动的进程中。然而，检测代码中是否存在 `__main__` 保护似乎并不简单。

此选项被认为不切实际而放弃。

### 使用 `forkserver`

乍看之下，`forkserver` 似乎是问题的良好解决方案。然而，其工作方式与 vLLM 作为库使用时面临与 `spawn` 相同的挑战。

### 始终强制使用 `spawn`

一种清理方法是始终强制使用 `spawn`，并记录当 vLLM 作为库使用时需要 `__main__` 保护。这会破坏现有代码并使 vLLM 更难使用，违背了让 `LLM` 类尽可能易用的愿望。

我们不会将这种复杂性推给用户，而是保留复杂性以尽力让事情正常工作。

## 未来工作

未来我们可能需要考虑不同的工作进程管理方式来解决这些挑战。

1. 我们可以实现类似 `forkserver` 的机制，但通过运行自己的子进程和自定义的工作进程管理入口点来启动进程管理器（启动一个 `vllm-manager` 进程）。

2. 我们可以探索其他更适合我们需求的库。例如：

- <https://github.com/joblib/loky>