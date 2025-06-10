---
title: 故障排除
---
[](){ #troubleshooting }

本文档概述了一些您可以考虑的故障排除策略。如果您认为发现了一个错误，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)，查看是否已被报告。如果没有，请[提交一个新问题](https://github.com/vllm-project/vllm/issues/new/choose)，并提供尽可能多的相关信息。

!!! note
    在调试问题后，记得关闭任何定义的调试环境变量，或者直接启动一个新的 shell，以避免受残留的调试设置影响。否则，系统可能会因为启用了调试功能而变慢。

## 模型下载挂起

如果模型尚未下载到磁盘，vLLM 将从网络下载，这可能需要时间并依赖于您的网络连接。建议先使用 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 下载模型，并将本地路径传递给 vLLM。这样可以隔离问题。

## 从磁盘加载模型挂起

如果模型很大，从磁盘加载可能需要较长时间。注意模型的存储位置。某些集群使用跨节点的共享文件系统，例如分布式文件系统或网络文件系统，可能会很慢。最好将模型存储在本地磁盘上。此外，关注 CPU 内存使用情况，当模型过大时，可能会占用大量 CPU 内存，导致操作系统因频繁的磁盘和内存交换而变慢。

!!! note
    为了隔离模型下载和加载问题，您可以使用 `--load-format dummy` 参数跳过加载模型权重。这样可以检查模型下载和加载是否是瓶颈。

## 内存不足

如果模型太大，无法适应单个 GPU，您将遇到内存不足（OOM）错误。考虑采用[这些选项](../configuration/conserving_memory.md)来减少内存消耗。

## 生成质量变化

在 v0.8.0 中，默认采样参数的来源在 <gh-pr:12622> 中发生了变化。在 v0.8.0 之前，默认采样参数来自 vLLM 的中立默认设置。从 v0.8.0 开始，默认采样参数来自模型创建者提供的 `generation_config.json`。

在大多数情况下，这应该会带来更高质量的响应，因为模型创建者通常知道哪些采样参数最适合他们的模型。然而，在某些情况下，模型创建者提供的默认值可能会导致性能下降。

您可以通过使用 `--generation-config vllm`（在线）或 `generation_config="vllm"`（离线）尝试旧的默认值来检查是否出现这种情况。如果尝试后生成质量有所改善，建议继续使用 vLLM 默认值，并向 <https://huggingface.co> 的模型创建者提出请求，更新他们的默认 `generation_config.json`，以生成更高质量的结果。

## 启用更多日志

如果其他策略无法解决问题，可能是 vLLM 实例在某处卡住了。您可以使用以下环境变量来帮助调试问题：

- `export VLLM_LOGGING_LEVEL=DEBUG` 以启用更多日志。
- `export CUDA_LAUNCH_BLOCKING=1` 以识别导致问题的 CUDA 内核。
- `export NCCL_DEBUG=TRACE` 以启用 NCCL 的更多日志。
- `export VLLM_TRACE_FUNCTION=1` 以记录所有函数调用，供日志文件检查，以确定哪个函数崩溃或挂起。

## 网络配置错误

如果您的网络配置复杂，vLLM 实例可能无法获取正确的 IP 地址。您可能会看到类似 `DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` 的日志，IP 地址应该是正确的。如果不正确，请使用环境变量 `export VLLM_HOST_IP=<your_ip_address>` 覆盖 IP 地址。

您可能还需要设置 `export NCCL_SOCKET_IFNAME=<your_network_interface>` 和 `export GLOO_SOCKET_IFNAME=<your_network_interface>` 来指定 IP 地址的网络接口。

## 在 `self.graph.replay()` 附近出错

如果 vLLM 崩溃，并且错误跟踪在 `vllm/worker/model_runner.py` 中的 `self.graph.replay()` 附近捕获到错误，这是 CUDAGraph 中的 CUDA 错误。要识别导致错误的特定 CUDA 操作，您可以在命令行中添加 `--enforce-eager`，或在 [LLM][vllm.LLM] 类中设置 `enforce_eager=True`，以禁用 CUDAGraph 优化并隔离导致错误的精确 CUDA 操作。

[](){ #troubleshooting-incorrect-hardware-driver }

## 硬件/驱动错误

如果 GPU/CPU 通信无法建立，您可以使用以下 Python 脚本并按照下面的说明，确认 GPU/CPU 通信是否正常工作。

```python
# 测试 PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"预期 {world_size}，实际 {value}"

print("PyTorch NCCL 测试成功！")

# 测试 PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"预期 {world_size}，实际 {value}"

print("PyTorch GLOO 测试成功！")

if world_size <= 1:
    exit()

# 测试 vLLM NCCL，使用 CUDA 图
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
# pynccl 在 0.6.5+ 版本中默认启用，
# 但在 0.6.4 及以下版本中，我们需要手动启用。
# 为向后兼容保留代码，因为用户倾向于阅读最新文档。
pynccl.disabled = False

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    out = pynccl.all_reduce(data, stream=s)
    value = out.mean().item()
    assert value == world_size, f"预期 {world_size}，实际 {value}"

print("vLLM NCCL 测试成功！")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = out.mean().item()
assert value == world_size, f"预期 {world_size}，实际 {value}"

print("vLLM NCCL 使用 CUDA 图测试成功！")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
```

如果您在单节点上测试，请根据您想使用的 GPU 数量调整 `--nproc-per-node`：

```console
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<GPU数量> test.py
```

如果您在多节点上测试，请根据您的设置调整 `--nproc-per-node` 和 `--nnodes`，并将 `MASTER_ADDR` 设置为主节点的正确 IP 地址，确保所有节点都可访问。然后运行：

```console
NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test.py
```

如果脚本成功运行，您应该会看到消息 `sanity check is successful!`。

如果测试脚本挂起或崩溃，通常意味着硬件/驱动在某些方面出现问题。您应该联系系统管理员或硬件供应商寻求进一步帮助。作为常见解决方法，您可以尝试调整一些 NCCL 环境变量，例如 `export NCCL_P2P_DISABLE=1`，看看是否有帮助。请查看[其文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)以获取更多信息。请仅将这些环境变量作为临时解决方法，因为它们可能会影响系统性能。最佳解决方案仍是修复硬件/驱动，使测试脚本能够成功运行。

!!! note
    多节点环境比单节点环境更复杂。如果您看到类似 `torch.distributed.DistNetworkError` 的错误，可能是网络/DNS 设置不正确。在这种情况下，您可以通过命令行参数手动指定节点排名和 IP：

    - 在第一个节点运行：`NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 0 --master_addr $MASTER_ADDR test.py`。
    - 在第二个节点运行：`NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 1 --master_addr $MASTER_ADDR test.py`。

    根据您的设置调整 `--nproc-per-node`、`--nnodes` 和 `--node-rank`，确保在不同节点上执行不同的命令（使用不同的 `--node-rank`）。

[](){ #troubleshooting-python-multiprocessing }

## Python 多进程

### `RuntimeError` 异常

如果您在日志中看到类似以下的警告：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA 已初始化。
    我们必须使用 `spawn` 多进程启动方法。设置
    VLLM_WORKER_MULTIPROC_METHOD 为 'spawn'。请参阅
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    获取更多信息。
```

或来自 Python 的错误如下：

```console
RuntimeError:
        在当前进程完成其引导阶段之前，尝试启动一个新进程。

        这可能意味着您没有使用 fork 来启动子进程，
        并且您忘记在主模块中使用正确的习惯用法：

            if __name__ == '__main__':
                freeze_support()
                ...

        如果程序不会被冻结以生成可执行文件，
        可以省略 "freeze_support()" 行。

        要解决此问题，请参阅
        https://docs.python.org/3/library/multiprocessing.html
        中的“主模块安全导入”部分。
```

那么您必须更新您的 Python 代码，将 `vllm` 的使用置于 `if __name__ == '__main__':` 块中。例如，不要这样：

```python
import vllm

llm = vllm.LLM(...)
```

而是改为这样：

```python
if __name__ == '__main__':
    import vllm

    llm = vllm.LLM(...)
```

## `torch.compile` 错误

vLLM 高度依赖 `torch.compile` 来优化模型以获得更好的性能，这引入了对 `torch.compile` 功能和 `triton` 库的依赖。默认情况下，我们使用 `torch.compile` 来[优化一些函数](https://github.com/vllm-project/vllm/pull/10406)。在运行 vLLM 之前，您可以通过运行以下脚本检查 `torch.compile` 是否按预期工作：

```python
import torch

@torch.compile
def f(x):
    # 一个简单的函数来测试 torch.compile
    x = x + 1
    x = x * 2
    x = x.sin()
    return x

x = torch.randn(4, 4).cuda()
print(f(x))
```

如果它从 `torch/_inductor` 目录引发错误，通常意味着您使用的自定义 `triton` 库与您使用的 PyTorch 版本不兼容。请参见[此问题](https://github.com/vllm-project/vllm/issues/12219)作为示例。

## 模型检查失败

如果您看到类似以下的错误：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: 模型架构 ['<arch>'] 检查失败。请查看日志以获取更多详细信息。
```

这意味着 vLLM 无法导入模型文件。通常，这与缺少依赖项或 vLLM 构建中的过时二进制文件有关。请仔细阅读日志以确定错误的根本原因。

## 不支持的模型

如果您看到类似以下的错误：

```text
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' 对象不可迭代
```

或：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: 模型架构 ['<arch>'] 目前不支持。支持的架构：[……]
```

但您确定模型在[支持的模型列表][supported-models]中，可能是 vLLM 的模型解析出现问题。在这种情况下，请按照[这些步骤](../configuration/model_resolution.md)明确指定模型的 vLLM 实现。

## 无法推断设备类型

如果您看到类似 `RuntimeError: Failed to infer device type` 的错误，这意味着 vLLM 无法推断运行环境的设备类型。您可以查看[代码](gh-file:vllm/platforms/__init__.py)，了解 vLLM 如何推断设备类型以及为何无法按预期工作。在[此 PR](gh-pr:14195) 之后，您还可以设置环境变量 `VLLM_LOGGING_LEVEL=DEBUG` 以查看更详细的日志，帮助调试问题。

## 已知问题

- 在 `v0.5.2`、`v0.5.3` 和 `v0.5.3.post1` 中，存在由 [zmq](https://github.com/zeromq/pyzmq/issues/2000) 引起的错误，可能会根据机器配置偶尔导致 vLLM 挂起。解决方法是升级到最新版本的 `vllm`，以包含[修复](gh-pr:6759)。
- 为规避 NCCL 的[错误](https://github.com/NVIDIA/nccl/issues/1234)，所有 vLLM 进程将设置环境变量 `NCCL_CUMEM_ENABLE=0` 以禁用 NCCL 的 `cuMem` 分配器。这不会影响性能，仅提供内存优势。当外部进程希望与 vLLM 的进程建立 NCCL 连接时，也应设置此环境变量，否则环境设置不一致将导致 NCCL 挂起或崩溃，如在 [RLHF 集成](https://github.com/OpenRLHF/OpenRLHF/pull/604) 和[讨论](gh-issue:5723#issuecomment-2554389656) 中观察到的。