---
title: 分布式推理与服务
---
[](){ #distributed-serving }

## 如何决定分布式推理策略？

在深入探讨分布式推理和服务的细节之前，我们首先明确何时使用分布式推理以及有哪些策略可供选择。常见的实践如下：

- **单 GPU（无分布式推理）**：如果你的模型可以适配单个 GPU，你可能不需要使用分布式推理。只需使用单个 GPU 运行推理即可。
- **单节点多 GPU（张量并行推理）**：如果模型太大，无法适配单个 GPU，但可以适配单节点内的多个 GPU，你可以使用张量并行。张量并行大小是你希望使用的 GPU 数量。例如，如果一个节点有 4 个 GPU，你可以将张量并行大小设置为 4。
- **多节点多 GPU（张量并行加流水线并行推理）**：如果模型太大，无法适配单节点，你可以结合使用张量并行和流水线并行。张量并行大小是每个节点内使用的 GPU 数量，流水线并行大小是你希望使用的节点数量。例如，如果有 2 个节点共 16 个 GPU（每节点 8 个 GPU），你可以将张量并行大小设置为 8，流水线并行大小设置为 2。

简而言之，你需要增加 GPU 和节点数量，直到有足够的 GPU 内存来容纳模型。张量并行大小应为每个节点内的 GPU 数量，流水线并行大小应为节点数量。

!!! note
    在添加足够的 GPU 和节点以容纳模型后，你可以先运行 vLLM，它会打印类似 `# GPU blocks: 790` 的日志。将这个数字乘以 `16`（块大小），即可大致得到当前配置下可服务的最大 token 数量。如果这个数字不令人满意，例如你想要更高的吞吐量，可以进一步增加 GPU 或节点数量，直到块数量足够。

!!! note
    有一个特殊情况：如果模型可以适配单节点内的多个 GPU，但 GPU 数量无法均分模型大小，你可以使用流水线并行，它会沿层分割模型并支持不均匀分割。在这种情况下，张量并行大小应设为 1，流水线并行大小应设为 GPU 数量。

## 在单节点上运行 vLLM

vLLM 支持分布式张量并行和流水线并行推理与服务。目前，我们支持 [Megatron-LM 的张量并行算法](https://arxiv.org/pdf/1909.08053.pdf)。我们使用 [Ray](https://github.com/ray-project/ray) 或 Python 原生多进程来管理分布式运行时。在单节点部署时可以使用多进程，多节点推理目前需要 Ray。

当未在 Ray 放置组中运行且同一节点上有足够的 GPU 可用于配置的 `tensor_parallel_size` 时，默认使用多进程，否则使用 Ray。可以通过 `LLM` 类的 `distributed_executor_backend` 参数或 API 服务器的 `--distributed-executor-backend` 参数覆盖此默认值。将其设置为 `mp` 表示多进程，`ray` 表示 Ray。对于多进程情况，无需安装 Ray。

要使用 `LLM` 类运行多 GPU 推理，将 `tensor_parallel_size` 参数设置为你希望使用的 GPU 数量。例如，在 4 个 GPU 上运行推理：

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

要运行多 GPU 服务，在启动服务器时传入 `--tensor-parallel-size` 参数。例如，在 4 个 GPU 上运行 API 服务器：

```console
 vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

你还可以额外指定 `--pipeline-parallel-size` 以启用流水线并行。例如，在 8 个 GPU 上结合流水线并行和张量并行运行 API 服务器：

```console
 vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## 在多节点上运行 vLLM

如果单节点没有足够的 GPU 来容纳模型，你可以使用多个节点运行模型。确保所有节点的执行环境一致非常重要，包括模型路径和 Python 环境。推荐的方法是使用 Docker 镜像来确保环境一致，并通过将主机映射到相同的 Docker 配置来隐藏主机异构性。

第一步是启动容器并将其组织成一个集群。我们提供了辅助脚本 <gh-file:examples/online_serving/run_cluster.sh> 来启动集群。请注意，此脚本启动的 Docker 不具备运行性能分析和跟踪工具所需的管理员权限。如需此功能，可以在 Docker 运行命令中通过 `--cap-add` 选项添加 `CAP_SYS_ADMIN`。

选择一个节点作为头节点，并运行以下命令：

```console
bash run_cluster.sh \
                vllm/vllm-openai \
                ip_of_head_node \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=ip_of_this_node
```

在其余工作节点上，运行以下命令：

```console
bash run_cluster.sh \
                vllm/vllm-openai \
                ip_of_head_node \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=ip_of_this_node
```

然后你将得到一个由**容器**组成的 Ray 集群。请注意，运行这些命令的 shell 必须保持活跃以维持集群。任何 shell 断开都会终止集群。此外，`ip_of_head_node` 参数应为头节点的 IP 地址，且所有工作节点都能访问。每个工作节点的 IP 地址应在 `VLLM_HOST_IP` 环境变量中指定，且每个节点不同。请检查集群的网络配置，确保节点间可以通过指定的 IP 地址通信。

!!! warning
    最佳实践是将 `VLLM_HOST_IP` 设置为 vLLM 集群的私有网络段地址。此处传输的流量未加密，端点间交换的数据格式可能被恶意利用执行任意代码。请确保此网络无法被不受信任的第三方访问。

!!! warning
    由于这是一个由**容器**组成的 Ray 集群，接下来的所有命令都应在**容器**中执行，否则你将在主机上执行命令，而主机未连接到 Ray 集群。要进入容器，可以使用 `docker exec -it node /bin/bash`。

然后，在任意节点上，使用 `docker exec -it node /bin/bash` 进入容器，执行 `ray status` 和 `ray list nodes` 检查 Ray 集群的状态。你应该能看到正确的节点数和 GPU 数。

之后，在任意节点上再次使用 `docker exec -it node /bin/bash` 进入容器。**在容器中**，你可以像所有 GPU 在一个节点上一样正常使用 vLLM：vLLM 将利用 Ray 集群中所有节点的 GPU 资源，因此只需在此节点上运行 `vllm` 命令，而无需在其他节点上运行。通常的做法是将张量并行大小设置为每个节点内的 GPU 数量，流水线并行大小设置为节点数量。例如，如果你有 2 个节点共 16 个 GPU（每节点 8 个 GPU），你可以将张量并行大小设置为 8，流水线并行大小设置为 2：

```console
 vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 8 \
     --pipeline-parallel-size 2
```

你也可以仅使用张量并行而不使用流水线并行，只需将张量并行大小设置为集群中的 GPU 总数。例如，如果你有 2 个节点共 16 个 GPU（每节点 8 个 GPU），你可以将张量并行大小设置为 16：

```console
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16
```

要使张量并行性能高效，应确保节点间通信高效，例如使用高速网络卡（如 Infiniband）。要正确设置集群以使用 Infiniband，可在 `run_cluster.sh` 脚本中追加参数，如 `--privileged -e NCCL_IB_HCA=mlx5`。请联系你的系统管理员以获取更多关于如何设置这些标志的信息。确认 Infiniband 是否正常工作的一种方法是设置环境变量 `NCCL_DEBUG=TRACE` 运行 vLLM，例如 `NCCL_DEBUG=TRACE vllm serve ...`，并检查日志中的 NCCL 版本和使用的网络。如果日志中显示 `[send] via NET/S SOCKET`，表示 NCCL 使用的是原始 TCP Socket，跨节点张量并行的效率不高。如果日志中显示 `[send] via NET/IB/GDRDMA`，表示 NCCL 使用了带 GPU-Direct RDMA 的 Infiniband，效率较高。

!!! warning
    启动 Ray 集群后，最好检查节点间的 GPU-GPU 通信。设置可能较为复杂。请参阅 [sanity check script][troubleshooting-incorrect-hardware-driver] 获取更多信息。如果需要为通信配置设置一些环境变量，可以在 `run_cluster.sh` 脚本中追加，例如 `-e NCCL_SOCKET_IFNAME=eth0`。请注意，在 shell 中设置环境变量（例如 `NCCL_SOCKET_IFNAME=eth0 vllm serve ...`）仅对同一节点内的进程有效，对其他节点的进程无效。创建集群时设置环境变量是推荐的方式。详见 <gh-issue:6803>。

!!! warning
    请确保模型已下载到所有节点（路径相同），或模型下载到所有节点均可访问的分布式文件系统。

    当使用 HuggingFace 仓库 ID 引用模型时，应在 `run_cluster.sh` 脚本中追加你的 HuggingFace 令牌，例如 `-e HF_TOKEN=`。推荐的方式是先下载模型，然后使用路径引用模型。

!!! warning
    如果你持续收到错误消息 `Error: No available node types can fulfill resource request`，但集群中有足够的 GPU，可能是因为你的节点有多个 IP 地址，vLLM 无法找到正确的地址，尤其是在多节点推理时。请确保 vLLM 和 Ray 使用相同的 IP 地址。你可以在 `run_cluster.sh` 脚本中将 `VLLM_HOST_IP` 环境变量设置为正确的 IP 地址（每个节点不同！），并检查 `ray status` 和 `ray list nodes` 以查看 Ray 使用的 IP 地址。详见 <gh-issue:7815>。