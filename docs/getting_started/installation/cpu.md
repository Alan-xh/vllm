# CPU

vLLM 是一个支持以下 CPU 变体的 Python 库。选择您的 CPU 类型以查看特定于供应商的说明：

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:installation"

=== "Apple 硅"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:installation"

## 要求

- Python: 3.9 -- 3.12

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:requirements"

=== "Apple 硅"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:requirements"

## 使用 Python 设置

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### 预构建轮子

目前没有预构建的 CPU 轮子。

### 从源代码构建轮子

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:build-wheel-from-source"

=== "Apple 硅"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:build-wheel-from-source"

## 使用 Docker 设置

### 预构建镜像

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:pre-built-images"

### 从源代码构建镜像

```console
$ docker build -f docker/Dockerfile.cpu --tag vllm-cpu-env --target vllm-openai .

# 启动 OpenAI 服务器 
$ docker run --rm \
             --privileged=true \
             --shm-size=4g \
             -p 8000:8000 \
             -e VLLM_CPU_KVCACHE_SPACE=<KV 缓存空间> \
             -e VLLM_CPU_OMP_THREADS_BIND=<用于推理的 CPU 核心> \
             vllm-cpu-env \
             --model=meta-llama/Llama-3.2-1B-Instruct \
             --dtype=bfloat16 \
             其他 vLLM OpenAI 服务器参数
```

!!! tip
    对于 ARM 或 Apple 硅，请使用 `docker/Dockerfile.arm`

!!! tip
    对于 IBM Z (s390x)，请使用 `docker/Dockerfile.s390x`，并在 `docker run` 中使用标志 `--dtype float`

## 支持的功能

vLLM CPU 后端支持以下 vLLM 功能：

- 张量并行
- 模型量化（`INT8 W8A8, AWQ, GPTQ`）
- 分块预填充
- 前缀缓存
- FP8-E5M2 KV 缓存

## 相关运行时环境变量

- `VLLM_CPU_KVCACHE_SPACE`：指定 KV 缓存大小（例如，`VLLM_CPU_KVCACHE_SPACE=40` 表示 40 GiB 的 KV 缓存空间），更大的设置将允许 vLLM 并行运行更多请求。该参数应根据硬件配置和用户的内存管理模式进行设置。默认值为 `0`。
- `VLLM_CPU_OMP_THREADS_BIND`：指定专用于 OpenMP 线程的 CPU 核心。例如，`VLLM_CPU_OMP_THREADS_BIND=0-31` 表示 32 个 OpenMP 线程绑定在 0-31 CPU 核心上。`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` 表示有两个张量并行进程，rank0 的 32 个 OpenMP 线程绑定在 0-31 CPU 核心上，rank1 的 OpenMP 线程绑定在 32-63 CPU 核心上。设置为 `auto` 时，每个 rank 的 OpenMP 线程绑定到每个 NUMA 节点的 CPU 核心上。设置为 `all` 时，每个 rank 的 OpenMP 线程使用系统上所有可用的 CPU 核心。默认值为 `auto`。
- `VLLM_CPU_NUM_OF_RESERVED_CPU`：指定不专用于每个 rank 的 OpenMP 线程的 CPU 核心数量。该变量仅在 `VLLM_CPU_OMP_THREADS_BIND` 设置为 `auto` 时生效。默认值为 `0`。
- `VLLM_CPU_MOE_PREPACK`：是否为 MoE 层使用预打包。这将传递给 `ipex.llm.modules.GatedMLPMOE`。默认值为 `1`（True）。在不支持的 CPU 上，您可能需要将其设置为 `0`（False）。

## 性能提示

- 我们强烈建议使用 TCMalloc 以获得高性能内存分配和更好的缓存局部性。例如，在 Ubuntu 22.4 上，您可以运行：

```console
sudo apt-get install libtcmalloc-minimal4 # 安装 TCMalloc 库
find / -name *libtcmalloc* # 查找动态链接库路径
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # 将库添加到 LD_PRELOAD
python examples/offline_inference/basic/basic.py # 运行 vLLM
```

- 在使用在线服务时，建议为服务框架预留 1-2 个 CPU 核心，以避免 CPU 超额认购。例如，在具有 32 个物理 CPU 核心的平台上，为框架预留 CPU 30 和 31，并将 CPU 0-29 用于 OpenMP：

```console
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-29
vllm serve facebook/opt-125m
```

 或使用默认的自动线程绑定：

```console
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=2
vllm serve facebook/opt-125m
```

- 如果在具有超线程的机器上使用 vLLM CPU 后端，建议使用 `VLLM_CPU_OMP_THREADS_BIND` 仅在每个物理 CPU 核心上绑定一个 OpenMP 线程，或使用默认的自动线程绑定功能。在具有 16 个逻辑 CPU 核心 / 8 个物理 CPU 核心的超线程启用平台上：

```console
$ lscpu -e # 检查逻辑 CPU 核心与物理 CPU 核心的映射

# “CPU”列表示逻辑 CPU 核心 ID，“CORE”列表示物理核心 ID。在此平台上，两个逻辑核心共享一个物理核心。
CPU NODE SOCKET CORE L1d:L1i:L2:L0 ONLINE    MAXMHZ   MINMHZ      MHZ
0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000

# 在此平台上，建议仅将 OpenMP 线程绑定在逻辑 CPU 核心 0-7 或 8-15 上
$ export VLLM_CPU_OMP_THREADS_BIND=0-7
$ python examples/offline_inference/basic/basic.py
```

- 如果在具有 NUMA 的多插槽机器上使用 vLLM CPU 后端，请注意使用 `VLLM_CPU_OMP_THREADS_BIND` 设置 CPU 核心，以避免跨 NUMA 节点内存访问。

## 其他注意事项

- CPU 后端与 GPU 后端有显著差异，因为 vLLM 架构最初是为 GPU 使用优化的。需要进行多项优化以提升其性能。

- 将 HTTP 服务组件与推理组件解耦。在 GPU 后端配置中，HTTP 服务和分词任务在 CPU 上运行，而推理在 GPU 上运行，通常不会出现问题。然而，在基于 CPU 的设置中，HTTP 服务和分词可能导致显著的上下文切换和缓存效率降低。因此，强烈建议将这两个组件分开以提高性能。

- 在启用 NUMA 的基于 CPU 的设置中，内存访问性能可能会受到 [拓扑](https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/performance_tuning/tuning_guide.md#non-uniform-memory-access-numa) 的显著影响。对于 NUMA 架构，张量并行是提高性能的一个选项。

  - 张量并行支持服务和离线推理。通常，每个 NUMA 节点被视为一张 GPU 卡。以下是启用张量并行 = 2 的服务示例脚本：

    ```console
    VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-31|32-63" vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp
    ```

    或使用默认的自动线程绑定：

    ```console
    VLLM_CPU_KVCACHE_SPACE=40 vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp
    ```

  - 对于 `VLLM_CPU_OMP_THREADS_BIND` 中的每个线程 ID 列表，用户应确保列表中的线程属于同一个 NUMA 节点。

  - 同时，用户还应注意每个 NUMA 节点的内存容量。每个 TP rank 的内存使用量是 `权重分片大小` 和 `VLLM_CPU_KVCACHE_SPACE` 的总和，如果超过单个 NUMA 节点的容量，TP 工作进程将因内存不足而被终止。