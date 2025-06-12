# --8<-- [start:installation]

本节提供在英特尔 Gaudi 设备上运行 vLLM 的说明。

!!! warning
    该设备没有预构建的轮子或镜像，因此必须从源代码构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：Ubuntu 22.04 LTS
- Python：3.10
- 英特尔 Gaudi 加速器
- 英特尔 Gaudi 软件版本 1.18.0

请按照[Gaudi 安装指南](https://docs.habana.ai/en/latest/Installation_Guide/index.html)中的说明设置执行环境。为了获得最佳性能，请遵循[优化训练平台指南](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html)中描述的方法。

## 配置新环境

### 环境验证

要验证英特尔 Gaudi 软件是否正确安装，请运行以下命令：

```console
hl-smi # 验证 hl-smi 是否在 PATH 中，并且每个 Gaudi 加速器都可见
apt list --installed | grep habana # 验证是否安装了 habanalabs-firmware-tools、habanalabs-graph、habanalabs-rdma-core、habanalabs-thunk 和 habanalabs-container-runtime
pip list | grep habana # 验证是否安装了 habana-torch-plugin、habana-torch-dataloader、habana-pyhlml 和 habana-media-loader
pip list | grep neural # 验证是否安装了 neural_compressor
```

有关更多详细信息，请参阅[英特尔 Gaudi 软件堆栈验证](https://docs.habana.ai/en/latest/Installation_Guide/SW_Verification.html#platform-upgrade)。

### 运行 Docker 镜像

强烈建议使用来自英特尔 Gaudi 镜像库的最新 Docker 镜像。有关更多详细信息，请参阅[英特尔 Gaudi 文档](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers)。

使用以下命令运行 Docker 镜像：

```console
docker pull vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
docker run \
  -it \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
```

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前没有预构建的英特尔 Gaudi 轮子。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

要从源代码构建并安装 vLLM，请运行以下命令：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/hpu.txt
python setup.py develop
```

目前，最新的功能和性能优化在 Gaudi 的 [vLLM 分叉](https://github.com/HabanaAI/vllm-fork) 中开发，我们会定期将它们上游到 vLLM 主仓库。要安装最新的 [HabanaAI/vLLM-fork](https://github.com/HabanaAI/vllm-fork)，请运行以下命令：

```console
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout habana_main
pip install -r requirements/hpu.txt
python setup.py develop
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

目前没有预构建的英特尔 Gaudi 镜像。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```console
docker build -f docker/Dockerfile.hpu -t vllm-hpu-env  .
docker run \
  -it \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --rm vllm-hpu-env
```

!!! tip
    如果您遇到以下错误：`docker: Error response from daemon: Unknown runtime specified habana.`，请参阅[英特尔 Gaudi 软件堆栈和驱动安装](https://docs.habana.ai/en/v1.18.0/Installation_Guide/Bare_Metal_Fresh_OS.html)中的“使用容器安装”部分。确保已安装 `habana-container-runtime` 包，并且 `habana` 容器运行时已注册。

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]

## 支持的功能

- [离线推理][offline-inference]
- 通过 [OpenAI 兼容服务器][openai-compatible-server] 进行在线服务
- HPU 自动检测 - 无需在 vLLM 中手动选择设备
- 针对英特尔 Gaudi 加速器启用的分页 KV 缓存算法
- 英特尔 Gaudi 定制实现的 Paged Attention、KV 缓存操作、预填充注意力、均方根层归一化、旋转位置编码
- 支持多卡推理的张量并行
- 使用 [HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) 进行推理，以加速低批量延迟和吞吐量
- 支持线性偏置的注意力（ALiBi）

## 不支持的功能

- 束搜索
- LoRA 适配器
- 量化
- 预填充分块（混合批量推理）

## 支持的配置

以下配置已验证可在 Gaudi2 设备上正常运行。未列出的配置可能有效，也可能无效。

- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  单卡 HPU，或在 2x 和 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
  在 8x HPU 上使用张量并行，BF16 数据类型，随机或贪婪采样

## 性能调优

### 执行模式

目前，vLLM 在 HPU 上支持四种执行模式，具体取决于选择的 HPU PyTorch 桥接后端（通过 `PT_HPU_LAZY_MODE` 环境变量）以及 `--enforce-eager` 标志。

|   `PT_HPU_LAZY_MODE` |   `enforce_eager` | 执行模式           |
|----------------------|-------------------|--------------------|
|                    0 |                 0 | torch.compile      |
|                    0 |                 1 | PyTorch 急切模式   |
|                    1 |                 0 | HPU Graphs         |
  <figcaption>vLLM 执行模式</figcaption>

!!! warning
    在 1.18.0 版本中，所有使用 `PT_HPU_LAZY_MODE=0` 的模式都属于高度实验性，仅应用于验证功能正确性。其性能将在后续版本中改进。在 1.18.0 版本中，为了获得最佳性能，请使用 HPU Graphs 或 PyTorch 延迟模式。

[](){ #gaudi-bucketing-mechanism }

### 分桶机制

英特尔 Gaudi 加速器在处理固定张量形状的模型时性能最佳。[英特尔 Gaudi 图编译器](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html#graph-compiler-and-runtime)负责生成优化的二进制代码，以在 Gaudi 上实现给定的模型拓扑。在默认配置下，生成的二进制代码可能高度依赖输入和输出张量的形状，并且在遇到相同拓扑中不同形状的张量时可能需要图重新编译。虽然生成的二进制代码能够高效利用 Gaudi，但编译本身可能在端到端执行中引入显著的开销。在动态推理服务场景中，需要尽量减少图编译的次数，并降低在服务器运行时发生图编译的风险。目前，这是通过在 `batch_size` 和 `sequence_length` 两个维度上对模型的前向传播进行“分桶”来实现的。

!!! note
    分桶可以显著减少所需的图数量，但它不处理任何图编译和设备代码生成——这些在预热和 HPU 图捕获阶段完成。

分桶范围由三个参数决定：`min`、`step` 和 `max`。这些参数可以分别为提示（prompt）和解码（decode）阶段以及批次大小和序列长度维度分别设置。这些参数可以在 vLLM 启动时的日志中观察到：

```text
INFO 08-01 21:37:59 hpu_model_runner.py:493] 提示分桶配置（最小值，步长，最大预热）bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-01 21:37:59 hpu_model_runner.py:499] 生成 24 个提示分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-01 21:37:59 hpu_model_runner.py:504] 解码分桶配置（最小值，步长，最大预热）bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-01 21:37:59 hpu_model_runner.py:509] 生成 48 个解码分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
```

`min` 确定分桶的最小值，`step` 确定分桶之间的间隔，`max` 确定分桶的上限。此外，`min` 和 `step` 之间的间隔有特殊处理——`min` 会乘以连续的 2 的幂，直到达到 `step`。我们称这个阶段为“加速阶段”，用于以最小的浪费处理较低的批次大小，同时允许在较大的批次大小上进行更大的填充。

示例（包含加速阶段）：

```text
min = 2, step = 32, max = 64
=> 加速阶段 = (2, 4, 8, 16)
=> 稳定阶段 = (32, 64)
=> 分桶 = 加速阶段 + 稳定阶段 => (2, 4, 8, 16, 32, 64)
```

示例（不包含加速阶段）：

```text
min = 128, step = 128, max = 512
=> 加速阶段 = ""
=> 稳定阶段 = (128, 256, 384, 512)
=> 分桶 = 加速阶段 + 稳定阶段 => (128, 256, 384, 512)
```

在记录的场景中，为提示（预填充）运行生成了 24 个分桶，为解码运行生成了 48 个分桶。每个分桶对应于给定模型的特定张量形状的单独优化设备二进制文件。每当处理一批请求时，会在批次大小和序列长度维度上填充到最小的可能分桶。

!!! warning
    如果请求在任一维度上超过最大分桶大小，将不进行填充处理，其处理可能需要图编译，可能会显著增加端到端延迟。分桶的边界可以通过环境变量进行用户配置，可以增加上界以避免这种情况。

例如，如果一个包含 3 个序列、最大序列长度为 412 的请求进入空闲的 vLLM 服务器，它将被填充并执行为 `(4, 512)` 预填充分桶，因为 `batch_size`（序列数）将被填充到 4（高于 3 的最接近批次大小维度），最大序列长度将被填充到 512（高于 412 的最接近序列长度维度）。在预填充阶段后，它将作为 `(4, 512)` 解码分桶执行，并继续作为该分桶，直到批次维度发生变化（由于请求完成）——在这种情况下，它将变为 `(2, 512)` 分桶，或者上下文长度超过 512 个标记，在这种情况下，它将变为 `(4, 640)` 分桶。

!!! note
    分桶对客户端是透明的——序列长度维度的填充永远不会返回给客户端，批次维度的填充不会创建新请求。

### 预热

预热是一个可选但强烈推荐的步骤，在 vLLM 服务器开始监听之前执行。它为每个分桶使用虚拟数据执行前向传播。目标是预编译所有图表，并在服务器运行时在分桶边界内不产生任何图编译开销。每个预热步骤在 vLLM 启动时都会记录：

```text
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [预热][提示][1/24] batch_size:4 seq_len:1024 free_mem:79.16 GiB
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [预热][提示][2/24] batch_size:4 seq_len:896 free_mem:55.43 GiB
INFO 08-01 22:26:48 hpu_model_runner.py:1066] [预热][提示][3/24] batch_size:4 seq_len:768 free_mem:55.43 GiB
...
INFO 08-01 22:26:59 hpu_model_runner.py:1066] [预热][提示][24/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [预热][解码][1/48] batch_size:4 seq_len:2048 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [预热][解码][2/48] batch_size:4 seq_len:1920 free_mem:55.43 GiB
INFO 08-01 22:27:01 hpu_model_runner.py:1066] [预热][解码][3/48] batch_size:4 seq_len:1792 free_mem:55.43 GiB
...
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [预热][解码][47/48] batch_size:2 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [预热][解码][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
```

此示例使用与[分桶机制][gaudi-bucketing-mechanism]部分相同的分桶。每个输出行对应于执行单个分桶。分桶首次执行时，其图会被编译，并可以在后续重用，跳过进一步的图编译。

!!! tip
    编译所有分桶可能需要一些时间，可以通过设置 `VLLM_SKIP_WARMUP=true` 环境变量关闭。请注意，如果禁用预热，在首次执行某个分桶时可能会面临图编译。在开发中禁用预热是可以的，但在部署中强烈建议启用。

### HPU 图捕获

[HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) 是目前在英特尔 Gaudi 上运行 vLLM 的最优执行方法。启用 HPU 图时，执行图将在预热后提前跟踪（记录），以在推理期间重放，显著减少主机开销。记录可能需要大量内存，这需要在分配 KV 缓存时考虑。启用 HPU 图会影响可用 KV 缓存块的数量，但 vLLM 提供用户可配置的变量来控制内存管理。

当使用 HPU 图时，它们与 KV 缓存共享共同的内存池（“可用内存”），由 `gpu_memory_utilization` 标志（默认值为 `0.9`）确定。在分配 KV 缓存之前，会先加载模型权重，并使用虚拟数据执行模型的前向传播，以估计内存使用量。之后，`gpu_memory_utilization` 标志生效——在其默认值下，会将当时剩余设备内存的 90% 标记为可用。接下来，分配 KV 缓存，预热模型，并捕获 HPU 图。环境变量 `VLLM_GRAPH_RESERVED_MEM` 定义了用于 HPU 图捕获的内存比例。默认值（`VLLM_GRAPH_RESERVED_MEM=0.1`）下，10% 的可用内存将保留用于图捕获（以下简称“可用图内存”），剩余 90% 将用于 KV 缓存。环境变量 `VLLM_GRAPH_PROMPT_RATIO` 确定为提示和解码图保留的可用图内存比例。默认值（`VLLM_GRAPH_PROMPT_RATIO=0.3`）下，两个阶段具有相等的内存约束。较低的值对应于为提示阶段保留的可用图内存较少，例如 `VLLM_GRAPH_PROMPT_RATIO=0.2` 将为提示图保留 20% 的可用图内存，为解码图保留 80% 的可用图内存。

!!! note
    `gpu_memory_utilization` 并不对应于 HPU 的绝对内存使用量。它指定了加载模型并执行概要分析运行后的内存余量。如果设备有 100 GiB 的总内存，且在加载模型权重和执行概要分析运行后剩余 50 GiB 的空闲内存，`gpu_memory_utilization` 在默认值下将标记 50 GiB 的 90% 为可用，留出 5 GiB 的余量，不考虑总设备内存。

用户还可以分别为提示和解码阶段配置 HPU 图的捕获策略。策略会影响图捕获的顺序。实现了两种策略：

- `max_bs` - 图捕获队列将按批次大小降序排序。批次大小相等的桶按序列长度升序排序（例如 `(64, 128)`、`(64, 256)`、`(32, 128)`、`(32, 256)`、`(1, 128)`、`(1,256)`），解码的默认策略
- `min_tokens` - 图捕获队列将按每个图处理的标记数（`batch_size*sequence_length`）升序排序，提示的默认策略

当有大量请求待处理时，vLLM 调度器将尝试尽快填满解码的最大批次大小。当一个请求完成时，解码批次大小会减少。发生这种情况时，vLLM 将尝试为等待队列中的请求调度预填充迭代，以将解码批次大小恢复到之前的状态。这意味着在满载场景中，解码批次大小通常处于最大值，这使得捕获大批量 HPU 图变得至关重要，如 `max_bs` 策略所示。另一方面，预填充通常以非常低的批次大小（1-4）执行，这反映在 `min_tokens` 策略中。

!!! note
    `VLLM_GRAPH_PROMPT_RATIO` 不会为每个阶段（预填充和解码）的图设置硬性内存限制。vLLM 将首先尝试用尽可用预填充图内存（可用图内存 * `VLLM_GRAPH_PROMPT_RATIO`）来捕获预填充 HPU 图，然后对解码图和可用解码图内存池执行相同操作。如果一个阶段已完全捕获，且可用图内存池中仍有未使用的内存，vLLM 将尝试为另一阶段进一步捕获图，直到无法在不超出保留内存池的情况下捕获更多 HPU 图。以下示例中可以观察到该机制的行为。

每个描述的步骤都由 vLLM 服务器记录，如下所示（负值表示内存被释放）：

```text
INFO 08-02 17:37:44 hpu_model_runner.py:493] 提示分桶配置（最小值，步长，最大预热）bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-02 17:37:44 hpu_model_runner.py:499] 生成 24 个提示分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-02 17:37:44 hpu_model_runner.py:504] 解码分桶配置（最小值，步长，最大预热）bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-02 17:37:44 hpu_model_runner.py:509] 生成 48 个解码分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:37:52 hpu_model_runner.py:430] 在 hpu:0 上预加载模型权重占用了 14.97 GiB 的设备内存（14.97 GiB/94.62 GiB 已使用）和 2.95 GiB 的主机内存（475.2 GiB/1007 GiB 已使用）
INFO 08-02 17:37:52 hpu_model_runner.py:438] HPU 图包装占用了 0 B 的设备内存（14.97 GiB/94.62 GiB 已使用）和 -252 KiB 的主机内存（475.2 GiB/1007 GiB 已使用）
INFO 08-02 17:37:52 hpu_model_runner.py:442] 加载模型权重总共占用了 14.97 GiB 的设备内存（14.97 GiB/94.62 GiB 已使用）和 2.95 GiB 的主机内存（475.2 GiB/1007 GiB 已使用）
INFO 08-02 17:37:54 hpu_worker.py:134] 模型概要分析运行占用了 504 MiB 的设备内存（15.46 GiB/94.62 GiB 已使用）和 180.9 MiB 的主机内存（475.4 GiB/1007 GiB 已使用）
INFO 08-02 17:37:54 hpu_worker.py:158] 空闲设备内存：79.16 GiB，可用内存 39.58 GiB（gpu_memory_utilization=0.5），为 HPU 图保留 15.83 GiB（VLLM_GRAPH_RESERVED_MEM=0.4），为 KV 缓存保留 23.75 GiB
INFO 08-02 17:37:54 hpu_executor.py:85] HPU 块数：1519，CPU 块数：0
INFO 08-02 17:37:54 hpu_worker.py:190] 初始化缓存引擎占用了 23.73 GiB 的设备内存（39.2 GiB/94.62 GiB 已使用）和 -1.238 MiB 的主机内存（475.4 GiB/1007 GiB 已使用）
INFO 08-02 17:37:54 hpu_model_runner.py:1066] [预热][提示][1/24] batch_size:4 seq_len:1024 free_mem:55.43 GiB
...
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [预热][解码][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-02 17:38:22 hpu_model_runner.py:1159] 使用 15.85 GiB/55.43 GiB 的空闲设备内存用于 HPU 图，提示 7.923 GiB，解码 7.923 GiB（VLLM_GRAPH_PROMPT_RATIO=0.3）
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [预热][图/提示][1/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
...
INFO 08-02 17:38:26 hpu_model_runner.py:1066] [预热][图/提示][11/24] batch_size:1 seq_len:896 free_mem:48.77 GiB
INFO 08-02 17:38:27 hpu_model_runner.py:1066] [预热][图/解码][1/48] batch_size:4 seq_len:128 free_mem:47.51 GiB
...
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [预热][图/解码][48/48] batch_size:1 seq_len:2048 free_mem:47.35 GiB
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [预热][图/提示][12/24] batch_size:4 seq_len:256 free_mem:47.35 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [预热][图/提示][13/24] batch_size:2 seq_len:512 free_mem:45.91 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [预热][图/提示][14/24] batch_size:1 seq_len:1024 free_mem:44.48 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1066] [预热][图/提示][15/24] batch_size:2 seq_len:640 free_mem:43.03 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1128] 图/提示已捕获：15（62.5%）已用内存：14.03 GiB 分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (4, 128), (4, 256)]
INFO 08-02 17:38:43 hpu_model_runner.py:1128] 图/解码已捕获：48（100.0%）已用内存：161.9 MiB 分桶：[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:38:43 hpu_model_runner.py:1206] 预热在 49 秒内完成，分配了 14.19 GiB 的设备内存
INFO 08-02 17:38:43 hpu_executor.py:91] 初始化缓存引擎占用了 37.92 GiB 的设备内存（53.39 GiB/94.62 GiB 已使用）和 57.86 MiB 的主机内存（475.4 GiB/1007 GiB 已使用）
```

### 推荐的 vLLM 参数

- 我们建议在 Gaudi 2 上以 `block_size` 为 128 进行推理，适用于 BF16 数据类型。使用默认值（16、32）可能由于矩阵乘法引擎利用不足而导致性能次优（参见 [Gaudi 架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)）。
- 对于 Llama 7B 的最大吞吐量，我们建议在启用 HPU 图的情况下，以 128 或 256 的批次大小和 2048 的最大上下文长度运行。如果遇到内存不足问题，请参阅故障排除部分。

### 环境变量

**诊断和分析旋钮：**

- `VLLM_PROFILER_ENABLED`：如果为 `true`，启用高级分析器。生成的 JSON 跟踪可在 [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer) 中查看。默认为 `false`。
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`：如果为 `true`，在发生图编译时记录每个 vLLM 引擎步骤的图编译。建议与 `PT_HPU_METRICS_GC_DETAILS=1` 一起使用。默认为 `false`。
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL`：如果为 `true`，即使没有发生图编译，也始终记录每个 vLLM 引擎步骤的图编译。默认为 `false`。
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`：如果为 `true`，在发生 CPU 回退时记录每个 vLLM 引擎步骤的 CPU 回退。默认为 `false`。
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`：如果为 `true`，即使没有发生 CPU 回退，也始终记录每个 vLLM 引擎步骤的 CPU 回退。默认为 `false`。

**性能调优旋钮：**

- `VLLM_SKIP_WARMUP`：如果为 `true`，将跳过预热，默认为 `false`
- `VLLM_GRAPH_RESERVED_MEM`：用于 HPU 图捕获的内存百分比，默认为 `0.1`
- `VLLM_GRAPH_PROMPT_RATIO`：为提示图保留的可用图内存百分比，默认为 `0.3`
- `VLLM_GRAPH_PROMPT_STRATEGY`：确定提示图捕获顺序的策略，`min_tokens` 或 `max_bs`，默认为 `min_tokens`
- `VLLM_GRAPH_DECODE_STRATEGY`：确定解码图捕获顺序的策略，`min_tokens` 或 `max_bs`，默认为 `max_bs`
- `VLLM_{phase}_{dim}_BUCKET_{param}` - 配置分桶机制范围的 12 个环境变量集合
  * `{phase}` 为 `PROMPT` 或 `DECODE`
  * `{dim}` 为 `BS`、 `SEQ` 或 `BLOCK`
  * `{param}` 为 `MIN`、 `STEP` 或 `MAX`
  * 默认值：
    - 提示：
      - 批次大小最小值（`VLLM_PROMPT_BS_BUCKET_MIN`）：`1`
      - 批次大小步长（`VLLM_PROMPT_BS_BUCKET_STEP`）：`min(max_num_seqs, 32)`
      - 批次大小最大值（`VLLM_PROMPT_BS_BUCKET_MAX`）：`min(max_num_seqs, 64)`
      - 序列长度最小值（`VLLM_PROMPT_SEQ_BUCKET_MIN`）：`block_size`
      - 序列长度步长（`VLLM_PROMPT_SEQ_BUCKET_STEP`）：`block_size`
      - 序列长度最大值（`VLLM_PROMPT_SEQ_BUCKET_MAX`）：`max_model_len`
    - 解码：
      - 批次大小最小值（`VLLM_DECODE_BS_BUCKET_MIN`）：`1`
      - 批次大小步长（`VLLM_DECODE_BS_BUCKET_STEP`）：`min(max_num_seqs, 32)`
      - 批次大小最大值（`VLLM_DECODE_BS_BUCKET_MAX`）：`max_num_seqs`
      - 序列长度最小值（`VLLM_DECODE_BLOCK_BUCKET_MIN`）：`block_size`
      - 序列长度步长（`VLLM_DECODE_BLOCK_BUCKET_STEP`）：`block_size`
      - 序列长度最大值（`VLLM_DECODE_BLOCK_BUCKET_MAX`）：`max(128, (max_num_seqs*max_model_len)/block_size)`

此外，还有影响 vLLM 执行的 HPU PyTorch 桥接环境变量：

- `PT_HPU_LAZY_MODE`：如果为 `0`，将使用 Gaudi 的 PyTorch 急切后端；如果为 `1`，将使用 Gaudi 的 PyTorch 延迟后端。默认为 `1`。
- `PT_HPU_ENABLE_LAZY_COLLECTIVES`：对于使用 HPU 图的张量并行推理，必须为 `true`

## 故障排除：调整 HPU 图

如果您遇到设备内存不足问题或希望尝试以更高批次大小进行推理，请尝试通过以下方法调整 HPU 图：

- 调整 `gpu_memory_utilization` 旋钮。这将减少 KV 缓存的分配，为捕获更大批次大小的图留出一些空间。默认情况下，`gpu_memory_utilization` 设置为 0.9。它尝试在短时间分析运行后分配大约 90% 的 HBM 用于 KV 缓存。请注意，减少此值会减少可用 KV 缓存块的数量，从而减少可同时处理的有效最大标记数。
- 如果此方法效率不高，您可以完全禁用 `HPUGraph`。禁用 HPU 图后，您将以较低批次的延迟和吞吐量换取在较高批次上可能更高的吞吐量。您可以通过为服务器添加 `--enforce-eager` 标志（用于在线服务）或将 `enforce_eager=True` 参数传递给 LLM 构造函数（用于离线推理）来实现这一点。

# --8<-- [end:extra-information]
