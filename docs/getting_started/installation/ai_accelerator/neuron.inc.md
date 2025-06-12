# --8<-- [start:installation]

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) 是用于在 AWS Inferentia 和 AWS Trainium 驱动的 Amazon EC2 实例和 UltraServers（Inf1、Inf2、Trn1、Trn2 和 Trn2 UltraServer）上运行深度学习和生成式 AI 工作负载的软件开发工具包（SDK）。Trainium 和 Inferentia 都由完全独立的异构计算单元 NeuronCores 提供支持。本节描述如何设置环境以在 Neuron 上运行 vLLM。

!!! warning
    该设备没有预构建的轮子（wheels）或镜像，因此必须从源代码构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：Linux
- Python：3.9 或更高版本
- Pytorch：2.5/2.6
- 加速器：NeuronCore-v2（在 trn1/inf2 芯片中）或 NeuronCore-v3（在 trn2 芯片中）
- AWS Neuron SDK：2.23

## 配置新环境

### 启动 Trn1/Trn2/Inf2 实例并验证 Neuron 依赖

启动带有预装 Neuron 依赖的 Trainium 或 Inferentia 实例的最简单方法是按照此[快速入门指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami)使用 Neuron 深度学习 AMI（Amazon 机器镜像）。

- 启动实例后，按照[连接到您的实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)中的说明连接到实例。
- 进入实例后，通过运行以下命令激活预装的推理虚拟环境：
```console
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

有关替代设置说明（包括使用 Docker 和手动安装依赖项），请参阅 [NxD 推理设置指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html)。

!!! note
    NxD 推理是 Neuron 上运行推理的默认推荐后端。如果您希望使用旧版 [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) 库，请参阅 [Transformers NeuronX 设置](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/setup/index.html)。

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前没有预构建的 Neuron 轮子（wheels）。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

#### 从源代码安装 vLLM

按以下方式安装 vLLM：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -U -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

AWS Neuron 维护了一个 [vLLM 的 Github 分支](https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2)，地址为 [https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2](https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2)，其中包含了 vLLM V0 之外的多个功能。请使用 AWS 分支以获取以下功能：

- Llama-3.2 多模态支持
- 多节点分布式推理

有关更多详情和使用示例，请参阅 [vLLM 用户指南（NxD 推理）](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)。

要安装 AWS Neuron 分支，请运行以下命令：

```console
git clone -b neuron-2.23-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
cd upstreaming-to-vllm
pip install -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

请注意，AWS Neuron 分支仅用于支持 Neuron 硬件；未测试与其他硬件的兼容性。

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

目前没有预构建的 Neuron 镜像。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

有关构建 Docker 镜像的说明，请参阅 [部署 Docker 构建镜像从源代码][deployment-docker-build-image-from-source]。

请确保使用 <gh-file:docker/Dockerfile.neuron> 替代默认的 Dockerfile。

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]

[](){ #feature-support-through-nxd-inference-backend }
### 通过 NxD 推理后端支持的功能

当前 vLLM 和 Neuron 的集成依赖于 `neuronx-distributed-inference`（首选）或 `transformers-neuronx` 后端来执行大部分核心工作，包括 PyTorch 模型初始化、编译和运行时执行。因此，[Neuron 支持的大多数功能](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html) 也可以通过 vLLM 集成使用。

要通过 vLLM 入口配置 NxD 推理功能，请使用 `override_neuron_config` 设置。以字典形式（或在从 CLI 启动 vLLM 时使用 JSON 对象）提供要覆盖的配置。例如，要禁用自动分桶，请包含：
```console
override_neuron_config={
    "enable_bucketing":False,
}
```
或在从 CLI 启动 vLLM 时，传递：
```console
--override-neuron-config "{\"enable_bucketing\":false}"
```

或者，用户可以直接调用 NxDI 库来跟踪和编译模型，然后通过 `NEURON_COMPILED_ARTIFACTS` 环境变量在 vLLM 中加载预编译的工件以运行推理工作负载。

### 已知限制

- EAGLE 推测解码：NxD 推理要求 EAGLE 草稿检查点包括目标模型的 LM 头部权重。请参阅此[指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html#eagle-checkpoint-compatibility) 以了解如何转换预训练 EAGLE 模型检查点以与 NxDI 兼容。
- 量化：vLLM 中的原生量化流程在 NxD 推理上支持不佳。建议按照此[Neuron 量化指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html) 使用 NxD 推理量化和编译模型，然后将编译后的工件加载到 vLLM 中。
- 多 LoRA 服务：NxD 推理仅支持在服务器启动时加载 LoRA 适配器。目前不支持运行时动态加载 LoRA 适配器。请参阅[多 LoRA 示例](https://github.com/aws-neuron/upstreaming-to-vllm/blob/neuron-2.23-vllm-v0.7.2/examples/offline_inference/neuron_multi_lora.py)。
- 多模态支持：多模态支持仅通过 AWS Neuron 分支提供。此功能尚未上行到 vLLM 主分支，因为 NxD 推理目前依赖于对核心 vLLM 逻辑的某些调整来支持此功能。
- 多节点支持：跨多个 Trainium/Inferentia 实例的分布式推理仅在 AWS Neuron 分支上支持。请参阅此[多节点示例](https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2/examples/neuron/multi_node) 以运行。请注意，张量并行（跨 NeuronCores 的分布式推理）在 vLLM 主分支中可用。
- 推测解码中的已知边缘案例错误：在推测解码中，当序列长度接近最大模型长度时（例如，请求最大令牌数达到最大模型长度并忽略 eos），可能会出现边缘案例失败。在这种情况下，vLLM 可能会尝试分配额外的块以确保有足够的内存用于前瞻槽，但由于对分页注意力的支持不足，没有额外的 Neuron 块可供 vLLM 分配。AWS Neuron 分支中实现了终止前一迭代的解决方法，但由于修改了核心 vLLM 逻辑，尚未上行到 vLLM 主分支。

### 环境变量
- `NEURON_COMPILED_ARTIFACTS`：将此环境变量设置为预编译模型工件的目录，以避免服务器初始化时的编译时间。如果未设置此变量，Neuron 模块将执行编译并将工件保存在模型路径下的 `neuron-compiled-artifacts/{unique_hash}/` 子目录中。如果设置了此环境变量，但目录不存在或内容无效，Neuron 将回退到新的编译并将工件存储在指定的路径下。
- `NEURON_CONTEXT_LENGTH_BUCKETS`：上下文编码的分桶大小。（仅适用于 `transformers-neuronx` 后端）。
- `NEURON_TOKEN_GEN_BUCKETS`：令牌生成的分桶大小。（仅适用于 `transformers-neuronx` 后端）。

# --8<-- [end:extra-information]