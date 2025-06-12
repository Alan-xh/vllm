# --8<-- [start:installation]

vLLM 支持使用 ROCm 6.3 的 AMD GPU。

!!! warning
    此设备没有预构建的轮子（wheels），因此您必须使用预构建的 Docker 镜像或从源代码构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU：MI200s (gfx90a)、MI300 (gfx942)、Radeon RX 7900 系列 (gfx1100/1101)、Radeon RX 9000 系列 (gfx1200/1201)
- ROCm 6.3

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前没有预构建的 ROCm 轮子。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

0. 安装前提条件（如果您已经在安装了以下内容的 Docker 或环境中，则可跳过）：

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    安装 PyTorch 时，您可以从一个全新的 Docker 镜像开始，例如 `rocm/pytorch:rocm6.3_ubuntu24.04_pytorch_release_2.4.0` 或 `rocm/pytorch-nightly`。如果您使用的是 Docker 镜像，可以直接跳到第 3 步。

    或者，您可以使用 PyTorch 轮子安装 PyTorch。请参考 PyTorch [安装指南](https://pytorch.org/get-started/locally/)。示例：

    ```console
    # 安装 PyTorch
    $ pip uninstall torch -y
    $ pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly-rocm6.3
    ```

1. 安装 [ROCm 的 Triton Flash Attention](https://github.com/ROCm/triton)

    按照 [ROCm/triton](https://github.com/ROCm/triton/blob/triton-mlir/README.md) 的说明安装 ROCm 的 Triton Flash Attention（默认使用 triton-mlir 分支）：

    ```console
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/OpenAI/triton.git
    cd triton
    git checkout e5be006
    cd python
    pip3 install .
    cd ../..
    ```

    !!! note
        如果在构建 Triton 时遇到与下载包相关的 HTTP 问题，请重试，因为 HTTP 错误是间歇性的。

2. （可选）如果选择使用 CK Flash Attention，可以安装 [ROCm 的 Flash Attention](https://github.com/ROCm/flash-attention)

    按照 [ROCm/flash-attention](https://github.com/ROCm/flash-attention#amd-rocm-support) 的说明安装 ROCm 的 Flash Attention（v2.7.2）。或者，可以在发布版本中访问专为 vLLM 使用的轮子。

    例如，对于 ROCm 6.3，假设您的 gfx 架构为 `gfx90a`。要获取您的 gfx 架构，请运行 `rocminfo |grep gfx`。

    ```console
    git clone https://github.com/ROCm/flash-attention.git
    cd flash-attention
    git checkout b7d29fb
    git submodule update --init
    GPU_ARCHS="gfx90a" python3 setup.py install
    cd ..
    ```

    !!! note
        编译 flash-attention-2 时可能需要将 "ninja" 版本降级到 1.10（例如 `pip install ninja==1.10.2.4`）。

3. 如果选择自行构建 AITER 以使用特定分支或提交，可以按照以下步骤构建 AITER：

    ```console
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        您需要根据需要配置 `$AITER_BRANCH_OR_COMMIT`。

4. 构建 vLLM。例如，ROCm 6.3 上的 vLLM 可以通过以下步骤构建：

    ```bash
    pip install --upgrade pip

    # 构建并安装 AMD SMI
    pip install /opt/rocm/share/amd_smi

    # 安装依赖
    pip install --upgrade numba \
        scipy \
        huggingface-hub[cli,hf_transfer] \
        setuptools_scm
    pip install "numpy<2"
    pip install -r requirements-rocm.txt

    # 为 MI210/MI250/MI300 构建 vLLM
    export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
    python3 setup.py develop
    ```

    此过程可能需要 5-10 分钟。目前，`pip install .` 不适用于 ROCm 安装。

    !!! tip
        - 默认使用 Triton Flash Attention。为基准测试目的，建议在收集性能数据之前运行预热步骤。
        - Triton Flash Attention 目前不支持滑动窗口注意力。如果使用半精度，请使用 CK Flash Attention 以支持滑动窗口。
        - 要使用 CK Flash Attention 或 PyTorch 原生注意力，请使用以下标志 `export VLLM_USE_TRITON_FLASH_ATTN=0` 关闭 Triton Flash Attention。
        - 理想情况下，PyTorch 的 ROCm 版本应与 ROCm 驱动程序版本匹配。

!!! tip
    - 对于 MI300x (gfx942) 用户，为了获得最佳性能，请参阅 [MI300x 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) 以获取系统和工作流级别的性能优化和调优建议。
      对于 vLLM，请参阅 [vLLM 性能优化](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#vllm-performance-optimization)。

## 使用 Docker 进行设置（推荐）

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

[AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) 提供了一个预构建且优化的 Docker 镜像，专为在 AMD Instinct™ MI300X 加速器上验证推理性能而设计。

!!! tip
    请查看 [AMD Instinct MI300X 上的 LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html) 以获取有关如何使用此预构建 Docker 镜像的说明。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

从源代码构建 Docker 镜像是使用 ROCm 的 vLLM 的推荐方式。

#### （可选）构建包含 ROCm 软件栈的镜像

从 <gh-file:docker/Dockerfile.rocm_base> 构建一个 Docker 镜像，设置 vLLM 所需的 ROCm 软件栈。
**此步骤为可选步骤，因为 rocm_base 镜像通常已预构建并存储在 [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) 的 `rocm/vllm-dev:base` 标签下，以提升用户体验。**
如果您选择自行构建此 rocm_base 镜像，步骤如下。

用户必须使用 buildkit 启动 Docker 构建。用户可以在调用 Docker 构建命令时将 `DOCKER_BUILDKIT=1` 设置为环境变量，或者在 Docker 守护进程配置 `/etc/docker/daemon.json` 中设置 buildkit 并重启守护进程，配置如下：

```console
{
    "features": {
        "buildkit": true
    }
}
```

为 MI200 和 MI300 系列在 ROCm 6.3 上构建 vLLM，可以使用默认设置：

```console
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.rocm_base \
    -t rocm/vllm-dev:base .
```

#### 构建包含 vLLM 的镜像

首先，从 <gh-file:docker/Dockerfile.rocm> 构建一个 Docker 镜像，并从该镜像启动一个 Docker 容器。
用户必须使用 buildkit 启动 Docker 构建。用户可以在调用 Docker 构建命令时将 `DOCKER_BUILDKIT=1` 设置为环境变量，或者在 Docker 守护进程配置 `/etc/docker/daemon.json` 中设置 buildkit 并重启守护进程，配置如下：

```console
{
    "features": {
        "buildkit": true
    }
}
```

<gh-file:docker/Dockerfile.rocm> 默认使用 ROCm 6.3，但也支持 ROCm 5.7、6.0、6.1 和 6.2（在较旧的 vLLM 分支中）。
它提供了通过以下参数自定义 Docker 镜像构建的灵活性：

- `BASE_IMAGE`：指定运行 `docker build` 时使用的基本镜像。默认值 `rocm/vllm-dev:base` 是由 AMD 发布和维护的镜像，使用 <gh-file:docker/Dockerfile.rocm_base> 构建。
- `ARG_PYTORCH_ROCM_ARCH`：允许覆盖基本 Docker 镜像中的 gfx 架构值。

这些值可以在运行 `docker build` 时通过 `--build-arg` 选项传递。

为 MI200 和 MI300 系列在 ROCm 6.3 上构建 vLLM，可以使用默认设置：

```console
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm .
```

为 Radeon RX7900 系列 (gfx1100) 在 ROCm 6.3 上构建 vLLM，应选择替代的基本镜像：

```console
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE="rocm/vllm-dev:navi_base" \
    -f docker/Dockerfile.rocm \
    -t vllm-rocm \
    .
```

要运行上述 Docker 镜像 `vllm-rocm`，请使用以下命令：

```console
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```

其中 `<path/to/model>` 是模型存储的位置，例如 llama2 或 llama3 模型的权重。

## 支持的功能

请参阅 [功能与硬件兼容性矩阵][feature-x-hardware] 以获取功能支持信息。
# --8<-- [end:extra-information]