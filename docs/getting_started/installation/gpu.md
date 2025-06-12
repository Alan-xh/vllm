# GPU

vLLM 是一个 Python 库，支持以下 GPU 变体。选择您的 GPU 类型以查看特定于供应商的说明：

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:installation"

## 要求

- 操作系统：Linux
- Python：3.9 -- 3.12

!!! note
    vLLM 不原生支持 Windows。要在 Windows 上运行 vLLM，您可以使用带有兼容 Linux 发行版的 Windows 子系统（WSL），或使用一些社区维护的分支，例如 [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)。

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:requirements"

## 使用 Python 进行设置

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:create-a-new-python-environment"

=== "AMD ROCm"

    关于为此设备创建新的 Python 环境，没有额外信息。

=== "Intel XPU"

    关于为此设备创建新的 Python 环境，没有额外信息。

### 预构建轮子

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-wheels"

[](){ #build-from-source }

### 从源代码构建轮子

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-wheel-from-source"

## 使用 Docker 进行设置

### 预构建镜像

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-images"

### 从源代码构建镜像

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-image-from-source"

## 支持的功能

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:supported-features"