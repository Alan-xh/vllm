# --8<-- [start:installation]

vLLM 包含预编译的 C++ 和 CUDA (12.8) 二进制文件。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU：计算能力 7.0 或更高（例如 V100、T4、RTX20xx、A100、L4、H100 等）

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

### 创建一个新的 Python 环境

!!! note
    通过 `conda` 安装的 PyTorch 将静态链接 `NCCL` 库，这可能会导致 vLLM 尝试使用 `NCCL` 时出现问题。详情请参阅 <gh-issue:8420>。

为了获得高性能，vLLM 需要编译许多 CUDA 内核。不幸的是，编译过程会引入与不同 CUDA 版本和 PyTorch 版本的二进制不兼容问题，即使是同一 PyTorch 版本的不同构建配置也可能如此。

因此，建议在**全新**的环境中安装 vLLM。如果您使用的是不同的 CUDA 版本或希望使用现有的 PyTorch 安装，则需要从源代码构建 vLLM。请参阅[下方][build-from-source]以获取更多详情。

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

您可以使用 `pip` 或 `uv pip` 安装 vLLM：

```console
# 安装带 CUDA 12.8 的 vLLM。
# 如果您使用 pip。
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
# 如果您使用 uv。
uv pip install vllm --torch-backend=auto
```

我们建议利用 `uv` 通过 `--torch-backend=auto`（或 `UV_TORCH_BACKEND=auto`）[在运行时自动选择合适的 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)，它会检查已安装的 CUDA 驱动版本。如果需要选择特定后端（例如 `cu126`），请设置 `--torch-backend=cu126`（或 `UV_TORCH_BACKEND=cu126`）。如果此方法无效，请先尝试运行 `uv self update` 更新 `uv`。

!!! note
    NVIDIA Blackwell GPU（B200、GB200）需要至少 CUDA 12.8，因此请确保安装的 PyTorch 轮子版本至少为此版本。PyTorch 本身提供了一个[专用界面](https://pytorch.org/get-started/locally/)，以确定针对特定目标配置运行的适当 pip 命令。

目前，vLLM 的二进制文件默认使用 CUDA 12.8 和公共 PyTorch 发布版本进行编译。我们还提供使用 CUDA 12.6、11.8 和公共 PyTorch 发布版本编译的 vLLM 二进制文件：

```console
# 安装带 CUDA 11.8 的 vLLM。
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=312
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

[](){ #install-the-latest-code }

#### 安装最新代码

LLM 推理是一个快速发展领域，最新代码可能包含尚未发布的错误修复、性能改进和新功能。为了让用户无需等待下一次发布即可尝试最新代码，vLLM 为自 `v0.5.3` 以来的每个提交提供适用于 Linux 的 x86 平台的 CUDA 12 轮子。

##### 使用 `pip` 安装最新代码

```console
pip install -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
```

`--pre` 是 `pip` 考虑预发布版本所必需的。

另一种安装最新代码的方法是使用 `uv`：

```console
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

##### 使用 `pip` 安装特定版本

如果您想访问之前提交的轮子（例如，为了定位行为变化、性能回归），由于 `pip` 的限制，您必须通过在 URL 中嵌入提交哈希来指定轮子文件的完整 URL：

```console
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # 使用主分支的完整提交哈希
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

请注意，这些轮子是使用 Python 3.8 ABI 构建的（有关 ABI 的更多详细信息，请参阅 [PEP 425](https://peps.python.org/pep-0425/)），因此**它们与 Python 3.8 及更高版本兼容**。轮子文件名中的版本字符串（`1.0.0.dev`）只是一个占位符，用于统一 URL，轮子的实际版本包含在轮子元数据中（额外索引 URL 中列出的轮子具有正确的版本）。虽然我们不再支持 Python 3.8（因为 PyTorch 2.5 已取消对 Python 3.8 的支持），但为了保持与之前的轮子名称一致，轮子仍使用 Python 3.8 ABI 构建。

##### 使用 `uv` 安装特定版本

如果您想访问之前提交的轮子（例如，为了定位行为变化、性能回归），您可以在 URL 中指定提交哈希：

```console
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支的完整提交哈希
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

`uv` 方法适用于 vLLM `v0.6.6` 及更高版本，并提供了一个易于记忆的命令。`uv` 的一个独特功能是 `--extra-index-url` 中的包[优先级高于默认索引](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。如果最新的公共发布版本是 `v0.6.6.post1`，`uv` 的行为允许通过指定 `--extra-index-url` 安装早于 `v0.6.6.post1` 的提交。相比之下，`pip` 会将 `--extra-index-url` 和默认索引中的包合并，仅选择最新版本，这使得安装早于发布版本的开发版本变得困难。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

#### 使用仅 Python 构建（无需编译）

如果您只需要更改 Python 代码，可以在不进行编译的情况下构建和安装 vLLM。使用 `pip` 的 [`--editable` 标志](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)，您对代码的更改将在运行 vLLM 时反映出来：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

此命令将执行以下操作：

1. 检查您 vLLM 克隆中的当前分支。
1. 确定主分支中对应的基础提交。
1. 下载基础提交的预构建轮子。
1. 在安装中使用其编译的库。

!!! note
    1. 如果您更改了 C++ 或内核代码，则无法使用仅 Python 构建；否则，您将看到关于库未找到或未定义符号的导入错误。
    2. 如果您重新基于您的开发分支，建议卸载 vLLM 并重新运行上述命令，以确保您的库是最新的。

如果运行上述命令时遇到轮子未找到的错误，可能是因为您基于的主分支提交刚刚合并，轮子正在构建中。在这种情况下，您可以等待大约一小时后重试，或使用 `VLLM_PRECOMPILED_WHEEL_LOCATION` 环境变量手动指定之前的提交进行安装。

```console
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支的完整提交哈希
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
```

您可以在 [install-the-latest-code][install-the-latest-code] 中找到有关 vLLM 轮子的更多信息。

!!! note
    您的源代码的提交 ID 可能与最新的 vLLM 轮子不同，这可能会导致未知错误。
    建议使用与您安装的 vLLM 轮子相同的提交 ID 作为源代码。请参阅 [install-the-latest-code][install-the-latest-code] 获取有关如何安装指定轮子的说明。

#### 完整构建（带编译）

如果您想修改 C++ 或 CUDA 代码，则需要从源代码构建 vLLM。这可能需要几分钟：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

!!! tip
    从源代码构建需要大量编译。如果您反复从源代码构建，缓存编译结果会更有效。

    例如，您可以使用 `conda install ccache` 或 `apt install ccache` 安装 [ccache](https://github.com/ccache/ccache)。
    只要 `which ccache` 命令可以找到 `ccache` 二进制文件，构建系统就会自动使用它。第一次构建后，后续构建将快得多。

    当使用 `ccache` 与 `pip install -e .` 一起时，您应该运行 `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`。这是因为 `pip` 为每次构建创建一个随机名称的新文件夹，阻止 `ccache` 识别正在构建的相同文件。

    [sccache](https://github.com/mozilla/sccache) 与 `ccache` 类似，但具有在远程存储环境中利用缓存的能力。
    以下环境变量可用于配置 vLLM 的 `sccache` 远程：`SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`。我们还建议设置 `SCCACHE_IDLE_TIMEOUT=0`。

##### 使用现有的 PyTorch 安装

在某些情况下，PyTorch 依赖项无法通过 pip 轻松安装，例如：

- 使用 PyTorch 夜间版或自定义 PyTorch 构建来构建 vLLM。
- 在 aarch64 和 CUDA（GH200）上构建 vLLM，PyPI 上没有可用的 PyTorch 轮子。目前，只有 PyTorch 夜间版提供 aarch64 和 CUDA 的轮子。您可以运行 `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124` 来[安装 PyTorch 夜间版](https://pytorch.org/get-started/locally/)，然后在其上构建 vLLM。

使用现有的 PyTorch 安装构建 vLLM：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

##### 使用本地 cutlass 进行编译

目前，在开始构建过程之前，vLLM 会从 GitHub 获取 cutlass 代码。然而，在某些情况下，您可能希望使用本地版本的 cutlass。
为此，您可以设置环境变量 `VLLM_CUTLASS_SRC_DIR` 指向您的本地 cutlass 目录。

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass pip install -e .
```

##### 故障排除

为了避免系统过载，您可以通过环境变量 `MAX_JOBS` 限制同时运行的编译作业数量。例如：

```console
export MAX_JOBS=6
pip install -e .
```

这在您使用较弱的机器进行构建时尤其有用。例如，当使用 WSL 时，它默认[仅分配 50% 的总内存](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings)，因此使用 `export MAX_JOBS=1` 可以避免同时编译多个文件并耗尽内存。
副作用是构建过程会慢得多。

此外，如果您在构建 vLLM 时遇到问题，我们建议使用 NVIDIA PyTorch Docker 镜像。

```console
# 使用 `--ipc=host` 确保共享内存足够大。
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

如果您不想使用 Docker，建议安装完整的 CUDA 工具包。您可以从[官方网站](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装。安装后，将环境变量 `CUDA_HOME` 设置为 CUDA 工具包的安装路径，并确保 `nvcc` 编译器在您的 `PATH` 中，例如：

```console
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

以下是验证 CUDA 工具包是否正确安装的检查：

```console
nvcc --version # 验证 nvcc 是否在您的 PATH 中
${CUDA_HOME}/bin/nvcc --version # 验证 nvcc 是否在您的 CUDA_HOME 中
```

#### 不支持的操作系统构建

vLLM 仅能在 Linux 上完全运行，但为了开发目的，您仍然可以在其他系统（例如 macOS）上构建它，以实现导入和更方便的开发环境。二进制文件将不会被编译，并且在非 Linux 系统上无法工作。

只需在安装前禁用 `VLLM_TARGET_DEVICE` 环境变量：

```console
export VLLM_TARGET_DEVICE=empty
pip install -e .
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

请参阅 [deployment-docker-pre-built-image][deployment-docker-pre-built-image] 获取使用官方 Docker 镜像的说明。

另一种访问最新代码的方法是使用 Docker 镜像：

```console
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # 使用主分支的完整提交哈希
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}
```

这些 Docker 镜像仅用于 CI 和测试，不适合生产使用。它们将在几天后过期。

最新代码可能包含错误且可能不稳定。请谨慎使用。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

请参阅 [deployment-docker-build-image-from-source][deployment-docker-build-image-from-source] 获取构建 Docker 镜像的说明。

## 支持的功能

请参阅 [feature-x-hardware][feature-x-hardware] 兼容性矩阵以获取功能支持信息。
# --8<-- [end:extra-information]