---
title: 使用 Docker
---
[](){ #deployment-docker }

[](){ #deployment-docker-pre-built-image }

## 使用 vLLM 的官方 Docker 镜像

vLLM 提供了用于部署的官方 Docker 镜像。
该镜像可用于运行 OpenAI 兼容的服务器，并可在 Docker Hub 上以 [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags) 的形式获取。

```console
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

此镜像也可以与其他容器引擎（如 [Podman](https://podman.io/)）一起使用。

```console
podman run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-v0.1
```

您可以在镜像标签（`vllm/vllm-openai:latest`）后添加任何其他需要的 [engine-args][engine-args] 参数。

!!! note
    您可以使用 `ipc=host` 标志或 `--shm-size` 标志来允许容器访问主机的共享内存。vLLM 使用 PyTorch，而 PyTorch 在底层使用共享内存来在进程间共享数据，特别是在张量并行推理时。

!!! note
    为了避免许可问题（例如 <gh-issue:8030>），未包含可选依赖项。

    如果您需要使用这些依赖项（在接受许可条款后），
    可以在基础镜像上创建一个自定义 Dockerfile，添加一个额外的层来安装它们：

    ```Dockerfile
    FROM vllm/vllm-openai:v0.9.0

    # 例如，安装 `audio` 可选依赖项
    # 注意：确保 vLLM 的版本与基础镜像匹配！
    RUN uv pip install --system vllm[audio]==0.9.0
    ```

!!! tip
    一些新模型可能仅在 [HF Transformers](https://github.com/huggingface/transformers) 的主分支上可用。

    要使用 `transformers` 的开发版本，可以在基础镜像上创建一个自定义 Dockerfile，
    添加一个额外的层来从源代码安装它们的代码：

    ```Dockerfile
    FROM vllm/vllm-openai:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

[](){ #deployment-docker-build-image-from-source }

## 从源代码构建 vLLM 的 Docker 镜像

您可以通过提供的 <gh-file:docker/Dockerfile> 从源代码构建并运行 vLLM。要构建 vLLM：

```console
# 可选指定：--build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

!!! note
    默认情况下，vLLM 将为所有 GPU 类型构建以实现最广泛的分发。如果您仅为当前机器运行的 GPU 类型构建，
    可以添加参数 `--build-arg torch_cuda_arch_list=""`，让 vLLM 找到当前 GPU 类型并为其构建。

    如果您使用 Podman 而不是 Docker，您可能需要通过在运行 `podman build` 命令时添加 `--security-opt label=disable` 来禁用 SELinux 标签，
    以避免某些 [现有问题](https://github.com/containers/buildah/discussions/4184)。

## 为 Arm64/aarch64 构建

可以为 aarch64 系统（如 Nvidia Grace-Hopper）构建 Docker 容器。目前，这需要使用 PyTorch Nightly，属于**实验性**功能。使用 `--platform "linux/arm64"` 标志将尝试为 arm64 构建。

!!! note
    需要编译多个模块，因此此过程可能需要较长时间。建议使用 `--build-arg max_jobs=` 和 `--build-arg nvcc_threads=` 标志来加速构建过程。
    但是，确保 `max_jobs` 远大于 `nvcc_threads` 以获得最佳效果。注意并行作业的内存使用量，因为它可能很大（见下面的示例）。

```console
# 在 Nvidia GH200 服务器上构建的示例。（内存使用量：约 15GB，构建时间：约 1475 秒 / 25 分钟，镜像大小：6.93GB）
python3 use_existing_torch.py
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t vllm/vllm-gh200-openai:latest \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
  --build-arg vllm_fa_cmake_gpu_arches="90-real"
```

!!! note
    如果您在非 ARM 主机（例如 x86_64 机器）上构建 `linux/arm64` 镜像，您需要确保系统已设置为使用 QEMU 进行跨平台编译。这允许主机模拟 ARM64 执行。

    在主机上运行以下命令以注册 QEMU 用户静态处理程序：

    ```console
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

    设置 QEMU 后，您可以在 `docker build` 命令中使用 `--platform "linux/arm64"` 标志。

## 使用自定义构建的 vLLM Docker 镜像

要使用自定义构建的 Docker 镜像运行 vLLM：

```console
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

参数 `vllm/vllm-openai` 指定要运行的镜像，应替换为自定义构建镜像的名称（即构建命令中的 `-t` 标签）。

!!! note
    **仅适用于版本 0.4.1 和 0.4.2** - 这些版本的 vLLM Docker 镜像应在 root 用户下运行，因为运行时需要加载 root 用户主目录下的库，
    即 `/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`。如果您在其他用户下运行容器，可能需要先更改该库（及其所有父目录）的权限以允许用户访问，
    然后使用环境变量 `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` 运行 vLLM。