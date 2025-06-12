# --8<-- [start:installation]

张量处理单元（TPU）是谷歌专门为加速机器学习工作负载而开发的专用集成电路（ASIC）。TPU有不同版本，每种版本的硬件规格不同。有关TPU的更多信息，请参见[TPU系统架构](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)。有关vLLM支持的TPU版本的更多信息，请参见：

- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)

这些TPU版本允许您配置TPU芯片的物理排列。这可以提高吞吐量和网络性能。更多信息请参见：

- [TPU v6e拓扑](https://cloud.google.com/tpu/docs/v6e#configurations)
- [TPU v5e拓扑](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)
- [TPU v5p拓扑](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)
- [TPU v4拓扑](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

要使用Cloud TPU，您需要在Google Cloud Platform项目中获得TPU配额。TPU配额指定了您在一个GCP项目中可以使用的TPU数量，并以TPU版本、所需TPU数量和配额类型来定义。更多信息请参见[TPU配额](https://cloud.google.com/tpu/docs/quota#tpu_quota)。

有关TPU定价信息，请参见[Cloud TPU定价](https://cloud.google.com/tpu/pricing)。

您可能需要为TPU VM配置额外的持久存储。更多信息请参见[Cloud TPU数据的存储选项](https://cloud.devsite.corp.google.com/tpu/docs/storage-options)。

!!! warning
    该设备没有预构建的轮子，因此您必须使用预构建的Docker镜像或从源代码构建vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- Google Cloud TPU VM
- TPU版本：v6e、v5e、v5p、v4
- Python：3.10或更高版本

### 配置Cloud TPU

您可以使用[Cloud TPU API](https://cloud.google.com/tpu/docs/reference/rest)或[排队资源](https://cloud.google.com/tpu/docs/queued-resources)API（推荐）来配置Cloud TPU。本节展示如何使用排队资源API创建TPU。有关使用Cloud TPU API的更多信息，请参见[使用Create Node API创建Cloud TPU](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api)。排队资源允许您以排队的方式请求Cloud TPU资源。当您请求排队资源时，该请求会被添加到Cloud TPU服务维护的队列中。当请求的资源可用时，它会被分配给您的Google Cloud项目，供您立即独占使用。

!!! note
    在以下所有命令中，将全大写的参数名称替换为适当的值。有关更多信息，请参见参数描述表。

### 使用GKE配置Cloud TPU

有关在GKE中使用TPU的更多信息，请参见：
- <https://cloud.google.com/kubernetes-engine/docs/how-to/tpus>
- <https://cloud.google.com/kubernetes-engine/docs/concepts/tpus>
- <https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus>

## 配置新环境

### 使用排队资源API配置Cloud TPU

创建具有4个TPU芯片的TPU v5e：

```console
gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
--node-id TPU_NAME \
--project PROJECT_ID \
--zone ZONE \
--accelerator-type ACCELERATOR_TYPE \
--runtime-version RUNTIME_VERSION \
--service-account SERVICE_ACCOUNT
```

| 参数名称           | 描述                                                                                                                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| QUEUED_RESOURCE_ID | 用户指定的排队资源请求ID。                                                                                                                                                                       |
| TPU_NAME           | 用户指定的TPU名称，在排队资源创建时使用。                                                                                                                                                       |
| PROJECT_ID         | 您的Google Cloud项目。                                                                                                                                                                            |
| ZONE               | 您希望创建Cloud TPU的GCP区域。                                                                                                                                                                   |
| ACCELERATOR_TYPE   | 您希望使用的TPU版本。例如，指定TPU版本。                                                                                                                                                         |
| RUNTIME_VERSION    | TPU VM的运行时版本。例如，对于加载了一个或多个v6e TPU的VM，使用`v2-alpha-tpuv6e`。更多信息请参见[TPU VM镜像](https://cloud.google.com/tpu/docs/runtimes)。 |
  <figcaption>参数描述</figcaption>

使用SSH连接到您的TPU：

```bash
gcloud compute tpus tpu-vm ssh TPU_NAME --zone ZONE
```

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前没有预构建的TPU轮子。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

安装Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

为vLLM创建并激活Conda环境：

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
```

克隆vLLM仓库并进入vLLM目录：

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
```

卸载现有的`torch`和`torch_xla`包：

```bash
pip uninstall torch torch-xla -y
```

安装构建依赖项：

```bash
pip install -r requirements/tpu.txt
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```

运行安装脚本：

```bash
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

有关使用官方Docker镜像的说明，请参见[部署Docker预构建镜像][deployment-docker-pre-built-image]，确保将镜像名称`vllm/vllm-openai`替换为`vllm/vllm-tpu`。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

您可以使用<gh-file:docker/Dockerfile.tpu>来构建支持TPU的Docker镜像。

```console
docker build -f docker/Dockerfile.tpu -t vllm-tpu .
```

使用以下命令运行Docker镜像：

```console
# 确保添加`--privileged --net host --shm-size=16G`。
docker run --privileged --net host --shm-size=16G -it vllm-tpu
```

!!! note
    由于TPU依赖于需要静态形状的XLA，vLLM会对可能的输入形状进行分桶，并为每种形状编译一个XLA图。第一次运行的编译时间可能需要20~30分钟。然而，之后由于XLA图被缓存到磁盘（默认在`VLLM_XLA_CACHE_PATH`或`~/.cache/vllm/xla_cache`中），编译时间会减少到约5分钟。

!!! tip
    如果您遇到以下错误：

    ```console
    from torch._C import *  # noqa: F403
    ImportError: libopenblas.so.0: cannot open shared object file: No such
    file or directory
    ```

    使用以下命令安装OpenBLAS：

    ```console
    sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
    ```

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]

此设备没有额外信息。

# --8<-- [end:extra-information]