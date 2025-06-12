# --8<-- [start:installation]

vLLM 最初支持在 Intel GPU 平台上进行基本模型推理和服务。

!!! warning
    该设备没有预构建的 wheel 或镜像，因此您必须从源代码构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 支持的硬件：Intel 数据中心 GPU，Intel ARC GPU
- OneAPI 要求：oneAPI 2025.0

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前没有预构建的 XPU wheel。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

- 首先，安装所需的驱动程序和 Intel OneAPI 2025.0 或更高版本。
- 其次，安装用于构建 vLLM XPU 后端的 Python 包：

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- 然后，构建并安装 vLLM XPU 后端：

```console
VLLM_TARGET_DEVICE=xpu python setup.py install
```

!!! note
    - FP16 是当前 XPU 后端的默认数据类型。BF16 数据类型在 Intel 数据中心 GPU 上受支持，但尚不支持 Intel Arc GPU。

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

目前没有预构建的 XPU 镜像。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```console
$ docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
$ docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

## 支持的功能

XPU 平台支持**张量并行**推理/服务，并支持作为在线服务的 beta 功能的**管道并行**。我们需要 Ray 作为分布式运行时后端。例如，以下是一个参考执行：

```console
python -m vllm.entrypoints.openai.api_server \
     --model=facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=ray \
     --pipeline-parallel-size=2 \
     -tp=8
```

默认情况下，如果系统中未检测到现有的 Ray 实例，将自动启动一个 Ray 实例，其中 `num-gpus` 等于 `parallel_config.world_size`。我们建议在执行前正确启动一个 Ray 集群，请参考 <gh-file:examples/online_serving/run_cluster.sh> 辅助脚本。
# --8<-- [end:extra-information]