# vLLM CLI 指南

vllm 命令行工具用于运行和管理 vLLM 模型。你可以通过以下命令查看帮助信息：

```
vllm --help
```

可用命令：

```
vllm {chat,complete,serve,bench,collect-env,run-batch}
```

## serve

启动 vLLM OpenAI 兼容的 API 服务器。

示例：

```bash
# 使用指定模型启动
vllm serve meta-llama/Llama-2-7b-hf

# 指定端口
vllm serve meta-llama/Llama-2-7b-hf --port 8100

# 使用 --help 查看更多选项
# 列出所有参数组
vllm serve --help=listgroup

# 查看某个参数组
vllm serve --help=ModelConfig

# 查看单个参数
vllm serve --help=max-num-seqs

# 按关键字搜索
vllm serve --help=max
```

## chat

通过运行的 API 服务器生成聊天补全。

示例：

```bash
# 不带参数直接连接本地 API
vllm chat

# 指定 API 地址
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 使用单个提示进行快速聊天
vllm chat --quick "hi"
```

## complete

通过运行的 API 服务器根据给定的提示生成文本补全。

示例：

```bash
# 不带参数直接连接本地 API
vllm complete

# 指定 API 地址
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 使用单个提示进行快速补全
vllm complete --quick "人工智能的未来是"
```

## bench

运行基准测试，测试延迟、在线服务吞吐量和离线推理吞吐量。

要使用基准测试命令，请使用以下命令安装额外依赖：

```bash
pip install vllm[bench]
```

可用命令：

```bash
vllm bench {latency, serve, throughput}
```

### latency

测试单批次请求的延迟。

示例：

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

### serve

测试在线服务吞吐量。

示例：

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

### throughput

测试离线推理吞吐量。

示例：

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

## collect-env

开始收集环境信息。

```bash
vllm collect-env
```

## run-batch

运行批量提示并将结果写入文件。

示例：

```bash
# 使用本地文件运行
vllm run-batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# 使用远程文件
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## 更多帮助

要查看任何子命令的详细选项，请使用：

```bash
vllm <subcommand> --help
```