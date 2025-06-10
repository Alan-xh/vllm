---
title: 服务器参数
---
[](){ #serve-args }

`vllm serve` 命令用于启动兼容 OpenAI 的服务器。

## 命令行参数

`vllm serve` 命令用于启动兼容 OpenAI 的服务器。
要查看可用的命令行参数，请运行 `vllm serve --help`！

## 配置文件

您可以通过 [YAML](https://yaml.org/) 配置文件加载命令行参数。
参数名称必须是 [上面][serve-args] 列出的长格式。

例如：

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

要使用上述配置文件：

```bash
vllm serve --config config.yaml
```

!!! note
    如果一个参数同时通过命令行和配置文件提供，命令行的值将优先。
    优先级顺序为 `命令行 > 配置文件值 > 默认值`。
    例如：`vllm serve SOME_MODEL --config config.yaml`，SOME_MODEL 将优先于配置文件中的 `model`。