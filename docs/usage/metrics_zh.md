# 生产指标

vLLM 暴露了许多可用于监控系统健康状况的指标。这些指标通过 vLLM 兼容 OpenAI API 服务器的 `/metrics` 端点进行暴露。

您可以使用 Python 或 [Docker][deployment-docker] 启动服务器：

```console
vllm serve unsloth/Llama-3.2-1B-Instruct
```

然后查询该端点以获取服务器的最新指标：

```console
$ curl http://0.0.0.0:8000/metrics

# HELP vllm:iteration_tokens_total 每个 engine_step 的 token 数量直方图。
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
...
```

以下是暴露的指标：

```python
--8<-- "vllm/engine/metrics.py:metrics-definitions"
```

注意：当指标在版本 `X.Y` 中被弃用时，它们将在版本 `X.Y+1` 中被隐藏，但可以使用 `--show-hidden-metrics-for-version=X.Y` 逃生舱重新启用，然后在版本 `X.Y+2` 中被移除。