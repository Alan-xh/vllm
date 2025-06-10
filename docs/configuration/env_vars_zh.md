# 环境变量

vLLM 使用以下环境变量来配置系统：

!!! 警告
请注意，`VLLM_PORT` 和 `VLLM_HOST_IP` 设置的是 vLLM 内部使用的端口和 IP，而不是 API 服务器的端口和 IP。如果您使用 `--host $VLLM_HOST_IP` 和 `--port $VLLM_PORT` 启动 API 服务器，它将无法正常工作。

vLLM 使用的所有环境变量都以 `VLLM_` 为前缀。**Kubernetes 用户需特别注意**：请勿将服务命名为 `vllm`，否则 Kubernetes 设置的环境变量可能与 vLLM 的环境变量冲突，因为 [Kubernetes 会以大写服务名称作为前缀为每个服务设置环境变量](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables)。

```python
--8<-- "vllm/envs.py:env-vars-definition"
```