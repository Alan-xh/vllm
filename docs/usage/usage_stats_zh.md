# 使用情况统计收集

vLLM 默认收集匿名使用情况数据，以帮助工程团队更好地了解哪些硬件和模型配置被广泛使用。这些数据使他们能够优先处理最常见的工作负载。收集的数据透明透明，不包含任何敏感信息。

部分数据在清理和汇总后将公开发布，以造福社区。例如，您可以[在此处](https://2024.vllm.ai)查看 2024 年的使用情况报告。

## 收集哪些数据？

最新版本的 vLLM 收集的数据列表位于：<gh-file:vllm/usage/usage_lib.py>

以下是 v0.4.0 版本的示例：

```json
{
"uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
"provider": "GCP",
"num_cpu": 24,
"cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
"cpu_family_model_stepping": "6,85,7",
"total_memory": 101261135872,
"architecture": "x86_64",
"platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
"gpu_count": 2,
"gpu_type": "NVIDIA L4",
"gpu_memory_per_device": 23580639232,
"model_architecture": "OPTForCausalLM",
"vllm_version": "0.3.2+cu123",
"context": "LLM_CLASS",
"log_time": 1711663373492490000,
"source": "production",
"dtype": "torch.float16",
"tensor_parallel_size": 1,
"block_size": 16,
"gpu_memory_utilization": 0.9,
"quantization": null,
"kv_cache_dtype": "auto",
"enable_lora": false,
"enable_prefix_caching": false,
"enforce_eager": false,
"disable_custom_all_reduce": true
}
```

您可以通过运行以下命令预览收集的数据：

```bash
tail ~/.config/vllm/usage_stats.json
```

## 退出

您可以通过设置 `VLLM_NO_USAGE_STATS` 或 `DO_NOT_TRACK` 环境变量，或创建 `~/.config/vllm/do_not_track` 文件来退出使用情况统计收集：

```bash
# 以下任何方法都可以禁用使用情况统计收集
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```