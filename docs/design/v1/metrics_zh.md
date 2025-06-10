# 指标

确保 v1 LLM 引擎暴露的指标集是 v0 指标集的超集。

## 目标

- 实现 v0 和 v1 之间的指标对等。
- 优先使用场景是通过 Prometheus 访问这些指标，因为这是在生产环境中预期使用的。
- 提供日志支持，即将指标打印到信息日志，用于更临时的测试、调试、开发和探索性用例。

## 背景

vLLM 的指标可以分为以下几类：

1. **服务器级指标**：这些是跟踪 LLM 引擎状态和性能的全局指标，通常在 Prometheus 中以 Gauge 或 Counter 的形式暴露。
2. **请求级指标**：这些是跟踪单个请求特性（例如大小和时间）的指标，通常在 Prometheus 中以 Histogram 的形式暴露，通常是 SRE 监控 vLLM 时跟踪的 SLO。

我们的心智模型是，“服务器级指标”解释了“请求级指标”为何如此。

### v0 指标

在 v0 中，以下指标通过一个与 Prometheus 兼容的 `/metrics` 端点暴露，使用 `vllm:` 前缀：

- `vllm:num_requests_running`（Gauge）
- `vllm:num_requests_swapped`（Gauge）
- `vllm:num_requests_waiting`（Gauge）
- `vllm:gpu_cache_usage_perc`（Gauge）
- `vllm:cpu_cache_usage_perc`（Gauge）
- `vllm:gpu_prefix_cache_hit_rate`（Gauge）
- `vllm:cpu_prefix_cache_hit_rate`（Gauge）
- `vllm:prompt_tokens_total`（Counter）
- `vllm:generation_tokens_total`（Counter）
- `vllm:request_success_total`（Counter）
- `vllm:request_prompt_tokens`（Histogram）
- `vllm:request_generation_tokens`（Histogram）
- `vllm:time_to_first_token_seconds`（Histogram）
- `vllm:time_per_output_token_seconds`（Histogram）
- `vllm:e2e_request_latency_seconds`（Histogram）
- `vllm:request_queue_time_seconds`（Histogram）
- `vllm:request_inference_time_seconds`（Histogram）
- `vllm:request_prefill_time_seconds`（Histogram）
- `vllm:request_decode_time_seconds`（Histogram）
- `vllm:request_max_num_generation_tokens`（Histogram）
- `vllm:num_preemptions_total`（Counter）
- `vllm:cache_config_info`（Gauge）
- `vllm:lora_requests_info`（Gauge）
- `vllm:tokens_total`（Counter）
- `vllm:iteration_tokens_total`（Histogram）
- `vllm:time_in_queue_requests`（Histogram）
- `vllm:model_forward_time_milliseconds`（Histogram）
- `vllm:model_execute_time_milliseconds`（Histogram）
- `vllm:request_params_n`（Histogram）
- `vllm:request_params_max_tokens`（Histogram）
- `vllm:spec_decode_draft_acceptance_rate`（Gauge）
- `vllm:spec_decode_efficiency`（Gauge）
- `vllm:spec_decode_num_accepted_tokens_total`（Counter）
- `vllm:spec_decode_num_draft_tokens_total`（Counter）
- `vllm:spec_decode_num_emitted_tokens_total`（Counter）

这些指标在 [推断和提供服务 -> 生产指标](../../usage/metrics.md) 中有文档记录。

### Grafana 仪表板

vLLM 还提供了[一个参考示例](https://docs.vllm.ai/en/latest/examples/prometheus_grafana.html)，说明如何使用 Prometheus 收集和存储这些指标，并使用 Grafana 仪表板进行可视化。

在 Grafana 仪表板中暴露的指标子集表明哪些指标尤其重要：

- `vllm:e2e_request_latency_seconds_bucket` - 端到端请求延迟，以秒为单位。
- `vllm:prompt_tokens_total` - 提示令牌。
- `vllm:generation_tokens_total` - 生成令牌。
- `vllm:time_per_output_token_seconds` - 每输出令牌的延迟（TPOT），以秒为单位。
- `vllm:time_to_first_token_seconds` - 首令牌时间（TTFT）延迟，以秒为单位。
- `vllm:num_requests_running`（以及 `_swapped` 和 `_waiting`） - 处于运行、等待和交换状态的请求数量。
- `vllm:gpu_cache_usage_perc` - vLLM 使用的缓存块百分比。
- `vllm:request_prompt_tokens` - 请求提示长度。
- `vllm:request_generation_tokens` - 请求生成长度。
- `vllm:request_success_total` - 按完成原因统计的已完成请求数量：生成 EOS 令牌或达到最大序列长度。
- `vllm:request_queue_time_seconds` - 队列时间。
- `vllm:request_prefill_time_seconds` - 请求预填充时间。
- `vllm:request_decode_time_seconds` - 请求解码时间。
- `vllm:request_max_num_generation_tokens` - 序列组中的最大生成令牌。

有关此仪表板添加的 PR 可查看 [添加仪表板的 PR](gh-pr:2316)，了解有关选择背景的有趣且有用的信息。

### Prometheus 客户端库

Prometheus 支持最初通过 [aioprometheus 库](gh-pr:1890) 添加，但很快切换到 [prometheus_client](gh-pr:2730)。两个 PR 中讨论了切换的理由。

切换到 `aioprometheus` 后，我们失去了用于跟踪 HTTP 指标的 `MetricsMiddleware`，但通过 [prometheus_fastapi_instrumentator](gh-pr:15657) 恢复了该功能：

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### 多进程模式

在 v0 中，指标在引擎核心进程中收集，并使用多进程模式使其在 API 服务器进程中可用。参见 <gh-pr:7279>。

### 内置 Python/进程指标

`prometheus_client` 默认支持以下指标，但在使用多进程模式时未暴露：

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

这与我们的讨论相关，因为如果在 v1 中不再使用多进程模式，这些指标将重新可用。然而，这些指标是否足够有意义（因为它们不聚合组成 vLLM 实例的所有进程的统计数据）值得商榷。

### v0 PR 和问题

为提供背景，以下是添加 v0 指标的相关 PR：

- <gh-pr:1890>
- <gh-pr:2316>
- <gh-pr:2730>
- <gh-pr:4464>
- <gh-pr:7279>

另请注意 ["更佳可观察性"](gh-issue:3616) 功能，其中例如 [制定了详细路线图](gh-issue:3616#issuecomment-2030858781)。

## v1 设计

### v1 PR

为提供背景，以下是与 v1 指标问题 <gh-issue:10582> 相关的 PR：

- <gh-pr:11962>
- <gh-pr:11973>
- <gh-pr:10907>
- <gh-pr:12416>
- <gh-pr:12478>
- <gh-pr:12516>
- <gh-pr:12530>
- <gh-pr:12561>
- <gh-pr:12579>
- <gh-pr:12592>
- <gh-pr:12644>

### 指标收集

在 v1 中，我们希望将计算和开销从引擎核心进程移出，以最小化每次前向传递之间的时间。

V1 EngineCore 设计的总体思路是：
- **EngineCore** 是内循环，性能在此最为关键。
- **AsyncLLM** 是外循环，理想情况下与 GPU 执行重叠，因此这里是放置“开销”的理想位置。因此，`AsyncLLM.output_handler_loop` 是进行指标记录的理想位置。

我们将通过在前端 API 服务器中收集指标来实现这一点，并基于引擎核心进程返回的 `EngineCoreOutputs` 中可获取的信息。

### 间隔计算

许多指标是请求处理中各个事件之间的时间间隔。最佳实践是使用基于“单调时间”（`time.monotonic()`）的时间戳，而不是“壁钟时间”（`time.time()`）来计算间隔，因为前者不受系统时钟变化（例如 NTP）的影响。

需要注意的是，单调时钟在不同进程之间是不同的——每个进程都有自己的参考点。因此，比较来自不同进程的单调时间戳是没有意义的。

因此，为了计算间隔，我们必须比较来自同一进程的两个单调时间戳。

### 调度器统计

引擎核心进程将从调度器收集一些关键统计数据，例如在上次调度器通过后调度或等待的请求数量，并将这些统计数据包含在 `EngineCoreOutputs` 中。

### 引擎核心事件

引擎核心还将记录某些按请求的事件时间戳，以便前端可以计算这些事件之间的间隔。

事件包括：

- `QUEUED` - 请求被引擎核心接收并添加到调度器队列时。
- `SCHEDULED` - 请求首次被调度执行时。
- `PREEMPTED` - 请求被放回等待队列，以便为其他请求完成腾出空间。它将在未来重新调度并重新开始其预填充阶段。
- `NEW_TOKENS` - `EngineCoreOutput` 中包含的输出生成时。由于这对给定迭代中的所有请求是通用的，我们在 `EngineCoreOutputs` 上使用单一时间戳记录此事件。

计算的间隔包括：

- **队列间隔** - `QUEUED` 和最近的 `SCHEDULED` 之间的间隔。
- **预填充间隔** - 最近的 `SCHEDULED` 和后续的第一个 `NEW_TOKENS` 之间的间隔。
- **解码间隔** - 最近的 `SCHEDULED` 后的第一个和最后一个 `NEW_TOKENS` 之间的间隔。
- **推理间隔** - 最近的 `SCHEDULED` 和最后一个 `NEW_TOKENS` 之间的间隔。
- **令牌间间隔** - 连续的 `NEW_TOKENS` 之间的间隔。

换句话说：

![间隔计算 - 常见情况](../../assets/design/v1/metrics/intervals-1.png)

我们探索了前端使用前端可见事件的时间来计算这些间隔的可能性。然而，前端无法看到 `QUEUED` 和 `SCHEDULED` 事件的时间，并且由于我们需要基于同一进程的单调时间戳计算间隔，我们需要引擎核心为所有这些事件记录时间戳。

#### 间隔计算与抢占

当解码期间发生抢占时，由于已生成的令牌会被重用，我们认为抢占会影响令牌间、解码和推理间隔。

![间隔计算 - 抢占解码](../../assets/design/v1/metrics/intervals-2.png)

当预填充期间发生抢占（假设这种事件可能发生）时，我们认为抢占会影响首令牌时间和预填充间隔。

![间隔计算 - 抢占预填充](../../assets/design/v1/metrics/intervals-3.png)

### 前端统计收集

当前端处理单个 `EngineCoreOutputs`（即单个引擎核心迭代的输出）时，它会收集与该迭代相关的各种统计数据：

- 该迭代中生成的新令牌总数。
- 该迭代中完成的预填充处理的提示令牌总数。
- 该迭代中调度的任何请求的队列间隔。
- 该迭代中完成预填充的任何请求的预填充间隔。
- 该迭代中包含的所有请求的令牌间间隔（每输出令牌时间，TPOT）。
- 该迭代中完成预填充的任何请求的首令牌时间（TTFT）。然而，我们相对于前端首次接收请求的时间（`arrival_time`）计算此间隔，以考虑输入处理时间。

对于在给定迭代中完成的任何请求，我们还记录：

- 推理和解码间隔 - 相对于调度和首令牌事件，如上所述。
- 端到端延迟 - 前端 `arrival_time` 和前端接收最终令牌之间的间隔。

### 指标发布 - 日志

`LoggingStatLogger` 指标发布者每 5 秒输出一个包含关键指标的 `INFO` 日志消息：

- 当前运行/等待请求的数量。
- 当前 GPU 缓存使用率。
- 过去 5 秒内每秒处理的提示令牌数。
- 过去 5 秒内每秒生成的新令牌数。
- 最近 1k 次 kv 缓存块查询的前缀缓存命中率。

### 指标发布 - Prometheus

`PrometheusStatLogger` 指标发布者通过一个与 Prometheus 兼容的 `/metrics` HTTP 端点提供指标。Prometheus 实例可以配置为定期（例如每秒）轮询此端点，并将其值记录在其时间序列数据库中。Prometheus 通常通过 Grafana 使用，允许这些指标随时间绘制图表。

Prometheus 支持以下指标类型：

- **Counter**：随时间增加的值，永不减少，通常在 vLLM 实例重启时重置为零。例如，实例生命周期内生成的令牌数。
- **Gauge**：上下波动的值，例如当前调度执行的请求数量。
- **Histogram**：记录在桶中的指标样本计数。例如，TTFT 小于 1ms、5ms、10ms、20ms 等的请求数量。

Prometheus 指标还可以添加标签，允许根据匹配标签组合指标。在 vLLM 中，我们为每个指标添加一个 `model_name` 标签，包含该实例服务的模型名称。

示例输出：

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 97.0
vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 123.0
vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 138.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
```

注意 - 选择对用户最有用的直方图桶在广泛的用例中并不简单，需要随时间优化。

### 缓存配置信息

`prometheus_client` 支持 [Info 指标](https://prometheus.github.io/client_python/instrumenting/info/)，相当于一个永久设置为 1 的 `Gauge`，但通过标签暴露有趣的键/值对信息。这用于实例启动时仅需观察一次的不变信息，并允许在 Prometheus 中跨实例比较。

我们将此概念用于 `vllm:cache_config_info` 指标：

```
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",calculate_kv_scales="False",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",...} 1.0
```

然而，`prometheus_client` 在多进程模式下 [从不支持 Info 指标](https://github.com/prometheus/client_python/pull/300)，原因 [不明确](gh-pr:7279#discussion_r1710417152)。我们简单地使用一个设置为 1 的 `Gauge` 指标，并使用 `multiprocess_mode="mostrecent"` 替代。

### LoRA 指标

`vllm:lora_requests_info` `Gauge` 有些类似，只是值是当前的壁钟时间，并且在每次迭代中更新。

使用的标签名称包括：

- `running_lora_adapters`：每个适配器运行的请求数量，以逗号分隔的字符串格式化。
- `waiting_lora_adapters`：类似，但统计等待调度的请求。
- `max_lora` - 静态的“单个批次中最大 LoRA 数量”配置。

将多个适配器的运行/等待计数编码为逗号分隔的字符串似乎不太合理 - 我们可以使用标签来区分每个适配器的计数。这一设计应重新审视。

注意，使用了 `multiprocess_mode="livemostrecent"` - 使用来自当前运行进程的最新指标。

此功能在 <gh-pr:9477> 中添加，并且 [至少有一个已知用户](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/54)。如果我们重新审视此设计并废弃旧指标，应通过在 v0 中也进行更改并请求该项目迁移到新指标，以减少需要较长的废弃期。

### 前缀缓存指标

关于添加前缀缓存指标的讨论 <gh-issue:10582> 提出了一些可能与未来指标处理方式相关的有趣观点。

每次查询前缀缓存时，我们记录查询的令牌数和缓存中存在的查询令牌数（即命中）。

然而，关注的指标是命中率 - 即每次查询的命中次数。

对于日志记录，我们认为用户最好通过计算最近固定数量查询（当前固定为最近 1k 次查询）的命中率来服务。

对于 Prometheus，我们应利用其时间序列特性，允许用户计算其选择的时间间隔内的命中率。例如，过去 5 分钟的命中率 PromQL 查询：

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

为此，我们应在 Prometheus 中将查询和命中记录为计数器，而不是将命中率记录为 Gauge。

## 废弃指标

### 如何废弃

废弃指标不应轻率。用户可能未注意到指标已被废弃，当其突然（从用户角度看）被移除时，可能会造成不便，即使存在等效指标。

例如，`vllm:avg_prompt_throughput_toks_per_s` 如何被 [废弃](gh-pr:2764)（代码中有注释），[移除](gh-pr:12383)，然后被 [用户注意到](gh-issue:13218)。

一般来说：

1) 我们应谨慎废弃指标，特别是因为难以预测用户影响。
2) 我们应在 `/metrics` 输出的帮助字符串中包含显著的废弃通知。
3) 我们应在面向用户的文档和发布说明中列出废弃指标。
4) 我们应考虑通过 CLI 参数隐藏废弃指标，以便管理员在删除前 [有逃生通道](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)。

参见 [废弃策略](../../contributing/deprecation_policy.md) 以了解项目范围的废弃策略。

### 未实现 - `vllm:tokens_total`

由 <gh-pr:4464> 添加，但显然从未实现。可以直接移除。

### 重复 - 队列时间

`vllm:time_in_queue_requests` 直方图指标由 <gh-pr:9659> 添加，其计算为：

```
    self.metrics.first_scheduled_time = now
    self.metrics.time_in_queue = now - self.metrics.arrival_time
```

两周后，<gh-pr:4464> 添加了 `vllm:request_queue_time_seconds`，导致：

```
if seq_group.is_finished():
    if (seq_group.metrics.first_scheduled_time is not None and
            seq_group.metrics.first_token_time is not None):
        time_queue_requests.append(
            seq_group.metrics.first_scheduled_time -
            seq_group.metrics.arrival_time)
    ...
    if seq_group.metrics.time_in_queue is not None:
        time_in_queue_requests.append(
            seq_group.metrics.time_in_queue)
```

这似乎是重复的，应移除其中之一。后者被 Grafana 仪表板使用，因此我们应从 v0 中废弃或移除前者。

### 前缀缓存命中率

如上所述 - 我们现在暴露“查询”和“命中”计数器，而不是“命中率”Gauge。

### KV 缓存卸载

v0 中的两个指标与 v1 中不再相关的“交换”抢占模式有关：

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

在此模式下，当请求被抢占（例如为完成其他请求腾出 KV 缓存空间）时，我们将 KV 缓存块交换到 CPU 内存。这也称为“KV 缓存卸载”，通过 `--swap-space` 和 `--preemption-mode` 配置。

在 v0 中，[vLLM 长期支持束搜索](gh-issue:6226)。SequenceGroup 封装了共享相同提示 KV 块的 N 个序列的概念。这支持了请求之间的 KV 缓存块共享，以及分支的写时复制。CPU 交换是为这些束搜索用例设计的。

后来引入了前缀缓存，允许隐式共享 KV 缓存块。这被证明比 CPU 交换更好，因为块可以按需缓慢逐出，且逐出的提示部分可以重新计算。

在 v1 中，SequenceGroup 被移除，尽管 [“并行采样”（`n>1`）将需要替代](gh-issue:8306)。[束搜索已从核心移出（在 v0 中）](gh-issue:8306)。这部分代码复杂且使用不常见。

在 v1 中，前缀缓存更好（零开销）且默认启用，抢占和重新计算策略应更有效。

## 未来工作

### 并行采样

一些 v0 指标仅在“并行采样”场景下相关。这是请求中的 `n` 参数用于从同一提示请求多个完成的情况。

作为 <gh-pr:10980> 中添加并行采样支持的一部分，我们还应添加以下指标：

- `vllm:request_params_n`（Histogram）

观察每个已完成请求的 `n` 参数值。

- `vllm:request_max_num_generation_tokens`（Histogram）

观察每个已完成序列组中所有序列的最大输出长度。在没有并行采样的情况下，这等同于 `vllm:request_generation_tokens`。

### 推测解码

一些 v0 指标特定于“推测解码”。这是使用更快、近似的方法或模型生成候选令牌，然后用更大模型验证这些令牌。

- `vllm:spec_decode_draft_acceptance_rate`（Gauge）
- `vllm:spec_decode_efficiency`（Gauge）
- `vllm:spec_decode_num_accepted_tokens_total`（Counter）
- `vllm:spec_decode_num_draft_tokens_total`（Counter）
- `vllm:spec_decode_num_emitted_tokens_total`（Counter）

目前有一个 PR（<gh-pr:12193>）在审查中，计划为 v1 添加“提示查找（ngram）”推测解码。其他技术将随后跟进。我们应在此背景下重新审视 v0 指标。

注意 - 我们可能应像前缀缓存命中率一样，将接受率暴露为单独的接受和草稿计数器。效率可能也需要类似处理。

### 自动扩展和负载均衡

我们的指标常见用例是支持 vLLM 实例的自动扩展。

来自 [Kubernetes 服务工作组](https://github.com/kubernetes/community/tree/master/wg-serving) 的相关讨论，参见：

- [在 Kubernetes 中标准化大型模型服务器指标](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
- [为 Kubernetes 中的性能评估和自动扩展对 LLM 工作负载进行基准测试](https://docs.google.com/document/d/1k4Q4X14hW4vftElIuYGDu5KDe2LtV1XammoG-Xi3bbQ)
- [推理性能](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <gh-issue:5041> 和 <gh-pr:12726>。

这是一个非琐碎的主题。考虑 Rob 的评论：

> 我认为此指标应专注于估计最大并发量，这将导致平均请求长度 > 每秒查询数……因为这实际上是“饱和”服务器的原因。

一个明确目标是我们应暴露检测此饱和点所需的指标，以便管理员可以基于这些实现自动扩展规则。然而，为此，我们需要清楚管理员（和自动化监控系统）应如何判断实例接近饱和：

> 如何识别模型服务器计算的饱和点（无法通过更高请求率获得更多吞吐量，反而开始增加延迟的拐点），以便有效自动扩展？

### 指标命名

我们的指标命名方式可能需要重新审视：

1. 在指标名称中使用冒号似乎与 [“冒号保留用于用户定义的记录规则”](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels) 相悖。
2. 我们的大多数指标遵循以单位结尾的惯例，但并非全部如此。
3. 一些指标名称以 `_total` 结尾：

```
如果指标名称有 `_total` 后缀，它将被移除。在暴露计数器的时间序列时，将添加 `_total` 后缀。这是为了兼容 OpenMetrics 和 Prometheus 文本格式，因为 OpenMetrics 要求 `_total` 后缀。
```

### 添加更多指标

关于新指标的创意不胜枚举：

- 其他项目的示例，如 [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics)
- 特定用例提出的建议，如上述 Kubernetes 自动扩展主题
- 可能来自标准化努力的建议，如 [OpenTelemetry 生成 AI 语义约定](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)。

我们应谨慎添加新指标。虽然添加指标通常较为简单：

1. 它们可能难以移除 - 参见上文关于废弃的部分。
2. 启用时可能对性能产生显著影响。除非指标默认启用并在生产中使用，否则通常用处有限。
3. 它们对项目的开发和维护产生影响。v0 中添加的每个指标都使 v1 的工作更加耗时，并非所有指标都值得持续维护投资。

## 追踪 - OpenTelemetry

指标提供系统性能和健康的聚合视图，而追踪则跟踪单个请求在不同服务和组件中的移动。两者都属于“可观察性”的范畴。

v0 支持 OpenTelemetry 追踪：

- 由 <gh-pr:4687> 添加
- 使用 `--oltp-traces-endpoint` 和 `--collect-detailed-traces` 配置
- [OpenTelemetry 博客文章](https://opentelemetry.io/blog/2024/llm-observability/)
- [面向用户文档](https://docs.vllm.ai/en/latest/examples/opentelemetry.html)
- [博客文章](https://medium.com/@ronen.schaffer/follow-the-trail-supercharging-vllm-with-opentelemetry-distributed-tracing-aa655229b46f)
- [IBM 产品文档](https://www.ibm.com/docs/en/instana-observability/current?topic=mgaa-monitoring-large-language-models-llms-vllm-public-preview)

OpenTelemetry 有一个 [生成 AI 工作组](https://github.com/open-telemetry/community/blob/main/projects/gen-ai.md)。

由于指标本身是一个足够大的主题，我们将在 v1 中单独处理追踪问题。

### OpenTelemetry 模型前向与执行时间

在 v0 中，我们有以下两个指标：

- `vllm:model_forward_time_milliseconds`（Histogram） - 请求在批次中时模型前向传递的时间。
- `vllm:model_execute_time_milliseconds`（Histogram） - 模型执行函数的时间。包括模型前向、跨工作者的块/同步、CPU-GPU 同步时间和采样时间。

这些指标仅在启用 OpenTelemetry 追踪且使用 `--collect-detailed-traces=all/model/worker` 时启用。该选项的文档说明：

> 为指定模块收集详细追踪。这涉及可能成本高昂或阻塞的操作，因此可能对性能产生影响。

这些指标由 <gh-pr:7089> 添加，并在 OpenTelemetry 追踪中显示为：

```
-> gen_ai.latency.time_in_scheduler: Double(0.017550230026245117)
-> gen_ai.latency.time_in_model_forward: Double(3.151565277099609)
-> gen_ai.latency.time_in_model_execute: Double(3.6468167304992676)
```

我们已有 `inference_time` 和 `decode_time` 指标，因此问题在于是否常见用例足以证明高分辨率时间的开销。

由于我们将单独处理 OpenTelemetry 支持问题，我们将这些特定指标纳入该主题讨论。