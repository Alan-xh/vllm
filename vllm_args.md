# vLLM 配置选项说明

以下是 vLLM 所有命令行选项及其配置的中文翻译和解释，涵盖了模型运行、加载、解码、并行处理、缓存、多模态支持、LoRA、提示适配器、设备、推测解码、观察性和调度等方面的详细描述。\
可以使用 `python3 -m vllm.entrypoints.openai.api_server -h` 查看

## 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--allow-credentials` | 允许使用凭据。 | `False` |
| `--allowed-headers ALLOWED_HEADERS` | 允许的请求头列表。 | `['*']` |
| `--allowed-methods ALLOWED_METHODS` | 允许的请求方法列表。 | `['*']` |
| `--allowed-origins ALLOWED_ORIGINS` | 允许的来源列表。 | `['*']` |
| `--api-key API_KEY` | 如果提供，服务器将要求在请求头中提供此密钥。 | `None` |
| `--chat-template CHAT_TEMPLATE` | 聊天模板的文件路径或单行形式的模板内容，适用于指定模型。 | `None` |
| `--chat-template-content-format {auto,string,openai}` | 聊天模板中消息内容的渲染格式：<br>- `string`：将内容渲染为字符串，例如 `"Hello World"`<br>- `openai`：将内容渲染为 OpenAI 格式的字典列表，例如 `[{"type": "text", "text": "Hello world!"}]` | `auto` |
| `--disable-fastapi-docs` | 禁用 FastAPI 的 OpenAPI 模式、Swagger UI 和 ReDoc 端点。 | `False` |
| `--disable-frontend-multiprocessing` | 如果启用，将在同一进程中运行 OpenAI 前端服务器和模型服务引擎。 | `False` |
| `--disable-log-requests` | 禁用请求日志记录。 | `False` |
| `--disable-log-stats` | 禁用统计日志记录。 | `False` |
| `--disable-uvicorn-access-log` | 禁用 uvicorn 访问日志。 | `False` |
| `--enable-auto-tool-choice` | 为支持的模型启用自动工具选择，需配合 `--tool-call-parser` 指定解析器。 | `False` |
| `--enable-prompt-tokens-details` | 如果启用，将在用量统计中包含 `prompt_tokens_details`。 | `False` |
| `--enable-request-id-headers` | 如果启用，API 服务器将在响应中添加 `X-Request-Id` 头。注意：高 QPS 下会影响性能。 | `False` |
| `--enable-server-load-tracking` | 如果启用，将在应用程序状态中启用 `server_load_metrics` 跟踪。 | `False` |
| `--enable-ssl-refresh` | 当 SSL 证书文件更改时，刷新 SSL 上下文。 | `False` |
| `--host HOST` | 主机名。 | `None` |
| `--lora-modules LORA_MODULES [LORA_MODULES ...]` | LoRA 模块配置，支持 `name=path` 格式或 JSON 格式。例如：<br>- 旧格式：`'name=path'`<br>- 新格式：`{"name": "name", "path": "lora_path", "base_model_name": "id"}` | `None` |
| `--max-log-len MAX_LOG_LEN` | 日志中打印的提示字符或提示 ID 号的最大数量，默认无限制。 | `None` |
| `--middleware MIDDLEWARE` | 应用于应用程序的额外 ASGI 中间件，支持多个 `--middleware` 参数。值应为导入路径。如果提供函数，vLLM 将使用 `@app.middleware('http')` 添加；如果提供类，vLLM 将使用 `app.add_middleware()` 添加。 | `[]` |
| `--port PORT` | 端口号。 | `8000` |
| `--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]` | 提示适配器配置，格式为 `name=path`，支持多个适配器。 | `None` |
| `--response-role RESPONSE_ROLE` | 当 `request.add_generation_prompt=true` 时返回的角色名称。 | `assistant` |
| `--return-tokens-as-token-ids` | 当指定 `--max-logprobs` 时，将单个 token 表示为 `token_id:{token_id}` 形式的字符串，以便识别无法 JSON 编码的 token。 | `False` |
| `--root-path ROOT_PATH` | 当应用程序位于基于路径的路由代理后面时，FastAPI 的根路径。 | `None` |
| `--ssl-ca-certs SSL_CA_CERTS` | CA 证书文件路径。 | `None` |
| `--ssl-cert-reqs SSL_CERT_REQS` | 是否要求客户端证书（参见标准库 ssl 模块）。 | `0` |
| `--ssl-certfile SSL_CERTFILE` | SSL 证书文件路径。 | `None` |
| `--ssl-keyfile SSL_KEYFILE` | SSL 密钥文件路径。 | `None` |
| `--tool-call-parser ` | 根据使用的模型选择工具调用解析器，用于将模型生成的工具调用解析为 OpenAI API 格式。需配合 `--enable-auto-tool-choice` 使用。可选 {deepseek_v3,granite-20b-fc,granite,hermes,internlm,jamba,llama4_pythonic,llama4_json,llama3_json,mistral,phi4_mini_json,pythonic} | `None` |
| `--tool-parser-plugin TOOL_PARSER_PLUGIN` | 指定工具解析器插件，用于解析模型生成的工具调用为 OpenAI API 格式，插件中注册的名称可用于 `--tool-call-parser`。 | `` |
| `--use-v2-block-manager` | [已弃用] 块管理器 v1 已被移除，SelfAttnBlockSpaceManager（即块管理器 v2）现为默认值。设置此标志为 True 或 False 对 vLLM 行为无影响。 | `True` |
| `--uvicorn-log-level {debug,info,warning,error,critical,trace}` | uvicorn 的日志级别。 | `info` |
| `-h, --help` | 显示帮助信息并退出。 | - |

## ModelConfig（模型配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH` | 允许 API 请求从服务器文件系统指定的目录读取本地图像或视频。此操作存在安全风险，仅在可信环境中启用。 | `` |
| `--code-revision CODE_REVISION` | 在 Hugging Face Hub 上使用的模型代码特定版本，可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。 | `None` |
| `--config-format {auto,hf,mistral}` | 加载模型配置的格式：<br>- `auto`：优先尝试加载 Hugging Face 格式配置，若不可用则尝试 Mistral 格式<br>- `hf`：加载 Hugging Face 格式配置<br>- `mistral`：加载 Mistral 格式配置 | `auto` |
| `--disable-async-output-proc` | 禁用异步输出处理，可能导致性能降低。 | `False` |
| `--disable-cascade-attn, --no-disable-cascade-attn` | 禁用 V1 的级联注意力。虽然级联注意力不改变数学正确性，但禁用它有助于避免潜在的数值问题。注意：即使设置为 False，仅在启发式判断有益时才会使用级联注意力。 | `False` |
| `--disable-sliding-window, --no-disable-sliding-window` | 是否禁用滑动窗口。如果启用，将禁用模型的滑动窗口功能，限制为滑动窗口大小。如果模型不支持滑动窗口，此参数将被忽略。 | `False` |
| `--dtype {auto,bfloat16,float,float16,float32,half}` | 模型权重和激活的数据类型：<br>- `auto`：对 FP32 和 FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度<br>- `half`：FP16，推荐用于 AWQ 量化<br>- `float16`：同 `half`<br>- `bfloat16`：在精度和范围之间平衡<br>- `float`：FP32 的简写<br>- `float32`：FP32 精度 | `auto` |
| `--enable-prompt-embeds, --no-enable-prompt-embeds` | 如果启用，允许通过 `prompt_embeds` 键传递文本嵌入作为输入。注意：启用此功能将使图编译时间翻倍。 | `False` |
| `--enable-sleep-mode, --no-enable-sleep-mode` | 启用引擎的睡眠模式（仅支持 CUDA 平台）。 | `False` |
| `--enforce-eager, --no-enforce-eager` | 是否始终使用 PyTorch 的急切模式。如果启用，将禁用 CUDA 图并始终以急切模式执行模型。如果禁用，将混合使用 CUDA 图和急切执行以获得最佳性能和灵活性。 | `False` |
| `--generation-config GENERATION_CONFIG` | 生成配置的文件夹路径。默认值为 `auto`，将从模型路径加载生成配置。如果设置为 `vllm`，不加载生成配置，使用 vLLM 默认值。如果设置为文件夹路径，将从指定路径加载生成配置。如果配置中指定了 `max_new_tokens`，则为所有请求设置服务器范围的输出 token 数量限制。 | `auto` |
| `--hf-config-path HF_CONFIG_PATH` | 使用的 Hugging Face 配置的名称或路径。如果未指定，将使用模型名称或路径。 | `None` |
| `--hf-overrides HF_OVERRIDES` | 如果是字典，将参数转发给 Hugging Face 配置。如果是可调用对象，将调用它来更新 Hugging Face 配置。 | `{}` |
| `--hf-token [HF_TOKEN]` | 用于远程文件访问的 HTTP 承载授权 token。如果为 `True`，将使用 `huggingface-cli login` 时生成的 token（存储在 `~/.huggingface`）。 | `None` |
| `--logits-processor-pattern LOGITS_PROCESSOR_PATTERN` | 可选的正则表达式，指定可通过 `logits_processors` 额外完成参数传递的有效 logits 处理器限定名称。默认值为 `None`，不允许任何处理器。 | `None` |
| `--max-logprobs MAX_LOGPROBS` | 当指定 `logprobs` 时，返回的最大对数概率数量。默认值来自 OpenAI Chat Completions API 的默认值。 | `20` |
| `--max-model-len MAX_MODEL_LEN` | 模型上下文长度（提示和输出）。如果未指定，将从模型配置中自动推导。支持人类可读格式，如 `1k`（1000）、`1K`（1024）、`25.6k`（25,600）。 | `None` |
| `--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE` | CUDA 图覆盖的最大序列长度。如果序列的上下文长度超过此值，将回退到急切模式。对于编码器-解码器模型，如果编码器输入的序列长度超过此值，也将回退到急切模式。 | `8192` |
| `--model MODEL` | 使用的 Hugging Face 模型的名称或路径。还将用作 `model_name` 标签的内容，用于指标输出（当未指定 `served_model_name` 时）。 | `facebook/opt-125m` |
| `--model-impl {auto,vllm,transformers}` | 使用的模型实现：<br>- `auto`：优先使用 vLLM 实现，若不可用则回退到 Transformers 实现<br>- `vllm`：使用 vLLM 模型实现<br>- `transformers`：使用 Transformers 模型实现 | `auto` |
| `--override-generation-config OVERRIDE_GENERATION_CONFIG` | 覆盖或设置生成配置，例如 `{"temperature": 0.5}`。如果与 `--generation-config auto` 一起使用，覆盖参数将与模型的默认配置合并。如果与 `--generation-config vllm` 一起使用，仅使用覆盖参数。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--override-neuron-config OVERRIDE_NEURON_CONFIG` | 初始化或覆盖特定于 Neuron 设备的非默认配置，例如 `{"cast_logits_dtype": "bloat16"}`。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--override-pooler-config OVERRIDE_POOLER_CONFIG` | 初始化或覆盖池化模型的非默认池化配置，例如 `{"pooling_type": "mean", "normalize": false}`。 | `None` |
| `--quantization ` | 权重量化的方法。如果为 `None`，将首先检查模型配置文件中的 `quantization_config` 属性。如果也为 `None`，则假设模型权重未量化，并使用 `dtype` 确定权重的数据类型。可选{aqlm,auto-round,awq,awq_marlin,bitblas,bitsandbytes,compressed-tensors,deepspeedfp,experts_int8,fbgemm_fp8,fp8,gguf,gptq,gptq_bitblas,gptq_marlin,gptq_marlin_24,hqq,ipex,marlin,modelopt,modelopt_fp4,moe_wna16,neuron_quant,ptpc_fp8,qqq,quark,torchao,tpu_int8,None} | `None` |
| `--revision REVISION` | 使用的特定模型版本，可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。 | `None` |
| `--rope-scaling ROPE_SCALING` | RoPE 缩放配置，例如 `{"rope_type":"dynamic","factor":2.0}`。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--rope-theta ROPE_THETA` | RoPE theta 值，与 `rope_scaling` 一起使用。在某些情况下，调整 RoPE theta 可提高缩放模型的性能。 | `None` |
| `--seed SEED` | 用于可重现性的随机种子。在 V0 中初始化为 `None`，在 V1 中初始化为 `0`。 | `None` |
| `--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]` | API 中使用的模型名称。如果提供多个名称，服务器将响应任一名称。响应中的 `model` 字段将使用此列表中的第一个名称。如果未指定，将与 `--model` 参数相同。注意：此名称还将用于 Prometheus 指标的 `model_name` 标签内容，如果提供多个名称，指标标签将使用第一个名称。 | `None` |
| `--skip-tokenizer-init, --no-skip-tokenizer-init` | 跳过分词器和去分词器的初始化。期望输入提供有效的 `prompt_token_ids` 且提示为 `None`。生成的输出将包含 token ID。 | `False` |
| `--task {auto,classify,draft,embed,embedding,generate,reward,score,transcription}` | 使用模型执行的任务。每个 vLLM 实例仅支持一种任务，即使模型支持多种任务。如果模型仅支持一种任务，可使用 `auto` 自动选择；否则，必须明确指定任务。 | `auto` |
| `--tokenizer TOKENIZER` | 使用的 Hugging Face 分词器的名称或路径。如果未指定，将使用模型名称或路径。 | `None` |
| `--tokenizer-mode {auto,custom,mistral,slow}` | 分词器模式：<br>- `auto`：如果可用，将使用快速分词器<br>- `slow`：始终使用慢速分词器<br>- `mistral`：始终使用 `mistral_common` 的分词器<br>- `custom`：使用 `--tokenizer` 选择预注册的分词器 | `auto` |
| `--tokenizer-revision TOKENIZER_REVISION` | 在 Hugging Face Hub 上使用的分词器特定版本，可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。 | `None` |
| `--trust-remote-code, --no-trust-remote-code` | 信任远程代码（例如来自 Hugging Face）以下载模型和分词器。 | `False` |

## LoadConfig（加载配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--download-dir DOWNLOAD_DIR` | 下载和加载权重的目录，默认为 Hugging Face 的默认缓存目录。 | `None` |
| `--ignore-patterns IGNORE_PATTERNS [IGNORE_PATTERNS ...]` | 加载模型时忽略的文件模式列表。默认忽略 `original/**/*` 以避免重复加载 LLaMA 的检查点。 | `None` |
| `--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer,runai_streamer_sharded,fastsafetensors}` | 模型权重的加载格式：<br>- `auto`：尝试加载 safetensors 格式的权重，若不可用则回退到 PyTorch bin 格式<br>- `pt`：加载 PyTorch bin 格式的权重<br>- `safetensors`：加载 safetensors 格式的权重<br>- `npcache`：加载 PyTorch 格式的权重并存储 numpy 缓存以加速加载<br>- `dummy`：用随机值初始化权重，主要用于性能分析<br>- `tensorizer`：使用 CoreWeave 的 tensorizer 库进行快速权重加载<br>- `runai_streamer`：使用 Run:ai Model Streamer 加载 Safetensors 权重<br>- `bitsandbytes`：使用 bitsandbytes 量化加载权重<br>- `sharded_state`：从预分片检查点文件加载权重，支持高效加载张量并行模型<br>- `gguf`：从 GGUF 格式文件加载权重（详情见 [GGUF 文档](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)）<br>- `mistral`：加载 Mistral 模型使用的合并 safetensors 文件<br>- `fastsafetensors`：使用快速 safetensors 加载 | `auto` |
| `--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG` | 模型加载器的额外配置，将传递给所选加载格式对应的加载器。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--pt-load-map-location PT_LOAD_MAP_LOCATION` | PyTorch 检查点加载的映射位置，支持仅在特定设备（如 `cuda`）上加载的检查点，例如 `{"": "cuda"}`。也支持设备映射，如从 GPU 1 到 GPU 0：`{"cuda:1": "cuda:0"}`。详情见 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.load.html)。 | `cpu` |
| `--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH` | 此参数已无效果，请勿设置，将在 v0.10.0 中移除。 | `None` |
| `--use-tqdm-on-load, --no-use-tqdm-on-load` | 是否启用 tqdm 以显示模型权重加载的进度条。 | `True` |

## DecodingConfig（解码配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--enable-reasoning, --no-enable-reasoning` | [已弃用] 自 v0.9.0 起已弃用，请使用 `--reasoning-parser` 指定推理解析器后端。此标志将在 v0.10.0 中移除。当指定 `--reasoning-parser` 时，推理模式将自动启用。 | `None` |
| `--guided-decoding-backend {auto,guidance,lm-format-enforcer,outlines,xgrammar}` | 默认使用的引导解码引擎（支持 JSON 模式、正则表达式等）。`auto` 将根据请求内容和后端库支持情况自动选择，行为可能随版本变化。 | `auto` |
| `--guided-decoding-disable-additional-properties, --no-guided-decoding-disable-additional-properties` | 如果启用，`guidance` 后端将不在 JSON 模式中使用 `additionalProperties`，以更好地与 `outlines` 和 `xgrammar` 对齐。仅支持 `guidance` 后端。 | `False` |
| `--guided-decoding-disable-any-whitespace, --no-guided-decoding-disable-any-whitespace` | 如果启用，模型在引导解码期间不会生成任何空白字符。仅支持 `xgrammar` 和 `guidance` 后端。 | `False` |
| `--guided-decoding-disable-fallback, --no-guided-decoding-disable-fallback` | 如果启用，vLLM 不会在错误时回退到其他后端。 | `False` |
| `--reasoning-parser {deepseek_r1,granite,qwen3}` | 根据使用的模型选择推理解析器，用于将推理内容解析为 OpenAI API 格式。 | `` |

## ParallelConfig（并行配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--data-parallel-address DATA_PARALLEL_ADDRESS, -dpa DATA_PARALLEL_ADDRESS` | 数据并行集群头节点的地址。 | `None` |
| `--data-parallel-rpc-port DATA_PARALLEL_RPC_PORT, -dpp DATA_PARALLEL_RPC_PORT` | 数据并行 RPC 通信的端口。 | `None` |
| `--data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE` | 数据并行组的数量。MoE 层将根据张量并行大小和数据并行大小的乘积进行分片。 | `1` |
| `--data-parallel-size-local DATA_PARALLEL_SIZE_LOCAL, -dpl DATA_PARALLEL_SIZE_LOCAL` | 本节点上运行的数据并行副本数量。 | `None` |
| `--disable-custom-all-reduce, --no-disable-custom-all-reduce` | 禁用自定义 all-reduce 内核并回退到 NCCL。 | `False` |
| `--distributed-executor-backend {external_launcher,mp,ray,uni,None}` | 分布式模型工作进程使用的后端，可选 `ray` 或 `mp`（多进程）。如果管道并行大小和张量并行大小的乘积小于或等于可用 GPU 数量，将使用 `mp` 在单主机上处理。否则，如果安装了 Ray，将默认使用 `ray`，否则会失败。注意：TPU 和 HPU 仅支持 Ray 进行分布式推理。 | `None` |
| `--enable-expert-parallel, --no-enable-expert-parallel` | 对 MoE 层使用专家并行而非张量并行。 | `False` |
| `--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS` | 顺序加载模型时并行加载工作进程的最大数量，以避免在张量并行和大模型下出现 RAM OOM。 | `None` |
| `--pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE` | 管道并行组的数量。 | `1` |
| `--ray-workers-use-nsight, --no-ray-workers-use-nsight` | 是否使用 nsight 分析 Ray 工作进程，详情见 [Ray 文档](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler)。 | `False` |
| `--tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE` | 张量并行组的数量。 | `1` |
| `--worker-cls WORKER_CLS` | 使用的 worker 类的完整名称。如果为 `auto`，将根据平台自动确定 worker 类。 | `auto` |
| `--worker-extension-cls WORKER_EXTENSION_CLS` | 使用的 worker 扩展类的完整名称。worker 扩展类由 worker 类动态继承，用于在 collective_rpc 调用中注入新属性和方法。 | `` |

## CacheConfig（缓存配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--block-size {1,8,16,32,64,128}` | 连续缓存块的大小（以 token 数量计）。在 Neuron 设备上忽略此参数，设置为 `--max-model-len`。在 CUDA 设备上，仅支持最大 32 的块大小。在 HPU 设备上，块大小默认为 128。如果用户未指定，将在 `Platform.check_and_update_configs()` 中根据当前平台设置。 | `None` |
| `--calculate-kv-scales, --no-calculate-kv-scales` | 当 `kv_cache_dtype` 为 fp8 时，启用动态计算 `k_scale` 和 `v_scale`。如果禁用，将从模型检查点加载尺度（如果可用），否则尺度默认为 1.0。 | `False` |
| `--cpu-offload-gb CPU_OFFLOAD_GB` | 每个 GPU 卸载到 CPU 的空间（以 GiB 计）。默认值为 0，表示不卸载。可以看作虚拟增加 GPU 内存大小。例如，1 个 24 GB GPU 设置为 10 GiB，相当于 34 GB GPU。注意：需要快速的 CPU-GPU 互连，因为部分模型会动态从 CPU 内存加载到 GPU 内存。 | `0` |
| `--enable-prefix-caching, --no-enable-prefix-caching` | 是否启用前缀缓存。V0 中默认禁用，V1 中默认启用。 | `None` |
| `--gpu-memory-utilization GPU_MEMORY_UTILIZATION` | 用于模型执行的 GPU 内存比例（0 到 1）。例如，0.5 表示 50% 的 GPU 内存使用率。如果未指定，默认值为 0.9。此限制仅适用于当前 vLLM 实例。例如，同一 GPU 上运行的两个 vLLM 实例可各自设置 0.5 的内存使用率。 | `0.9` |
| `--kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2}` | KV 缓存存储的数据类型。如果为 `auto`，将使用模型数据类型。CUDA 11.8+ 支持 `fp8`（=fp8_e4m3）和 `fp8_e5m2`。ROCm（AMD GPU）支持 `fp8`（=fp8_e4m3）。 | `auto` |
| `--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE` | 使用的 GPU 块数量。如果指定，将覆盖分析的 `num_gpu_blocks`。如果为 `None`，则不起作用。用于测试抢占。 | `None` |
| `--prefix-caching-hash-algo {builtin,sha256}` | 前缀缓存的哈希算法：<br>- `builtin`：使用 Python 内置哈希<br>- `sha256`：抗冲突但有一定开销 | `builtin` |
| `--swap-space SWAP_SPACE` | 每个 GPU 的 CPU 交换空间大小（以 GiB 计）。 | `4` |

## TokenizerPoolConfig（分词器池配置）

> **注意**：此配置已弃用，将在未来版本中移除。设置这些参数无效，请从配置中移除。

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG` | 已弃用，设置此参数无效，请从配置中移除。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--tokenizer-pool-size TOKENIZER_POOL_SIZE` | 已弃用，设置此参数无效，请从配置中移除。 | `0` |
| `--tokenizer-pool-type TOKENIZER_POOL_TYPE` | 已弃用，设置此参数无效，请从配置中移除。 | `ray` |

## MultiModalConfig（多模态配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--disable-mm-preprocessor-cache, --no-disable-mm-preprocessor-cache` | 如果启用，禁用处理后的多模态输入缓存。 | `False` |
| `--limit-mm-per-prompt LIMIT_MM_PER_PROMPT` | 每个提示允许的每种模态输入项的最大数量。V0 中默认为 1，V1 中默认为 999。例如，允许每个提示最多 16 张图片和 2 个视频：`{"images": 16, "videos": 2}`。支持 JSON 字符串或单独传递 JSON 键。 | `{}` |
| `--mm-processor-kwargs MM_PROCESSOR_KWARGS` | 从 `transformers.AutoProcessor.from_pretrained` 获取的多模态处理器参数覆盖。支持的覆盖参数取决于运行的模型。例如，对于 Phi-3-Vision：`{"num_crops": 4}`。支持 JSON 字符串或单独传递 JSON 键。 | `None` |

## LoRAConfig（LoRA 配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--enable-lora, --no-enable-lora` | 如果启用，支持处理 LoRA 适配器。 | `None` |
| `--enable-lora-bias, --no-enable-lora-bias` | 为 LoRA 适配器启用偏置。 | `False` |
| `--fully-sharded-loras, --no-fully-sharded-loras` | 默认情况下，仅对 LoRA 计算进行部分张量并行分片。启用此选项将使用完全分片层。在高序列长度、最大秩或张量并行大小下，这可能更快。 | `False` |
| `--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS [LONG_LORA_SCALING_FACTORS ...]` | 指定多个缩放因子（可与基础模型缩放因子不同，例如 Long LoRA），以允许同时使用以这些缩放因子训练的多个 LoRA 适配器。如果未指定，仅允许使用基础模型缩放因子的适配器。 | `None` |
| `--lora-dtype {auto,bfloat16,float16}` | LoRA 的数据类型。如果为 `auto`，将默认使用基础模型的数据类型。 | `auto` |
| `--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE` | LoRA 适配器可添加的最大额外词汇量（添加到基础模型词汇量）。 | `256` |
| `--max-cpu-loras MAX_CPU_LORAS` | 存储在 CPU 内存中的 LoRA 最大数量，必须大于或等于 `max_loras`。 | `None` |
| `--max-lora-rank MAX_LORA_RANK` | 最大 LoRA 秩。 | `16` |
| `--max-loras MAX_LORAS` | 单批次中的最大 LoRA 数量。 | `1` |

## PromptAdapterConfig（提示适配器配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--enable-prompt-adapter, --no-enable-prompt-adapter` | 如果启用，支持处理提示适配器。 | `None` |
| `--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN` | 提示适配器 token 的最大数量。 | `0` |
| `--max-prompt-adapters MAX_PROMPT_ADAPTERS` | 单批次中的最大提示适配器数量。 | `1` |

## DeviceConfig（设备配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--device {auto,cpu,cuda,hpu,neuron,tpu,xpu}` | vLLM 执行的设备类型。此参数已弃用，将在未来版本中移除。现将根据当前平台自动设置。 | `auto` |

## SpeculativeConfig（推测解码配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--speculative-config SPECULATIVE_CONFIG` | 推测解码的配置，应为 JSON 字符串。 | `None` |

## ObservabilityConfig（观察性配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--collect-detailed-traces {all,model,worker,None} [{all,model,worker,None} ...]` | 仅当设置了 `--otlp-traces-endpoint` 时有效。如果启用，将为指定模块收集详细跟踪信息。这可能涉及高成本或阻塞操作，可能会影响性能。注意：为每个请求收集详细计时信息可能成本较高。 | `None` |
| `--otlp-traces-endpoint OTLP_TRACES_ENDPOINT` | 发送 OpenTelemetry 跟踪的目标 URL。 | `None` |
| `--show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION` | 启用自指定版本以来隐藏的已弃用 Prometheus 指标。例如，若某指标自 v0.7.0 起隐藏，可使用 `--show-hidden-metrics-for-version=0.7` 作为临时过渡，迁移到新指标。此指标可能在未来版本中完全移除。 | `None` |

## SchedulerConfig（调度器配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--cuda-graph-sizes CUDA_GRAPH_SIZES [CUDA_GRAPH_SIZES ...]` | CUDA 图捕获大小，默认值为 512。如果提供一个值，捕获列表将遵循模式 `[1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]`；如果提供多个值（例如 `1 2 128`），捕获列表将遵循提供的列表。 | `[512]` |
| `--disable-chunked-mm-input, --no-disable-chunked-mm-input` | 如果启用且启用了分块预填充，不希望部分调度多模态项目。仅在 V1 中使用。确保混合提示（如文本 token `TTTT` 后跟图像 token `IIIIIIIIII`）不会部分调度（如 `TTTTIIIII`，剩余 `IIIII`），而是在一步中调度 `TTTT`，下一步中调度 `IIIIIIIIII`。 | `False` |
| `--enable-chunked-prefill, --no-enable-chunked-prefill` | 如果启用，预填充请求可以根据剩余的 `max_num_batched_tokens` 进行分块。 | `None` |
| `--long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD` | 对于分块预填充，如果提示的 token 数量超过此值，则视为长提示。 | `0` |
| `--max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS` | 对于分块预填充，允许同时预填充的超过 `long_prefill_token_threshold` 的提示的最大数量。设置小于 `max_num_partial_prefills` 将允许短提示在某些情况下优先于长提示，改善延迟。 | `1` |
| `--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS` | 单次迭代中处理的最大 token 数量。如果用户未指定，将在 `EngineArgs.create_engine_config` 中根据使用上下文设置。 | `None` |
| `--max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS` | 对于分块预填充，允许同时部分预填充的序列最大数量。 | `1` |
| `--max-num-seqs MAX_NUM_SEQS` | 单次迭代中处理的最大序列数量。如果用户未指定，将在 `EngineArgs.create_engine_config` 中根据使用上下文设置。 | `None` |
| `--multi-step-stream-outputs, --no-multi-step-stream-outputs` | 如果禁用，多步调度将在所有步骤结束时流式传输输出。 | `True` |
| `--num-lookahead-slots NUM_LOOKAHEAD_SLOTS` | 每序列每步分配的槽位数量，超出已知 token ID。用于推测解码，存储可能被接受或不被接受的 token 的 KV 激活。注意：未来将由推测配置替换，此参数用于正确性测试。 | `0` |
| `--num-scheduler-steps NUM_SCHEDULER_STEPS` | 每次调度器调用的最大前向步骤数。 | `1` |
| `--preemption-mode {recompute,swap,None}` | 执行抢占的方式（通过重新计算或交换）。如果未指定，将按以下方式确定模式：默认使用重新计算，因为其开销低于交换。但当序列组具有多个序列（例如束搜索）时，不支持重新计算，将使用交换。 | `None` |
| `--scheduler-cls SCHEDULER_CLS` | 使用的调度器类。`vllm.core.scheduler.Scheduler` 为默认调度器。可以是类直接或类路径（如 `mod.custom_class`）。 | `vllm.core.scheduler.Scheduler` |
| `--scheduler-delay-factor SCHEDULER_DELAY_FACTOR` | 在调度下一个提示之前应用延迟（延迟因子乘以上一个提示的延迟）。 | `0.0` |
| `--scheduling-policy {fcfs,priority}` | 使用的调度策略：<br>- `fcfs`：先到先得，按到达顺序处理请求<br>- `priority`：根据给定的优先级（值越低越早处理）及到达时间决定处理顺序 | `fcfs` |

## VllmConfig（vLLM 配置）

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--additional-config ADDITIONAL_CONFIG` | 指定平台的额外配置。不同平台支持的配置可能不同。确保配置对所用平台有效，内容必须可哈希。 | `{}` |
| `--compilation-config COMPILATION_CONFIG, -O COMPILATION_CONFIG` | `torch.compile` 的模型编译配置。如果为数字（0、1、2、3），将解释为优化级别：<br>- 0：无优化，默认级别<br>- 1、2：仅用于内部测试<br>- 3：生产环境推荐级别<br>支持不带空格的 `-O`（如 `-O3` 等同于 `-O 3`）。可指定完整编译配置，例如 `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`。支持 JSON 字符串或单独传递 JSON 键。 | `{"inductor_compile_config": {"enable_auto_functionalized_v2": false}}` |
| `--kv-events-config KV_EVENTS_CONFIG` | 事件发布配置。支持 JSON 字符串或单独传递 JSON 键。 | `None` |
| `--kv-transfer-config KV_TRANSFER_CONFIG` | 分布式 KV 缓存传输配置。支持 JSON 字符串或单独传递 JSON 键。 | `None` |

- VllmConfig
    包含所有与 vllm 相关的配置的数据类。这简化了代码库中不同配置的传递。

