---
title: LoRA 适配器
---
[](){ #lora-adapter }

本文档展示如何在基础模型上使用 [LoRA 适配器](https://arxiv.org/abs/2106.09685) 与 vLLM 结合。

LoRA 适配器可以与任何实现了 [SupportsLoRA][vllm.model_executor.models.interfaces.SupportsLoRA] 接口的 vLLM 模型一起使用。

适配器可以按需在每个请求时高效加载，占用最小的资源。首先，我们下载适配器并将其保存到本地路径：

```python
from huggingface import snapshot_download

sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
```

然后我们实例化基础模型，并传入 `enable_lora=True` 标志：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)
```

我们现在可以提交提示词并调用 `llm.generate`，并传入 `lora_request` 参数。`LoRARequest` 的第一个参数是人类可识别的名称，第二个参数是适配器的全局唯一 ID，第三个参数是 LoRA 适配器的路径。

```python
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)

prompts = [
     "[user] 根据表结构编写 SQL 查询以回答问题。\n\n 上下文：CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n 问题：利隆圭国际机场的 ICAO 是什么？[/user] [assistant]",
     "[user] 根据表结构编写 SQL 查询以回答问题。\n\n 上下文：CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n 问题：当 Anchero Pantaleone 是选民时，国籍是什么？[/user] [assistant]",
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
)
```

请查看 <gh-file:examples/offline_inference/multilora_inference.py> 以获取如何使用异步引擎和更高级配置选项的 LoRA 适配器示例。

## 服务 LoRA 适配器

LoRA 适配模型也可以通过与 Open-AI 兼容的 vLLM 服务器进行服务。为此，我们使用 `--lora-modules {name}={path} {name}={path}` 来指定每个 LoRA 模块，在启动服务器时：

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```

!!! note
    提交 ID `0dfa347e8877a4d4ed19ee56c140fa518470028c` 可能会随时间变化。请检查您环境中的最新提交 ID 以确保使用正确的一个。

服务器入口点接受所有其他 LoRA 配置参数（`max_loras`、`max_lora_rank`、`max_cpu_loras` 等），这些参数将应用于所有后续请求。在查询 `/models` 端点时，我们应该能看到我们的 LoRA 以及其基础模型（如果未安装 `jq`，可以按照 [此指南](https://jqlang.org/download/) 进行安装）：

```bash
curl localhost:8000/v1/models | jq .
{
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-2-7b-hf",
            "object": "model",
            ...
        },
        {
            "id": "sql-lora",
            "object": "model",
            ...
        }
    ]
}
```

请求可以通过 `model` 请求参数指定 LoRA 适配器，就像指定任何其他模型一样。请求将根据服务器范围的 LoRA 配置进行处理（即与基础模型请求并行处理，如果提供了其他 LoRA 适配器请求并且 `max_loras` 设置足够高，则可能与其他 LoRA 适配器请求并行处理）。

以下是一个示例请求：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "旧金山是",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## 动态服务 LoRA 适配器

除了在服务器启动时服务 LoRA 适配器外，vLLM 服务器还支持通过专用 API 端点和插件在运行时动态配置 LoRA 适配器。当需要灵活地动态更改模型时，此功能尤为有用。

注意：在生产环境中启用此功能存在风险，因为用户可能参与模型适配器管理。

要启用动态 LoRA 配置，请确保环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 设置为 `True`。

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### 使用 API 端点
加载 LoRA 适配器：

要动态加载 LoRA 适配器，向 `/v1/load_lora_adapter` 端点发送 POST 请求，包含需要加载的适配器的详细信息。请求负载应包括适配器的名称和路径。

加载 LoRA 适配器的示例请求：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

如果请求成功，API 将返回 `200 OK` 状态码，`vllm serve` 返回响应体：`Success: LoRA adapter 'sql_adapter' added successfully`。如果发生错误，例如无法找到或加载适配器，将返回相应的错误消息。

卸载 LoRA 适配器：

要卸载之前加载的 LoRA 适配器，向 `/v1/unload_lora_adapter` 端点发送 POST 请求，包含要卸载的适配器的名称或 ID。

如果请求成功，API 将返回 `200 OK` 状态码，`vllm serve` 返回响应体：`Success: LoRA adapter 'sql_adapter' removed successfully`。

卸载 LoRA 适配器的示例请求：

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### 使用插件
或者，您可以使用 LoRAResolver 插件动态加载 LoRA 适配器。LoRAResolver 插件允许您从本地和远程源（如本地文件系统和 S3）加载 LoRA 适配器。在每次请求时，如果有一个尚未加载的新模型名称，LoRAResolver 将尝试解析并加载对应的 LoRA 适配器。

如果您想从不同源加载 LoRA 适配器，可以设置多个 LoRAResolver 插件。例如，您可以为本地文件设置一个解析器，为 S3 存储设置另一个解析器。vLLM 将加载它找到的第一个 LoRA 适配器。

您可以安装现有的插件或实现自己的插件。默认情况下，vLLM 提供了一个 [从本地目录加载 LoRA 适配器的解析器插件](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)。要启用此解析器，请将 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 设置为 True，将 `VLLM_PLUGINS` 设置为包含 `lora_filesystem_resolver`，然后将 `VLLM_LORA_RESOLVER_CACHE_DIR` 设置为本地目录。当 vLLM 收到使用 LoRA 适配器 `foobar` 的请求时，它将首先在本地目录中查找目录 `foobar`，并尝试将该目录的内容加载为 LoRA 适配器。如果成功，请求将正常完成，并且该适配器将在服务器上可供正常使用。

或者，按照以下示例步骤实现自己的插件：

1. 实现 LoRAResolver 接口。

    简单的 S3 LoRAResolver 实现示例：

    ```python
    import os
    import s3fs
    from vllm.lora.request import LoRARequest
    from vllm.lora.resolver import LoRAResolver

    class S3LoRAResolver(LoRAResolver):
        def __init__(self):
            self.s3 = s3fs.S3FileSystem()
            self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
            self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

        async def resolve_lora(self, base_model_name, lora_name):
            s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
            local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

            # 从 S3 下载 LoRA 到本地路径
            await self.s3._get(
                s3_path, local_path, recursive=True, maxdepth=1
            )

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_path=local_path,
                lora_int_id=abs(hash(lora_name))
            )
            return lora_request
    ```

2. 注册 `LoRAResolver` 插件。

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    有关更多详细信息，请参阅 [vLLM 的插件系统](../design/plugin_system.md)。

## `--lora-modules` 的新格式

在之前的版本中，用户通过以下格式提供 LoRA 模块，可以是键值对或 JSON 格式。例如：

```bash
--lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```

这只会包括每个 LoRA 模块的 `name` 和 `path`，但无法指定 `base_model_name`。现在，您可以使用 JSON 格式同时指定 base_model_name、name 和 path。例如：

```bash
--lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

为了提供向后兼容性支持，您仍然可以使用旧的键值格式（name=path），但在这种情况下 `base_model_name` 将保持未指定。

## 模型卡中的 LoRA 模型谱系

`--lora-modules` 的新格式主要是为了支持在模型卡中显示父模型信息。以下是您当前响应如何支持此功能的说明：

- LoRA 模型 `sql-lora` 的 `parent` 字段现在链接到其基础模型 `meta-llama/Llama-2-7b-hf`。这正确反映了基础模型与 LoRA 适配器之间的层次关系。
- `root` 字段指向 LoRA 适配器的工件位置。

```bash
$ curl http://localhost:8000/v1/models

{
    "object": "list",
    "data": [
        {
        "id": "meta-llama/Llama-2-7b-hf",
        "object": "model",
        "created": 1715644056,
        "owned_by": "vllm",
        "root": "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/",
        "parent": null,
        "permission": [
            {
            .....
            }
        ]
        },
        {
        "id": "sql-lora",
        "object": "model",
        "created": 1715644056,
        "owned_by": "vllm",
        "root": "~/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
        "parent": "meta-llama/Llama-2-7b-hf",
        "permission": [
            {
            ....
            }
        ]
        }
    ]
}
```