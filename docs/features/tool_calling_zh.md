# 工具调用

vLLM 当前支持命名的函数调用，以及聊天完成 API 中 `tool_choice` 字段的 `auto`、`required`（自 `vllm>=0.8.3` 起）和 `none` 选项。

## 快速入门

启动启用工具调用的服务器。此示例使用 Meta 的 Llama 3.1 8B 模型，因此需要使用 vLLM 示例目录中的 llama3 工具调用聊天模板：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

接下来，向模型发送一个请求，使其使用可用的工具：

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"获取 {location} 的天气，使用 {unit} 单位..."
tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定地点的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市和州，例如 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "旧金山的天气如何？"}],
    tools=tools,
    tool_choice="auto"
)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"调用函数：{tool_call.name}")
print(f"参数：{tool_call.arguments}")
print(f"结果：{get_weather(**json.loads(tool_call.arguments))}")
```

示例输出：

```text
调用函数：get_weather
参数：{"location": "San Francisco, CA", "unit": "fahrenheit"}
结果：获取 San Francisco, CA 的天气，使用 fahrenheit 单位...
```

此示例展示了：

* 启用工具调用并设置服务器
* 定义处理工具调用的实际函数
* 使用 `tool_choice="auto"` 进行请求
* 处理结构化响应并执行相应的函数

您还可以通过设置 `tool_choice={"type": "function", "function": {"name": "get_weather"}}` 使用命名函数调用指定特定函数。请注意，这将使用引导解码后端，因此首次使用时会因 FSM 编译而有几秒（或更长）的延迟，之后会缓存以供后续请求使用。

请记住，调用者的责任是：

1. 在请求中定义适当的工具
2. 在聊天消息中包含相关上下文
3. 在应用程序逻辑中处理工具调用

有关更高级的用法，包括并行工具调用和不同模型特定的解析器，请参阅以下部分。

## 命名函数调用

vLLM 在聊天完成 API 中默认支持命名函数调用。它通过 Outlines 的引导解码实现，因此默认启用，并适用于任何支持的模型。您可以保证获得一个有效可解析的函数调用，但不一定是高质量的。

vLLM 将使用引导解码确保响应匹配 `tools` 参数中定义的 JSON 模式中的工具参数对象。为了获得最佳结果，建议在提示中指定预期的输出格式/模式，以确保模型的预期生成与引导解码后端强制生成的模式一致。

要使用命名函数，您需要在聊天完成请求的 `tools` 参数中定义函数，并在 `tool_choice` 参数中指定其中一个工具的 `name`。

## 必需函数调用

vLLM 支持聊天完成 API 中的 `tool_choice='required'` 选项。与命名函数调用类似，它也使用引导解码，因此默认启用并适用于任何支持的模型。必需的引导解码功能（带 `anyOf` 的 JSON 模式）目前仅在 V0 引擎的 `outlines` 引导解码后端中支持。然而，V1 引擎对替代解码后端的支持已在[路线图](https://docs.vllm.ai/en/latest/usage/v1_guide.html#feature-model)上。

当设置 `tool_choice='required'` 时，模型保证根据 `tools` 参数中指定的工具列表生成一个或多个工具调用。工具调用的数量取决于用户的查询。输出格式严格遵循 `tools` 参数中定义的模式。

## 自动函数调用

要启用此功能，您应设置以下标志：

* `--enable-auto-tool-choice` -- **必需** 自动工具选择。告诉 vLLM 您希望模型在认为合适时生成自己的工具调用。
* `--tool-call-parser` -- 选择要使用的工具解析器（如下列出）。未来将继续添加更多工具解析器，您还可以通过 `--tool-parser-plugin` 注册自己的工具解析器。
* `--tool-parser-plugin` -- **可选** 用于将用户定义的工具解析器注册到 vLLM 的工具解析器插件，注册的工具解析器名称可在 `--tool-call-parser` 中指定。
* `--chat-template` -- **可选** 用于自动工具选择。指定处理 `tool` 角色消息和包含先前生成工具调用的 `assistant` 角色消息的聊天模板路径。Hermes、Mistral 和 Llama 模型在其 `tokenizer_config.json` 文件中有与工具兼容的聊天模板，但您可以指定自定义模板。如果模型的 `tokenizer_config.json` 中配置了特定于工具使用的聊天模板，此参数可以设置为 `tool_use`。在这种情况下，将按照 `transformers` 规范使用。更多信息请参阅 HuggingFace 的[文档](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)；您可以在[这里](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)找到 `tokenizer_config.json` 的示例。

如果您喜欢的工具调用模型不受支持，请随时贡献解析器和工具使用聊天模板！

### Hermes 模型 (`hermes`)

所有比 Hermes 2 Pro 更新的 Nous Research Hermes 系列模型都应受支持。

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_注意：由于创建过程中的合并步骤，Hermes 2 **Theta** 模型的工具调用质量和能力已知有所下降_。

标志：`--tool-call-parser hermes`

### Mistral 模型 (`mistral`)

支持的模型：

* `mistralai/Mistral-7B-Instruct-v0.3`（已确认）
* 其他支持函数调用的 Mistral 模型也兼容。

已知问题：

1. Mistral 7B 在正确生成并行工具调用方面存在困难。
2. Mistral 的 `tokenizer_config.json` 聊天模板要求工具调用 ID 必须正好为 9 位数字，这比 vLLM 生成的 ID 短得多。由于不满足此条件时会抛出异常，因此提供了以下额外的聊天模板：

* <gh-file:examples/tool_chat_template_mistral.jinja> - 这是“官方”Mistral 聊天模板，但进行了调整，使其与 vLLM 的工具调用 ID 兼容（将 `tool_call_id` 字段截断为最后 9 位）。
* <gh-file:examples/tool_chat_template_mistral_parallel.jinja> - 这是一个“更好”的版本，当提供工具时会添加工具使用系统提示，从而在并行工具调用时显著提高可靠性。

推荐标志：`--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 模型 (`llama3_json`)

支持的模型：

所有 Llama 3.1、3.2 和 4 模型都应受支持。

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

支持的工具调用是[基于 JSON 的工具调用](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)。对于 Llama-3.2 模型引入的[基于 Python 的工具调用](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling)，请参见下面的 `pythonic` 工具解析器。对于 Llama 4 模型，建议使用 `llama4_pythonic` 工具解析器。

其他工具调用格式，如内置的 Python 工具调用或自定义工具调用，不受支持。

已知问题：

1. Llama 3 不支持并行工具调用，但 Llama 4 模型支持。
2. 模型可能生成格式错误的参数，例如将数组序列化为字符串而不是数组。

vLLM 为 Llama 3.1 和 3.2 提供了两个基于 JSON 的聊天模板：

* <gh-file:examples/tool_chat_template_llama3.1_json.jinja> - 这是 Llama 3.1 模型的“官方”聊天模板，但进行了调整以更好地与 vLLM 配合使用。
* <gh-file:examples/tool_chat_template_llama3.2_json.jinja> - 这在 Llama 3.1 聊天模板基础上扩展，增加了对图像的支持。

推荐标志：`--tool-call-parser llama3_json --chat-template {见上文}`

vLLM 还为 Llama 4 提供了一个基于 Python 和 JSON 的聊天模板，但推荐使用 Python 工具调用：
* <gh-file:examples/tool_chat_template_llama4_pythonic.jinja> - 这是基于 Llama 4 模型的[官方聊天模板](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)。

对于 Llama 4 模型，使用 `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja`。

#### IBM Granite

支持的模型：

* `ibm-granite/granite-3.0-8b-instruct`

推荐标志：`--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

<gh-file:examples/tool_chat_template_granite.jinja>：这是从 HuggingFace 上的原始模板修改而来的聊天模板。支持并行函数调用。

* `ibm-granite/granite-3.1-8b-instruct`

推荐标志：`--tool-call-parser granite`

可以直接使用 HuggingFace` 上的聊天模板。支持并行函数调用。

* `ibm-granite/granite-20b-functioncalling`

推荐标志：`--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

<gh-file:examples/tool_chat_template_granite_20b_fc.jinja>：这是从 HuggingFace 上的原始聊天模板修改而来的版本，因为原始版本与 vLLM 不兼容。它融合了 Hermes 模板中的功能描述元素，并遵循[论文](https://arxiv.org/abs/2407.00121/) 中“响应生成模式”的系统提示方式。支持并行函数调用。

#### InternLM 模型 (`internlm`)

支持的模型：

* `internlm/internlm2_5-7b-chat`（已确认）
* 其他支持 internlm2.5 的功能调用模型也兼容。

已知问题：

* 虽然此实现也支持 InternLM2，但在测试 `internlm/internlm2-chat-7b 模型时，工具调用结果不稳定。

推荐标志：`--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

#### Jamba 模型 (`jamba`)

AI21 的 Jamba-1.5 模型受支持。

* `ai21labs/AI21-Jamba-`

标志：`--tool-call-parser jamba`

#### Qwen 模型

对于 Qwen2.5，`tokenizer_config.json` 中的聊天模板已包含对 Hermes 风格工具使用的支持。因此，您可以使用 `hermes` 解析器为 Qwen 模型启用工具调用。更多详细信息，请参考官方 [Qwen 文档](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)

* `Qwen/Qwen2.5-*`

* `Qwen/QwQ-32B`

标志：`--tool-call-parser hermes`

#### DeepSeek-V3 模型 (`deepseek_v3`)

支持的模型：

* `deepseek-ai/DeepSeek-V3-0324`（使用 `<gh-file:examples/tool_chat_template_deepseekv3.jinja>`）

* `deepseek-ai/DeepSeek-R1-0528`（使用 `<gh-file:examples/tool_chat_template_deepseekr1.jinja>`）

标志：`--tool-call-parser deepseek_v3 --chat-template {见上文}`

#### 支持 Python 的模型调用 (`pythonic`)

越来越多的模型使用 Python 列表来表示工具调用，而不是使用 JSON。这具有支持并行工具调用的天然优势，并消除了关于工具调用所需 JSON 模式的歧义。`py` 工具解析器可以支持此类模型。

作为一个具体的例子，这些模型可能通过以下方式查询旧金山和西雅图的天气：

```python
[get_weather(city='San Francisco', metric='celsius')， get_weather(city='Seattle')， metric='celsius')]
```

限制：

* 模型不得在同一生成中同时生成文本和工具调用。对于特定模型来说，这可能不难改变，但目前社区对工具调用开始和结束时应发出哪些 token 尚未达成共识。（特别是，Llama 3.2 模型不发出此类 token。）

* Llama 的较小模型在使用工具方面效果有限。

支持的模型：

* `meta-llama/Llama-3.2-1B-Instruct`\*（使用 `<gh-file:examples/tool_chat_template_llama3.2_pythonic.jinja>`）

* `meta-llama/Llama-3.2-3B-Instruct`\*（使用 `<gh-file:examples/tool_chat_template_llama3.2_pythonic.jinja>`）

* `Team-ACE/ToolACE-8B`（使用 `<gh-file:examples/tool_chat_template_toolace.jinja>`）

* `fixie-ai/ultravox-v0_4-ToolACE-8B`（使用 `<gh-file:examples/tool_chat_template_toolace.jinja>`）

* `meta-llama/Llama-4-Scout-17B-16E-Instruct`\*（使用 `<gh-file:examples/tool_chat_template_llama4_pythonic.jinja>`）

* `meta-llama/Llama-4-Maverick-17B-128E-Instruct`\*（使用 `<gh-file:examples/tool_chat_template_llama4_pythonic.jinja>`）

标志：`--tool-call-parser pythonic --chat-template {见上文}`

---

**警告**

Llama 的较小模型经常无法以正确格式发出工具调用。您的体验可能会有所不同。

---

## 如何编写工具解析器插件

工具解析器插件是一个包含一个或多个 ToolParser 实现的 Python 文件。您可以编写类似于 `<gh-file:vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py>` 中的 `Hermes2ProToolParser` 的工具解析器。

以下是插件文件的摘要：

```python
# 导入所需的包

# 定义工具解析器并将其注册到 vllm
# 注册模块中的名称列表可以在
# --tool-call-parser 中使用。您可以在这里定义多个
# 工具解析器。
@ToolParserManager.register_module(["example"])
class ExampleToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(self, tokenizer)

    # 调整请求。例如，将 skip_special_tokens 设置为 False
    # 用于工具调用输出。
    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest：
        return request

    # 实现流式调用的工具调用解析
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        return delta

    # 实现非流式调用的工具解析
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation：
        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=text)

```

然后，您可以在命令行中这样使用此插件：

```console
    --enable-auto-tool-choice \
    --tool-parser-plugin <插件文件的绝对路径>
    --tool-call-parser example \
    --chat-template <您的聊天模板> \
```