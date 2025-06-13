---
title: 推理输出
---
[](){ #reasoning-outputs }

vLLM 支持推理模型，例如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，这些模型设计用于生成包含推理步骤和最终结论的输出。

推理模型在其输出中会返回一个额外的 `reasoning_content` 字段，该字段包含得出最终结论的推理步骤。其他模型的输出中不包含此字段。

## 支持的模型

vLLM 当前支持以下推理模型：

| 模型系列 | 解析器名称 | 结构化输出支持 | 工具调用 |
|--------------|-------------|------------------|-------------|
| [DeepSeek R1 系列](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `deepseek_r1` | `guided_json`, `guided_regex` | ❌ |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `guided_json`, `guided_regex` | ✅ |
| [IBM Granite 3.2 语言模型](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a) | `granite` | ❌ | ❌ |
| [Qwen3 系列](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `qwen3` | `guided_json`, `guided_regex` | ✅ |

!!! note
    IBM Granite 3.2 的推理功能默认是禁用的；要启用它，必须在 `chat_template_kwargs` 中传递 `thinking=True`。
    Qwen3 系列的推理功能默认是启用的。要禁用它，必须在 `chat_template_kwargs` 中传递 `enable_thinking=False`。

## 快速入门

要使用推理模型，需要在请求聊天完成端点时指定 `--reasoning-parser` 标志。`--reasoning-parser` 标志指定用于从模型输出中提取推理内容的解析器。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

接下来，向模型发送一个请求，响应中应包含推理内容。

```python
from openai import OpenAI

# 修改 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# 第一轮
messages = [{"role": "user", "content": "9.11 和 9.8，哪个更大？"}]
# 对于 granite，添加：`extra_body={"chat_template_kwargs": {"thinking": True}}`
# 对于 Qwen3 系列，如果想在推理模式下禁用思考，添加：
# extra_body={"chat_template_kwargs": {"enable_thinking": False}}
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content:", reasoning_content)
print("content:", content)
```

`reasoning_content` 字段包含得出最终结论的推理步骤，而 `content` 字段包含最终结论。

## 流式聊天完成

推理模型也支持流式聊天完成。`reasoning_content` 字段在 [聊天完成响应块](https://platform.openai.com/docs/api-reference/chat/streaming) 的 `delta` 字段中可用。

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1694268190,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "reasoning_content": "是",
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
```

OpenAI Python 客户端库未正式支持流式输出的 `reasoning_content` 属性。但客户端支持响应中的额外属性。可以使用 `hasattr` 检查响应中是否存在 `reasoning_content` 属性。例如：

```python
from openai import OpenAI

# 修改 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = [{"role": "user", "content": "9.11 和 9.8，哪个更大？"}]
# 对于 granite，添加：`extra_body={"chat_template_kwargs": {"thinking": True}}`
# 对于 Qwen3 系列，如果想在推理模式下禁用思考，添加：
# extra_body={"chat_template_kwargs": {"enable_thinking": False}}
stream = client.chat.completions.create(model=model,
                                        messages=messages,
                                        stream=True)

print("client: 开始流式聊天完成...")
printed_reasoning_content = False
printed_content = False

for chunk in stream:
    reasoning_content = None
    content = None
    # 检查内容是 reasoning_content 还是 content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    elif hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content

    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("reasoning_content:", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    elif content is not None:
        if not printed_content:
            printed_content = True
            print("\ncontent:", end="", flush=True)
        # 提取并打印内容
        print(content, end="", flush=True)
```

请记得在访问 `reasoning_content` 之前检查其是否存在。可以查看 [示例](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py)。

## 结构化输出

推理内容也可在结构化输出中获得。像 `xgrammar` 这样的结构化输出引擎将使用推理内容生成结构化输出。目前仅在 v0 引擎中支持。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

以下是一个示例客户端：

```python
from openai import OpenAI
from pydantic import BaseModel

# 修改 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

class People(BaseModel):
    name: str
    age: int

json_schema = People.model_json_schema()

prompt = ("生成一个包含一个随机人员姓名和年龄的 JSON。")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)
```

## 工具调用

当同时启用工具调用和推理解析器时，推理内容也可用。此外，工具调用仅从 `content` 字段解析函数，而不是从 `reasoning_content` 字段。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取给定地点的当前天气",
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

print(response)
tool_call = response.choices[0].message.tool_calls[0].function

print(f"reasoning_content: {response.choices[0].message.reasoning_content}")
print(f"调用函数: {tool_call.name}")
print(f"参数: {tool_call.arguments}")
```

更多示例，请参阅 <gh-file:examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py>。

## 限制

- 推理内容仅在在线服务的聊天完成端点 (`/v1/chat/completions`) 中可用。

## 如何支持新的推理模型

您可以添加一个新的 `ReasoningParser`，类似于 <gh-file:vllm/reasoning/deepseek_r1_reasoning_parser.py>。

```python
# 导入所需的包

from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)

# 定义一个推理解析器并注册到 vLLM
# register_module 中的名称列表可用于 --reasoning-parser。
@ReasoningParserManager.register_module(["example"])
class ExampleParser(ReasoningParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        实例方法，用于从不完整的响应中提取推理内容；
        用于处理推理调用和流式传输。由于需要状态，
        必须是实例方法 - 当前的 token/diff，以及之前解析和提取的信息（参见构造函数）
        """

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        从完整的模型生成字符串中提取推理内容。

        用于非流式响应，我们在发送给客户端之前已有完整的模型响应。

        参数：
        model_output: str
            要从中提取推理内容的模型生成字符串。

        request: ChatCompletionRequest
            用于生成 model_output 的请求对象。

        返回：
        tuple[Optional[str], Optional[str]]
            包含推理内容和内容的元组。
        """
```

此外，要启用结构化输出，您需要创建一个新的 `Reasoner`，类似于 <gh-file:vllm/reasoning/deepseek_r1_reasoning_parser.py> 中的内容。

```python
@dataclass
class DeepSeekReasoner(Reasoner):
    """
    DeepSeek R 系列模型的推理器。
    """
    start_token_id: int
    end_token_id: int

    start_token: str = "<think>"
    end_token: str = "</think>"

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
        return cls(start_token_id=tokenizer.encode(
            "<think>", add_special_tokens=False)[0],
                   end_token_id=tokenizer.encode("</think>",
                                                 add_special_tokens=False)[0])

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
    ...
```

像 [xgrammar](https://github.com/mlc-ai/xgrammar) 这样的结构化输出引擎将使用 `end_token_id` 检查模型输出中是否存在推理内容，并在存在时跳过结构化输出。

最后，您可以通过使用 `--reasoning-parser` 标志为模型启用推理。

```bash
vllm serve <model_tag> --reasoning-parser example
```