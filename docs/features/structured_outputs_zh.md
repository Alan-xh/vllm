---
title: 结构化输出
---
[](){ #structured-outputs }

vLLM 支持使用 [xgrammar](https://github.com/mlc-ai/xgrammar) 或 [guidance](https://github.com/guidance-ai/llguidance) 作为后端生成结构化输出。本文档展示了一些可用的不同选项的示例，用于生成结构化输出。

## 在线服务（OpenAI API）

您可以使用 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat](https://platform.openai.com/docs/api-reference/chat) API 生成结构化输出。

支持以下参数，这些参数必须作为额外参数添加：

- `guided_choice`：输出将严格为选项之一。
- `guided_regex`：输出将遵循正则表达式模式。
- `guided_json`：输出将遵循 JSON 模式。
- `guided_grammar`：输出将遵循上下文无关文法。
- `structural_tag`：在生成的文本中的指定标签内遵循 JSON 模式。

您可以在 [OpenAI 兼容服务器][openai-compatible-server] 页面查看支持的参数完整列表。

结构化输出在 OpenAI 兼容服务器中默认支持。您可以通过设置 `vllm serve` 的 `--guided-decoding-backend` 标志来选择使用的后端。默认后端为 `auto`，它会根据请求的细节尝试选择适当的后端。您也可以选择特定的后端，并附带一些选项。完整的选项列表可在 `vllm serve --help` 文本中找到。

现在让我们来看每个案例的示例，从最简单的 `guided_choice` 开始：

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "分类此情感：vLLM 很棒！"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)
```

下一个示例展示如何使用 `guided_regex`。目标是根据一个简单的正则表达式模板生成一个电子邮件地址：

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "为在 Enigma 工作的 Alan Turing 生成一个示例电子邮件地址，以 .com 结尾并换行。例如：alan.turing@enigma.com\n",
        }
    ],
    extra_body={"guided_regex": r"\w+@\w+\.com\n", "stop": ["\n"]},
)
print(completion.choices[0].message.content)
```

结构化文本生成中最相关的功能之一是生成具有预定义字段和格式的有效 JSON。为此，我们可以通过两种方式使用 `guided_json` 参数：

- 直接使用 [JSON Schema](https://json-schema.org/)
- 定义一个 [Pydantic 模型](https://docs.pydantic.dev/latest/)，然后从中提取 JSON Schema（通常是更简单的选项）。

下一个示例展示如何使用 Pydantic 模型与 `guided_json` 参数：

```python
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "生成一个 JSON，包含 90 年代最具标志性汽车的品牌、型号和类型",
        }
    ],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)
```

!!! tip
    虽然不是严格必须的，但通常在提示中指明 JSON 模式以及字段的填充方式会显著改善结果。

最后，我们有 `guided_grammar` 选项，这可能是最难使用的，但功能非常强大。它允许我们定义像 SQL 查询这样的完整语言。它通过使用上下文无关的 EBNF 文法工作。例如，我们可以用来定义简化 SQL 查询的特定格式：

```python
simplified_sql_grammar = """
    root ::= select_statement

    select_statement ::= "SELECT " column " from " table " where " condition

    column ::= "col_1 " | "col_2 "

    table ::= "table_1 " | "table_2 "

    condition ::= column "= " number

    number ::= "1 " | "2 "
"""

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "生成一个 SQL 查询，显示 'users' 表中的 'username' 和 'email'。",
        }
    ],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)
```

完整示例：<gh-file:examples/online_serving/openai_chat_completion_structured_outputs.py>

## 实验性自动解析（OpenAI API）

本节介绍 OpenAI 对 `client.chat.completions.create()` 方法的 beta 包装器，提供与 Python 特定类型的更丰富集成。

在撰写本文时（`openai==1.54.4`），这是 OpenAI 客户端库中的一个“beta”功能。代码参考可在 [此处](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104) 找到。

以下是一个使用 Pydantic 模型获取结构化输出的简单示例：

```python
from pydantic import BaseModel
from openai import OpenAI

class Info(BaseModel):
    name: str
    age: int

client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
completion = client.beta.chat.completions.parse(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "我的名字是 Cameron，我 28 岁。我的名字和年龄是什么？"},
    ],
    response_format=Info,
    extra_body=dict(guided_decoding_backend="outlines"),
)

message = completion.choices[0].message
print(message)
assert message.parsed
print("姓名：", message.parsed.name)
print("年龄：", message.parsed.age)
```

输出：

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
姓名：Cameron
年龄：28
```

以下是使用嵌套 Pydantic 模型处理逐步数学解法的更复杂示例：

```python
from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str

client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
completion = client.beta.chat.completions.parse(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的专业数学导师。"},
        {"role": "user", "content": "解方程 8x + 31 = 2。"},
    ],
    response_format=MathResponse,
    extra_body=dict(guided_decoding_backend="outlines"),
)

message = completion.choices[0].message
print(message)
assert message.parsed
for i, step in enumerate(message.parsed.steps):
    print(f"步骤 #{i}：", step)
print("答案：", message.parsed.final_answer)
```

输出：

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "首先，让我们隔离包含变量 'x' 的项。为此，我们从方程两边减去 31。", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "通过从两边减去 31，我们将方程简化为 8x = -29。", "output": "8x = -29"}, { "explanation": "接下来，通过将方程两边除以 8 来隔离 'x'。", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="首先，让我们隔离包含变量 'x' 的项。为此，我们从方程两边减去 31。", output='8x + 31 - 31 = 2 - 31'), Step(explanation='通过从两边减去 31，我们将方程简化为 8x = -29。', output='8x = -29'), Step(explanation="接下来，通过将方程两边除以 8 来隔离 'x'。", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
步骤 #0：explanation="首先，让我们隔离包含变量 'x' 的项。为此，我们从方程两边减去 31。" output='8x + 31 - 31 = 2 - 31'
步骤 #1：explanation='通过从两边减去 31，我们将方程简化为 8x = -29。' output='8x = -29'
步骤 #2：explanation="接下来，通过将方程两边除以 8 来隔离 'x'。" output='8x / 8 = -29 / 8'
答案：x = -29/8
```

`structural_tag` 的示例可在以下链接找到：<gh-file:examples/online_serving/openai_chat_completion_structured_outputs_structural_tag.py>

## 离线推理

离线推理允许使用相同类型的引导解码。要使用它，我们需要使用 `SamplingParams` 中的 `GuidedDecodingParams` 类来配置引导解码。`GuidedDecodingParams` 中的主要可用选项包括：

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

这些参数可以以与上述在线服务示例相同的方式使用。以下是使用 `choice` 参数的一个示例：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
outputs = llm.generate(
    prompts="分类此情感：vLLM 很棒！",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

完整示例：<gh-file:examples/offline_inference/structured_outputs.py>