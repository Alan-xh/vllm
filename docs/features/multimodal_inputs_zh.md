---
title: 多模态输入
---
[](){ #multimodal-inputs }

本页面教您如何将多模态输入传递给 vLLM 中的[多模态模型][supported-mm-models]。

!!! note
    我们正在积极迭代多模态支持。有关即将推出的更改，请参见[此 RFC](gh-issue:4194)，
    如果您有任何反馈或功能请求，请在 GitHub 上[开启一个 issue](https://github.com/vllm-project/vllm/issues/new/choose)。

## 离线推理

要输入多模态数据，请遵循 [vllm.inputs.PromptType][] 中的以下模式：

- `prompt`：提示应遵循 HuggingFace 上记录的格式。
- `multi_modal_data`：这是一个遵循 [vllm.multimodal.inputs.MultiModalDataDict][] 定义模式的字典。

### 图像输入

您可以将单个图像传递到多模态字典的 `'image'` 字段，如以下示例所示：

```python
from vllm import LLM

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# 请参考 HuggingFace 仓库以使用正确的格式
prompt = "USER: <image>\n此图像的内容是什么？\nASSISTANT:"

# 使用 PIL.Image 加载图像
image = PIL.Image.open(...)

# 单提示推理
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

# 批量推理
image_1 = PIL.Image.open(...)
image_2 = PIL.Image.open(...)
outputs = llm.generate(
    [
        {
            "prompt": "USER: <image>\n此图像的内容是什么？\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\n此图像的颜色是什么？\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        }
    ]
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

完整示例：<gh-file:examples/offline_inference/vision_language.py>

要在同一文本提示中替换多个图像，您可以传入一个图像列表，如下所示：

```python
from vllm import LLM

llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,  # 需要加载 Phi-3.5-vision
    max_model_len=4096,  # 否则可能无法适应较小的 GPU
    limit_mm_per_prompt={"image": 2},  # 接受的最大数量
)

# 请参考 HuggingFace 仓库以使用正确的格式
prompt = "<|user|>\n<|image_1|>\n<|image_2|>\n每张图像的内容是什么？<|end|>\n<|assistant|>\n"

# 使用 PIL.Image 加载图像
image1 = PIL.Image.open(...)
image2 = PIL.Image.open(...)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": [image1, image2]
    },
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

完整示例：<gh-file:examples/offline_inference/vision_language_multi_image.py>

多图像输入可以扩展为视频字幕生成。我们以 [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 为例展示这一点，因为它支持视频：

```python
from vllm import LLM

# 指定视频的最大帧数为 4。此值可以更改。
llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

# 创建请求负载。
video_frames = ... # 加载您的视频，确保帧数仅为之前指定的数量。
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这组帧。假设这些帧属于同一视频。"},
    ],
}
for i in range(len(video_frames)):
    base64_image = encode_image(video_frames[i]) # base64 编码。
    new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    message["content"].append(new_image)

# 执行推理并记录输出。
outputs = llm.chat([message])

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

### 视频输入

您可以直接将 NumPy 数组列表传递到多模态字典的 `'video'` 字段，而不是使用多图像输入。

完整示例：<gh-file:examples/offline_inference/vision_language.py>

### 音频输入

您可以将元组 `(array, sampling_rate)` 传递到多模态字典的 `'audio'` 字段。

完整示例：<gh-file:examples/offline_inference/audio_language.py>

### 嵌入输入

要直接将预计算的嵌入（属于图像、视频或音频数据类型）输入到语言模型中，
请将形状为 `(num_items, feature_size, 语言模型的隐藏大小)` 的张量传递到多模态字典的相应字段。

```python
from vllm import LLM

# 使用图像嵌入作为输入进行推理
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# 请参考 HuggingFace 仓库以使用正确的格式
prompt = "USER: <image>\n此图像的内容是什么？\nASSISTANT:"

# 单张图像的嵌入
# 形状为 (1, image_feature_size, 语言模型的隐藏大小) 的 torch.Tensor
image_embeds = torch.load(...)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image_embeds},
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

对于 Qwen2-VL 和 MiniCPM-V，我们接受与嵌入一起的附加参数：

```python
# 根据您的模型构建提示
prompt = ...

# 多张图像的嵌入
# 形状为 (num_images, image_feature_size, 语言模型的隐藏大小) 的 torch.Tensor
image_embeds = torch.load(...)

# Qwen2-VL
llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})
mm_data = {
    "image": {
        "image_embeds": image_embeds,
        # image_grid_thw 需要用于计算位置编码。
        "image_grid_thw": torch.load(...),  # 形状为 (1, 3) 的 torch.Tensor
    }
}

# MiniCPM-V
llm = LLM("openbmb/MiniCPM-V-2_6", trust_remote_code=True, limit_mm_per_prompt={"image": 4})
mm_data = {
    "image": {
        "image_embeds": image_embeds,
        # image_sizes 需要用于计算切片图像的细节。
        "image_sizes": [image.size for image in images],  # 图像大小列表
    }
}

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": mm_data,
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

## 在线服务

我们的 OpenAI 兼容服务器通过 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 接受多模态数据。

!!! warning
    使用 Chat Completions API **需要**聊天模板。
    对于 HF 格式模型，默认聊天模板定义在 `chat_template.json` 或 `tokenizer_config.json` 中。

    如果没有默认聊天模板，我们将首先查找 <gh-file:vllm/transformers_utils/chat_templates/registry.py> 中的内置回退模板。
    如果没有回退模板，将会抛出错误，您需要通过 `--chat-template` 参数手动提供聊天模板。

    对于某些模型，我们在 <gh-dir:vllm/examples> 中提供了替代聊天模板。
    例如，VLM2Vec 使用 <gh-file:examples/template_vlm2vec.jinja>，这与 Phi-3-Vision 的默认模板不同。

### 图像输入

图像输入根据 [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) 提供支持。
以下是使用 Phi-3.5-Vision 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt '{"image":2}'
```

然后，您可以按以下方式使用 OpenAI 客户端：

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 单图像输入推理
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            # 注意：不需要使用图像标记 `<image>` 进行提示格式化，
            # 因为提示将由 API 服务器自动处理。
            {"type": "text", "text": "这张图片里有什么？"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
)
print("聊天完成输出：", chat_response.choices[0].message.content)

# 多图像输入推理
image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这些图片中的动物是什么？"},
            {"type": "image_url", "image_url": {"url": image_url_duck}},
            {"type": "image_url", "image_url": {"url": image_url_lion}},
        ],
    }],
)
print("聊天完成输出：", chat_response.choices[0].message.content)
```

完整示例：<gh-file:examples/online_serving/openai_chat_completion_client_for_multimodal.py>

!!! tip
    vLLM 还支持从本地文件路径加载：您可以在启动 API 服务器/引擎时通过 `--allowed-local-media-path` 指定允许的本地媒体路径，
    并在 API 请求中将文件路径作为 `url` 传递。

!!! tip
    无需在 API 请求的文本内容中放置图像占位符 - 它们已经由图像内容表示。
    实际上，您可以通过交错文本和图像内容，在文本中间放置图像占位符。

!!! note
    默认情况下，通过 HTTP URL 获取图像的超时时间为 `5` 秒。
    您可以通过设置环境变量来覆盖此设置：

    ```console
    export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
    ```

### 视频输入

您可以使用 `video_url` 传递视频文件，而不是 `image_url`。以下是使用 [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --task generate --max-model-len 8192
```

然后，您可以按以下方式使用 OpenAI 客户端：

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"

## 在负载中使用视频 URL
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "这个视频里有什么？"
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("来自视频 URL 的聊天完成输出：", result)
```

完整示例：<gh-file:examples/online_serving/openai_chat_completion_client_for_multimodal.py>

!!! note
    默认情况下，通过 HTTP URL 获取视频的超时时间为 `30` 秒。
    您可以通过设置环境变量来覆盖此设置：

    ```console
    export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>
    ```

### 音频输入

音频输入根据 [OpenAI Audio API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 提供支持。
以下是使用 Ultravox-v0.5-1B 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

然后，您可以按以下方式使用 OpenAI 客户端：

```python
import base64
import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset

def encode_base64_content_from_url(content_url: str) -> str:
    """将从远程 URL 获取的内容编码为 base64 格式。"""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 支持 librosa 支持的任何格式
audio_url = AudioAsset("winning_call").url
audio_base64 = encode_base64_content_from_url(audio_url)

chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "这个音频里有什么？"
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "wav"
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)

result = chat_completion_from_base64.choices[0].message.content
print("来自输入音频的聊天完成输出：", result)
```

或者，您可以使用 `audio_url`，这是图像输入的 `image_url` 的音频对应版本：

```python
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "这个音频里有什么？"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("来自音频 URL 的聊天完成输出：", result)
```

完整示例：<gh-file:examples/online_serving/openai_chat_completion_client_for_multimodal.py>

!!! note
    默认情况下，通过 HTTP URL 获取音频的超时时间为 `10` 秒。
    您可以通过设置环境变量来覆盖此设置：

    ```console
    export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>
    ```

### 嵌入输入处理

要直接将预计算的嵌入（属于图像、视频或音频数据类型）输入到语言模型中，
请将形状张量传递到多模态字典的相应字段。
#### 图像嵌入输入
对于图像嵌入，您可以将 base64 编码的张量传递到 `image_embeds` 字段。
以下示例展示了如何将图像嵌入传递到 OpenAI 服务器：

```python
image_embedding = torch.load(...)
grid_thw = torch.load(...) # Qwen/Qwen2-VL-2B-Instruct 所需

buffer = io.BytesIO()
torch.save(image_embedding, buffer)
buffer.seek(0)
binary_data = buffer.read()
base64_image_embedding = base64.b64encode(binary_data).decode('utf-8')

client = OpenAI(
    # 默认为 os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 基本用法 - 这等价于离线推理的 LLaVA 示例
model = "llava-hf/llava-1.5-7b-hf"
embeds =  {
    "type": "image_embeds",
    "image_embeds": f"{base64_image_embedding}" 
}

# 传递附加参数（适用于 Qwen2-VL 和 MiniCPM-V）
model = "Qwen/Qwen2-VL-2B-Instruct"
embeds =  {
    "type": "image_embeds",
    "image_embeds": {
        "image_embeds": f"{base64_image_embedding}" , # 必需
        "image_grid_thw": f"{base64_image_grid}" # Qwen/Qwen2-VL-2B-Instruct 所需
    },
}
model = "openbmb/MiniCPM-V-2_6"
embeds =  {
    "type": "image_embeds",
    "image_embeds": {
        "image_embeds": f"{base64_image_embedding}" , # 必需
        "image_sizes": f"{base64_image_sizes}"  # openbmb/MiniCPM-V-2_6 所需
    } 
}
chat_completion = client.chat.completions.create(
    messages=[
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": [
        {
            "type": "text",
            "text": "这张图片里有什么？",
        },
        embeds,
        ],
    },
],
    model=model,
)
```

!!! note
    仅一个消息可以包含 `{"type": "image_embeds"}`。
    如果与需要附加参数的模型一起使用，您必须为每个参数提供张量，例如 `image_grid_thw`、`image_sizes` 等。