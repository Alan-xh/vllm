from __future__ import annotations

import json
import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel, Field
from fastapi import UploadFile
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)


class MsgType:
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class OpenAIBaseInput(BaseModel):
    user: Optional[str] = None
    # 额外值优先于客户端上定义的值或传递给此方法的值
    extra_headers: Optional[Dict] = None
    extra_query: Optional[Dict] = None
    extra_json: Optional[Dict] = Field(None, alias="extra_body")
    timeout: Optional[float] = None

    class Config:
        extra = "allow"


class OpenAIChatInput(OpenAIBaseInput):
    messages: List[ChatCompletionMessageParam]  # 消息列表：包含对话历史或用户输入的消息，格式为 ChatCompletionMessageParam 的列表
    model: str = 'qwen-3'  # 模型名称：指定使用的语言模型，默认为 'qwen-3'
    frequency_penalty: Optional[float] = None  # 频率惩罚：控制模型重复生成相同内容的倾向，值越高越倾向于避免重复
    function_call: Optional[completion_create_params.FunctionCall] = None  # 函数调用：指定模型是否调用外部函数，包含函数调用参数
    functions: List[completion_create_params.Function] = None  # 函数列表：定义可供模型调用的外部函数列表
    logit_bias: Optional[Dict[str, int]] = None  # 逻辑偏差：用于调整特定 token 的生成概率，键为 token，值为偏差值
    logprobs: Optional[bool] = None  # 是否返回对数概率：如果为 True，模型会返回每个 token 的对数概率
    max_tokens: Optional[int] = None  # 最大 token 数：限制生成文本的最大 token 数量
    n: Optional[int] = None  # 生成数量：指定模型生成多个不同响应的数量
    presence_penalty: Optional[float] = None  # 存在惩罚：控制模型生成新内容的倾向，值越高越倾向于生成新颖内容
    response_format: completion_create_params.ResponseFormat = None  # 响应格式：指定模型输出的格式，例如 JSON 或文本
    seed: Optional[int] = None  # 随机种子：用于控制生成结果的随机性，固定种子可重现结果
    stop: Union[Optional[str], List[str]] = None  # 停止标志：指定生成停止的字符串或字符串列表
    stream: Optional[bool] = None  # 是否流式输出：如果为 True，模型将以流式方式逐部分返回结果
    temperature: Optional[float] = 0.7  # 温度参数：控制生成内容的随机性，值越高生成越随机，默认为 0.7
    tool_choice: Optional[Union[ChatCompletionToolChoiceOptionParam, str]] = None  # 工具选择：指定模型使用的工具或工具选择策略
    tools: List[Union[ChatCompletionToolParam, str]] = None  # 工具列表：定义模型可用的工具列表
    top_logprobs: Optional[int] = None  # 顶部对数概率：指定返回最高概率的 token 数量
    top_p: Optional[float] = None  # Top-p 采样：控制生成内容的多样性，值越小越聚焦于高概率 token


class OpenAIEmbeddingsInput(OpenAIBaseInput):
    input: Union[str, List[str]]
    model: str
    dimensions: Optional[int] = None
    encoding_format: Optional[Literal["float", "base64"]] = None


class OpenAIImageBaseInput(OpenAIBaseInput):
    model: str
    n: int = 1
    response_format: Optional[Literal["url", "b64_json"]] = None
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = "256x256"


class OpenAIImageGenerationsInput(OpenAIImageBaseInput):
    prompt: str
    quality: Literal["standard", "hd"] = None
    style: Optional[Literal["vivid", "natural"]] = None


class OpenAIImageVariationsInput(OpenAIImageBaseInput):
    image: Union[UploadFile, AnyUrl]


class OpenAIImageEditsInput(OpenAIImageVariationsInput):
    prompt: str
    mask: Union[UploadFile, AnyUrl]


class OpenAIAudioTranslationsInput(OpenAIBaseInput):
    file: Union[UploadFile, AnyUrl]
    model: str
    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: float = 0.7


class OpenAIAudioTranscriptionsInput(OpenAIAudioTranslationsInput):
    language: Optional[str] = None
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None


class OpenAIAudioSpeechInput(OpenAIBaseInput):
    input: str
    model: str
    voice: str
    response_format: Optional[
        Literal["mp3", "opus", "aac", "flac", "pcm", "wav"]
    ] = None
    speed: Optional[float] = None


class OpenAIFileInput(OpenAIBaseInput):
    file: UploadFile # FileTypes
    purpose: Literal["fine-tune", "assistants"] = "assistants"


class OpenAIBaseOutput(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    object: Literal[
        "chat.completion", "chat.completion.chunk"
    ] = "chat.completion.chunk"
    role: Literal["assistant"] = "assistant"
    finish_reason: Optional[str] = None
    created: int = Field(default_factory=lambda: int(time.time()))
    tool_calls: List[Dict] = []

    status: Optional[int] = None  # AgentStatus
    message_type: int = MsgType.TEXT
    message_id: Optional[str] = None  # 数据库表中的 id
    is_ref: bool = False  #天气显示在单独的扩展器中

    class Config:
        extra = "allow"

    def model_dump(self) -> dict:
        result = {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "created": self.created,
            "status": self.status,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "is_ref": self.is_ref,
            **(self.model_extra or {}),
        }

        if self.object == "chat.completion.chunk":
            result["choices"] = [
                {
                    "delta": {
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    },
                    "role": self.role,
                }
            ]
        elif self.object == "chat.completion":
            result["choices"] = [
                {
                    "message": {
                        "role": self.role,
                        "content": self.content,
                        "finish_reason": self.finish_reason,
                        "tool_calls": self.tool_calls,
                    }
                }
            ]
        return result

    def model_dump_json(self):
        return json.dumps(self.model_dump(), ensure_ascii=False)


class OpenAIChatOutput(OpenAIBaseOutput):
    ...