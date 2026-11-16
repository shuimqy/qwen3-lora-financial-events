from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=12000, description="完整公告原文")

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class Event(BaseModel):
    event_type: Literal["企业收购", "股份回购"]
    arguments: dict[str, list[str]]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    finish_reason: Literal["eos", "length"]
    latency_ms: float


class ExtractResponse(BaseModel):
    request_id: str
    status: Literal["ok", "invalid_output", "length_limit"]
    valid_json_schema: bool
    events: list[Event] | None
    raw_output: str
    validation_error: str | None
    warnings: list[dict]
    usage: Usage


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain non-whitespace characters")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己。"}],
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.8,
    }]})
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    max_new_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    adapter_mode: Literal["finetuned", "base"] = "finetuned"

    @field_validator("messages")
    @classmethod
    def validate_history(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if sum(len(message.content) for message in messages) > 24000:
            raise ValueError("Total message content exceeds 24000 characters")
        conversation = messages[1:] if messages[0].role == "system" else messages
        if not conversation or conversation[-1].role != "user":
            raise ValueError("Conversation must end with a user message")
        for index, message in enumerate(conversation):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("After the optional system message, alternate user and assistant")
        return messages


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatResponse(BaseModel):
    request_id: str
    model: str
    lora_enabled: bool
    message: AssistantMessage
    usage: Usage
