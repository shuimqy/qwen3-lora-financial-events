from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(StrEnum):
    CHAT = "CHAT"
    EXTRACT = "EXTRACT"
    CLARIFY = "CLARIFY"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "messages": [{"role": "user", "content": "请提取这篇公告的收购方和金额。"}],
        "document": "甲公司公告，拟以1亿元收购乙公司100%股权。",
    }]})
    messages: list[Message] = Field(min_length=1, max_length=24)
    document: str | None = Field(default=None, min_length=1, max_length=12000)

    @field_validator("messages")
    @classmethod
    def valid_history(cls, messages):
        if messages[-1].role != "user":
            raise ValueError("The last message must be from the user")
        if sum(len(message.content) for message in messages) > 24000:
            raise ValueError("Conversation exceeds 24000 characters")
        return messages

    @field_validator("document")
    @classmethod
    def nonblank_document(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Document must not be blank")
        return value


class IntentResponse(BaseModel):
    request_id: str
    intent: Intent
    source: Literal["rule", "llm", "fallback"]
    reason: str
    llm_used: bool
    needs_document: bool
    document_source: Literal["provided", "current_message", "history", "none"]
    clarification: str | None
    raw_label: str | None = None


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str | None = Field(default=None, min_length=1, max_length=12000)
    awaiting_document: bool = False

    @field_validator("document")
    @classmethod
    def nonblank_document(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Document must not be blank")
        return value


class ConversationRequest(IntentRequest):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "messages": [{"role": "user", "content": "你好！"}],
    }, {
        "messages": [{"role": "user", "content": "请提取这篇公告的事件字段。"}],
        "document": "甲公司公告，拟以1亿元收购乙公司100%股权。",
    }]})
    context: ConversationContext = Field(default_factory=ConversationContext)
    max_new_tokens: int = Field(default=256, ge=1, le=512)

    @field_validator("messages")
    @classmethod
    def alternating_history(cls, messages):
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("Messages must alternate user and assistant, starting with user")
        return messages


class ModelUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    finish_reason: Literal["eos", "length"]
    latency_ms: float = Field(ge=0)


class ExtractionEvent(BaseModel):
    event_type: Literal["企业收购", "股份回购"]
    arguments: dict[str, list[str]]


class ExtractionResult(BaseModel):
    status: Literal["ok", "invalid_output", "length_limit"]
    valid_json_schema: bool
    events: list[ExtractionEvent] | None
    raw_output: str
    validation_error: str | None
    warnings: list[dict]
    usage: ModelUsage


class ConversationResponse(BaseModel):
    request_id: str
    message: Message
    intent: Intent
    source: Literal["rule", "llm", "fallback"]
    route: Literal["base_chat", "lora_extract", "clarification"]
    status: Literal["ok", "needs_document", "clarification", "invalid_output", "length_limit"]
    context: ConversationContext
    extraction: ExtractionResult | None = None
    usage: ModelUsage | None = None
