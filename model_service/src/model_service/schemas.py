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
