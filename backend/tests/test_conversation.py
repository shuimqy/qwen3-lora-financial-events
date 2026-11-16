import copy

import httpx
import pytest
from fastapi.testclient import TestClient

from financial_backend.app import create_app
from financial_backend.config import Settings
from financial_backend.model_client import ModelClient, ModelServiceError

ARTICLE = "甲公司公告，拟以1亿元收购乙公司100%股权。"
USAGE = {"input_tokens": 30, "output_tokens": 10, "finish_reason": "eos", "latency_ms": 5}
EXTRACTION = {"status": "ok", "valid_json_schema": True,
              "events": [{"event_type": "企业收购", "arguments": {
                  "收购方": ["甲公司"], "被收购方": ["乙公司"], "交易金额": ["1亿元"]}}],
              "raw_output": "original raw output", "validation_error": None,
              "warnings": [], "usage": USAGE}


class FakeModel:
    def __init__(self, label="CHAT", extraction=None, failure=None):
        self.calls = []
        self.label = label
        self.extraction = copy.deepcopy(extraction if extraction is not None else EXTRACTION)
        self.failure = failure

    async def close(self):
        pass

    async def classify(self, request):
        self.calls.append(("classify", request))
        return self.label, "eos"

    async def chat(self, messages, document, max_new_tokens):
        self.calls.append(("chat", messages, document, max_new_tokens))
        if self.failure:
            raise self.failure
        return {"content": "你好，我是金融助手。", "usage": USAGE}

    async def extract(self, document):
        self.calls.append(("extract", document))
        if self.failure:
            raise self.failure
        return self.extraction


def ask(client, text, **kwargs):
    return client.post("/v1/chat", json={"messages": [{"role": "user", "content": text}], **kwargs})


def test_chat_branch_preserves_history_and_uses_chat_only():
    model = FakeModel()
    messages = [{"role": "user", "content": "你好！"},
                {"role": "assistant", "content": "你好"},
                {"role": "user", "content": "什么是股份回购？"}]
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        response = client.post("/v1/chat", json={"messages": messages, "max_new_tokens": 128})
    assert response.status_code == 200
    result = response.json()
    assert result["route"] == "base_chat" and result["extraction"] is None
    assert model.calls == [("chat", messages, None, 128)]


def test_extraction_uses_material_only_and_returns_structured_and_readable_answers():
    model = FakeModel()
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        result = ask(client, "请提取这篇公告的字段", document=ARTICLE).json()
    assert model.calls == [("extract", ARTICLE)]
    assert result["route"] == "lora_extract"
    assert "甲公司" in result["message"]["content"]
    assert result["extraction"]["raw_output"] == "original raw output"
    assert result["context"]["document"] == ARTICLE


def test_missing_material_then_supplement_is_deterministic_without_classifier():
    model = FakeModel(label="CHAT")
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        first = ask(client, "请提取收购公告的字段").json()
        assert first["status"] == "needs_document"
        assert first["context"]["awaiting_document"]
        assert model.calls == []
        second = client.post("/v1/chat", json={"messages": [
            {"role": "user", "content": "请提取收购公告的字段"}, first["message"],
            {"role": "user", "content": ARTICLE}], "context": first["context"]}).json()
    assert second["route"] == "lora_extract"
    assert not second["context"]["awaiting_document"]
    assert model.calls == [("extract", ARTICLE)]


def test_article_then_short_extraction_request_retains_current_material():
    model = FakeModel()
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        first = ask(client, ARTICLE).json()
        assert first["route"] == "clarification" and model.calls == []
        second = ask(client, "抽取", context=first["context"]).json()
    assert second["route"] == "lora_extract"
    assert model.calls == [("extract", ARTICLE)]


def test_new_inline_material_replaces_old_and_new_chat_has_no_old_document():
    model = FakeModel()
    new = "丙公司公告，拟以2亿元收购丁公司100%股权。"
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        result = ask(client, "请提取这篇公告的字段，公告原文：" + new,
                     context={"document": ARTICLE, "awaiting_document": False}).json()
        assert result["context"]["document"] == new
        result = ask(client, "你好！").json()
    assert result["context"]["document"] is None
    assert model.calls[-1][0] == "chat" and model.calls[-1][2] is None


def test_changing_task_cancels_pending_extraction():
    model = FakeModel()
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        result = ask(client, "什么是股份回购？", context={"awaiting_document": True}).json()
    assert result["route"] == "base_chat" and not result["context"]["awaiting_document"]


@pytest.mark.parametrize("status,valid,events", [
    ("invalid_output", False, None), ("length_limit", False, None), ("ok", True, []),
])
def test_invalid_truncated_and_empty_outputs_are_distinct(status, valid, events):
    data = copy.deepcopy(EXTRACTION)
    data.update(status=status, valid_json_schema=valid, events=events)
    model = FakeModel(extraction=data)
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        response = ask(client, "请提取公告字段", document=ARTICLE)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == status
    assert result["extraction"]["events"] == events
    assert "甲公司" not in result["message"]["content"]


def test_classifier_path_is_followed_by_exactly_one_generation():
    model = FakeModel(label="EXTRACT")
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        result = ask(client, "这笔交易谁出的钱", document=ARTICLE).json()
    assert result["source"] == "llm"
    assert [call[0] for call in model.calls] == ["classify", "extract"]


@pytest.mark.parametrize("status,code", [(503, "model_busy"), (422, "input_too_long")])
def test_generation_errors_are_not_assistant_messages(status, code):
    model = FakeModel(failure=ModelServiceError(code, "try again", status))
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        result = ask(client, "你好！")
    assert result.status_code == status
    assert result.json()["error"]["code"] == code
    assert "message" not in result.json()


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "assistant", "content": "你好"}, {"role": "user", "content": "你好"}]},
    {"messages": [{"role": "user", "content": "你好"}, {"role": "user", "content": "你好"}]},
    {"messages": [{"role": "user", "content": "你好"}], "max_new_tokens": 513},
    {"messages": [{"role": "user", "content": "你好"}], "context": {"document": " "}},
])
def test_bad_history_or_context_is_rejected_before_model_calls(body):
    model = FakeModel()
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        assert client.post("/v1/chat", json=body).status_code == 422
    assert model.calls == []


def test_http_client_uses_base_for_chat_and_lora_endpoint_for_extraction():
    calls = []

    def handler(request):
        import json
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/v1/extract":
            return httpx.Response(200, json=EXTRACTION)
        return httpx.Response(200, json={"lora_enabled": False,
                              "message": {"content": "你好"}, "usage": USAGE})

    model = ModelClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        assert ask(client, "你好！").status_code == 200
        assert ask(client, "请提取公告字段", document=ARTICLE).status_code == 200
    assert calls[0][0] == "/v1/chat" and calls[0][1]["adapter_mode"] == "base"
    assert calls[1] == ("/v1/extract", {"text": ARTICLE})


def test_bad_upstream_extraction_protocol_returns_gateway_error():
    bad = {**EXTRACTION, "events": None}
    model = ModelClient(Settings(_env_file=None), transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=bad)))
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        assert ask(client, "请提取公告字段", document=ARTICLE).status_code == 502


@pytest.mark.parametrize("text", [
    "1加1等于？", "我问一加一等于几", "7乘8等于多少", "帮我推荐几本历史书", "明天北京天气如何",
])
@pytest.mark.parametrize("wrong_label", ["EXTRACT", "CLARIFY"])
def test_unrelated_new_topic_cannot_reuse_previous_extraction(text, wrong_label):
    model = FakeModel(label=wrong_label)
    history = [
        {"role": "user", "content": "请提取公告字段，公告原文：" + ARTICLE},
        {"role": "assistant", "content": "企业收购，甲公司收购乙公司，交易金额1亿元。"},
        {"role": "user", "content": "1加1等于？"},
        {"role": "assistant", "content": "企业收购，甲公司收购乙公司，交易金额1亿元。"},
        {"role": "user", "content": text},
    ]
    with TestClient(create_app(Settings(_env_file=None), client=model)) as client:
        response = client.post("/v1/chat", json={"messages": history,
            "context": {"document": ARTICLE, "awaiting_document": False}})
    assert response.status_code == 200
    result = response.json()
    assert result["route"] == "base_chat" and result["extraction"] is None
    assert result["context"]["document"] == ARTICLE
    assert [call[0] for call in model.calls] == ["chat"]
