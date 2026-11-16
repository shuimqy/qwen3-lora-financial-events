import asyncio
import json

import httpx
import pytest

from financial_backend.config import Settings
from financial_backend.model_client import ModelClient, ModelServiceError, classifier_messages
from financial_backend.schemas import IntentRequest


def request():
    return IntentRequest(messages=[{"role": "user", "content": "谁买了谁，花了多少钱？"}])


def run_classification(handler):
    async def run():
        model = ModelClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
        try:
            return await model.classify(request())
        finally:
            await model.close()
    return asyncio.run(run())


def test_classifier_calls_base_greedy_with_small_generation_budget():
    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/v1/chat"
        assert body["adapter_mode"] == "base"
        assert body["temperature"] == 0 and body["max_new_tokens"] == 8
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"lora_enabled": False,
                              "message": {"content": "EXTRACT"}, "usage": {"finish_reason": "eos"}})
    assert run_classification(handler) == ("EXTRACT", "eos")


@pytest.mark.parametrize("status,body,code,expected_status", [
    (503, {}, "model_busy_or_not_ready", 503),
    (500, {}, "model_unavailable", 503),
    (200, [], "invalid_model_response", 502),
    (200, {"lora_enabled": True}, "base_mode_required", 502),
    (200, {"lora_enabled": False}, "invalid_model_response", 502),
    (200, {"lora_enabled": False, "message": {"content": None}}, "invalid_model_response", 502),
])
def test_upstream_status_and_response_are_checked(status, body, code, expected_status):
    with pytest.raises(ModelServiceError) as error:
        run_classification(lambda req: httpx.Response(status, json=body))
    assert error.value.code == code and error.value.status_code == expected_status


def test_timeout_does_not_become_a_chat_decision():
    def timeout(req):
        raise httpx.ReadTimeout("timeout", request=req)
    with pytest.raises(ModelServiceError, match="超时"):
        run_classification(timeout)


def test_context_is_bounded_and_keeps_current_instruction():
    body = IntentRequest(messages=[{"role": "user", "content": "甲" * 11000},
                                   {"role": "assistant", "content": "乙" * 11000},
                                   {"role": "user", "content": "请提取刚才公告的字段"}])
    prompt = classifier_messages(body)
    assert sum(len(item["content"]) for item in prompt) < 2500
    assert "请提取刚才公告的字段" in prompt[-1]["content"]
