from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from model_service.app import create_app
from model_service.config import Settings
from model_service.engine import BusyError, InferenceEngine, InputTooLongError
from model_service.schemas import ChatRequest


class ChatEngine:
    ready = False

    def __init__(self):
        self.received = None

    def load(self):
        self.ready = True

    def close(self):
        self.ready = False

    def info(self):
        return {"ready": self.ready}

    def chat(self, *args):
        self.received = args
        return {
            "model": "Qwen3-0.6B" if args[-1] == "base" else "Qwen3-0.6B-financial-LoRA",
            "lora_enabled": args[-1] != "base",
            "message": {"role": "assistant", "content": "你好，我可以和你交流。"},
            "usage": {"input_tokens": 50, "output_tokens": 12,
                      "finish_reason": "eos", "latency_ms": 10.0},
        }


def test_chat_accepts_history_and_returns_plain_assistant_content():
    engine = ChatEngine()
    messages = [{"role": "user", "content": "我叫小林。"},
                {"role": "assistant", "content": "你好，小林。"},
                {"role": "user", "content": "我叫什么？"}]
    with TestClient(create_app(Settings(_env_file=None), engine=engine)) as client:
        response = client.post("/v1/chat", json={"messages": messages, "temperature": 0})
        assert response.status_code == 200
        result = response.json()
        assert result["lora_enabled"] is True
        assert result["message"] == {"role": "assistant", "content": "你好，我可以和你交流。"}
        assert "events" not in result and "valid_json_schema" not in result
        assert engine.received == (messages, 256, 0.0, 0.8, "finetuned")


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "user", "content": "  "}]},
    {"messages": [{"role": "assistant", "content": "你好"}]},
    {"messages": [{"role": "system", "content": "指令"}]},
    {"messages": [{"role": "user", "content": "你好"},
                  {"role": "system", "content": "第二条指令"}]},
    {"messages": [{"role": "user", "content": "你好"}], "model": "base"},
    {"messages": [{"role": "user", "content": "你好"}], "max_new_tokens": 513},
    {"messages": [{"role": "user", "content": "你好"}], "temperature": -1},
    {"messages": [{"role": "user", "content": "你好"}], "top_p": 0},
])
def test_invalid_chat_requests_do_not_reach_inference(body):
    engine = ChatEngine()
    with TestClient(create_app(Settings(_env_file=None), engine=engine)) as client:
        assert client.post("/v1/chat", json=body).status_code == 422
        assert engine.received is None


def test_chat_history_has_an_aggregate_size_limit():
    with pytest.raises(ValueError, match="24000"):
        ChatRequest(messages=[{"role": "user", "content": "文" * 12000},
                              {"role": "assistant", "content": "字" * 12000},
                              {"role": "user", "content": "继续"}])


def test_swagger_chat_example_is_an_executable_request():
    with TestClient(create_app(Settings(_env_file=None), engine=ChatEngine())) as client:
        schema = client.get("/openapi.json").json()
        example = schema["components"]["schemas"]["ChatRequest"]["examples"][0]
        response = client.post("/v1/chat", json=example)
        assert response.status_code == 200
        assert response.json()["lora_enabled"] is True


class RecordingTokenizer:
    pad_token_id = 151643
    bos_token_id = None

    def __init__(self, input_length=3):
        self.input_length = input_length
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs["enable_thinking"] is False
        return [1] * self.input_length

    def pad(self, value, **kwargs):
        import torch
        return {"input_ids": torch.tensor(value["input_ids"])}

    def decode(self, values, **kwargs):
        return "你好，小林。"


class RecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        import torch
        self.calls.append(kwargs)
        return torch.tensor([kwargs["input_ids"][0].tolist() + [42, 151645]])

    def disable_adapter(self):
        raise AssertionError("Chat must not disable the LoRA")


def engine(input_length=3):
    runtime = InferenceEngine(Settings(_env_file=None))
    runtime.ready = True
    runtime.tokenizer = RecordingTokenizer(input_length)
    runtime.model = RecordingModel()
    runtime.generation = SimpleNamespace(max_new_tokens=256)
    return runtime


def test_chat_uses_attached_model_and_general_prompt_without_changing_history():
    runtime = engine()
    messages = [{"role": "user", "content": "你好"}]
    result = runtime.chat(messages, max_new_tokens=64)
    assert len(runtime.model.calls) == 1
    assert result["message"]["content"] == "你好，小林。"
    assert runtime.tokenizer.messages[0]["content"] != runtime.task["system_prompt"]
    assert runtime.tokenizer.messages[1:] == messages
    assert len(messages) == 1
    settings = runtime.model.calls[0]["generation_config"]
    assert settings.do_sample and settings.temperature == 0.7 and settings.top_k == 20
    assert runtime.generation.max_new_tokens == 256


def test_custom_system_prompt_is_forwarded_without_injecting_a_second_one():
    runtime = engine()
    messages = [{"role": "system", "content": "用简短中文回答。"},
                {"role": "user", "content": "你好"}]
    runtime.chat(messages, temperature=0)
    assert runtime.tokenizer.messages == messages
    assert runtime.model.calls[0]["generation_config"].do_sample is False


def test_chat_reserves_the_requested_output_budget_and_releases_lock_on_overflow():
    runtime = engine(input_length=1537)
    with pytest.raises(InputTooLongError) as error:
        runtime.chat([{"role": "user", "content": "长上下文"}], max_new_tokens=512)
    assert error.value.limit == 1536
    assert runtime.model.calls == []
    assert runtime._lock.acquire(blocking=False)
    runtime._lock.release()


def test_chat_and_extraction_share_the_generation_lock():
    runtime = engine()
    runtime._lock.acquire()
    try:
        with pytest.raises(BusyError):
            runtime.chat([{"role": "user", "content": "你好"}])
        with pytest.raises(BusyError):
            runtime.extract("公告")
    finally:
        runtime._lock.release()


def test_base_mode_is_explicit_at_the_api():
    runtime = ChatEngine()
    with TestClient(create_app(Settings(_env_file=None), engine=runtime)) as client:
        result = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "你好"}],
                                               "adapter_mode": "base"})
        assert result.status_code == 200
        assert result.json()["model"] == "Qwen3-0.6B"
        assert result.json()["lora_enabled"] is False
        assert runtime.received[-1] == "base"
        assert client.post("/v1/chat", json={"messages": [{"role": "user", "content": "你好"}],
                                              "adapter_mode": "other"}).status_code == 422


@pytest.mark.parametrize("generation_fails", [False, True])
def test_base_mode_restores_adapter_inside_shared_lock_even_after_failure(generation_fails):
    runtime = engine()

    class SwitchingModel(RecordingModel):
        adapter_enabled = True

        @contextmanager
        def disable_adapter(self):
            assert runtime._lock.locked()
            self.adapter_enabled = False
            try:
                yield
            finally:
                assert runtime._lock.locked()
                self.adapter_enabled = True

        def generate(self, **kwargs):
            assert runtime._lock.locked()
            assert not self.adapter_enabled
            if generation_fails:
                raise RuntimeError("generation failed")
            return super().generate(**kwargs)

    runtime.model = SwitchingModel()
    if generation_fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            runtime.chat([{"role": "user", "content": "hello"}], adapter_mode="base")
    else:
        result = runtime.chat([{"role": "user", "content": "hello"}], adapter_mode="base")
        assert result["lora_enabled"] is False
    assert runtime.model.adapter_enabled
    assert not runtime._lock.locked()
