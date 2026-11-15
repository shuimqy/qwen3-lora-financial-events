import pytest
from fastapi.testclient import TestClient

from model_service.app import create_app
from model_service.config import SERVICE_ROOT, Settings
from model_service.engine import BusyError, InferenceEngine, InputTooLongError


class FakeEngine:
    ready = False

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = []

    def load(self):
        self.ready = True

    def close(self):
        self.ready = False

    def info(self):
        return {"ready": self.ready, "device": "cpu"}

    def extract(self, text):
        self.received.append(text)
        if self.error:
            raise self.error
        return self.result or {
            "status": "ok", "valid_json_schema": True, "events": [],
            "raw_output": '{"events":[]}', "validation_error": None, "warnings": [],
            "usage": {"input_tokens": 300, "output_tokens": 5,
                      "finish_reason": "eos", "latency_ms": 10.0},
        }


def client(engine):
    return TestClient(create_app(Settings(_env_file=None), engine=engine))


def test_lifespan_health_and_request_preserve_input():
    engine = FakeEngine()
    with client(engine) as api:
        assert api.get("/health/live").status_code == 200
        assert api.get("/health/ready").json()["ready"] is True
        response = api.post("/v1/extract", json={"text": "  公告原文\n"})
        assert response.status_code == 200
        assert response.json()["events"] == []
        assert engine.received == ["  公告原文\n"]
    assert engine.ready is False


@pytest.mark.parametrize("body", [
    {"text": "   "}, {"text": "公告", "target": {"events": []}},
    {"text": "公告", "event_type": "企业收购"}, {"text": "文" * 12001}, {},
])
def test_request_rejects_blank_labels_and_excessive_character_length(body):
    engine = FakeEngine()
    with client(engine) as api:
        assert api.post("/v1/extract", json=body).status_code == 422
        assert engine.received == []


def test_token_limit_error_has_limit_and_does_not_silently_truncate():
    with client(FakeEngine(error=InputTooLongError(1800, 1792))) as api:
        response = api.post("/v1/extract", json={"text": "公告"})
        assert response.status_code == 422
        assert response.json()["error"]["input_tokens"] == 1800


def test_busy_generation_returns_retry_after():
    with client(FakeEngine(error=BusyError("busy"))) as api:
        response = api.post("/v1/extract", json={"text": "公告"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert response.json()["error"]["code"] == "model_busy"


def test_protocol_failure_is_distinct_from_a_negative_prediction():
    result = FakeEngine().extract("公告")
    result.update(status="invalid_output", valid_json_schema=False, events=None,
                  raw_output="invalid", validation_error="Invalid JSON")
    with client(FakeEngine(result=result)) as api:
        data = api.post("/v1/extract", json={"text": "公告"}).json()
        assert data["events"] is None and data["valid_json_schema"] is False
        assert data["raw_output"] == "invalid"


def test_not_ready_is_visible_to_the_business_backend():
    engine = FakeEngine()
    with client(engine) as api:
        engine.ready = False
        assert api.get("/health/ready").status_code == 503


def test_relative_model_paths_resolve_from_the_service_directory():
    settings = Settings(_env_file=None, base_model_path="../models/Qwen3-0.6B")
    assert settings.base_model_path == (SERVICE_ROOT.parent / "models/Qwen3-0.6B").resolve()


def test_engine_does_not_queue_while_another_generation_holds_the_lock():
    engine = InferenceEngine(Settings(_env_file=None))
    engine.ready = True
    engine._lock.acquire()
    try:
        with pytest.raises(BusyError):
            engine.extract("公告")
    finally:
        engine._lock.release()


def test_token_overflow_releases_generation_lock_before_any_generation():
    class OversizedTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 1793

    engine = InferenceEngine(Settings(_env_file=None))
    engine.ready = True
    engine.tokenizer = OversizedTokenizer()
    with pytest.raises(InputTooLongError) as error:
        engine.extract("公告")
    assert error.value.tokens == 1793
    assert engine._lock.acquire(blocking=False)
    engine._lock.release()
