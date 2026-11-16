import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings
from .engine import BusyError, InferenceEngine, InputTooLongError, NotReadyError
from .schemas import ChatRequest, ChatResponse, ExtractRequest, ExtractResponse

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    settings = settings or Settings()
    runtime = engine if engine is not None else InferenceEngine(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(runtime.load)
            yield
        finally:
            await run_in_threadpool(runtime.close)

    app = FastAPI(
        title="金融事件模型推理服务",
        version=__version__,
        description="独立模型服务：CPU加载本地Qwen3基座与单LoRA，不包含业务后端。",
        lifespan=lifespan,
    )
    app.state.engine = runtime

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok", "service": "financial-event-model-service", "version": __version__}

    @app.get("/health/ready", tags=["health"])
    async def ready():
        return JSONResponse(runtime.info(), status_code=200 if runtime.ready else 503)

    @app.get("/v1/model", tags=["model"])
    async def model_info():
        return runtime.info()

    async def infer(operation, *args):
        request_id = uuid4().hex
        try:
            result = await run_in_threadpool(operation, *args)
        except InputTooLongError as error:
            return JSONResponse(status_code=422, content={"error": {
                "code": "input_too_long", "message": str(error),
                "input_tokens": error.tokens, "max_input_tokens": error.limit,
                "request_id": request_id,
            }})
        except BusyError as error:
            return JSONResponse(status_code=503, headers={"Retry-After": "1"}, content={"error": {
                "code": "model_busy", "message": str(error), "request_id": request_id,
            }})
        except NotReadyError as error:
            return JSONResponse(status_code=503, content={"error": {
                "code": "model_not_ready", "message": str(error), "request_id": request_id,
            }})
        except Exception:
            logger.exception("Inference failed request_id=%s", request_id)
            return JSONResponse(status_code=500, content={"error": {
                "code": "inference_failed", "message": "模型推理失败，请查看服务日志。",
                "request_id": request_id,
            }})
        logger.info("Inference request_id=%s operation=%s status=%s latency_ms=%s", request_id,
                    operation.__name__, result.get("status", "ok"), result["usage"]["latency_ms"])
        return {"request_id": request_id, **result}

    @app.post("/v1/extract", response_model=ExtractResponse, tags=["inference"])
    async def extract(request: ExtractRequest):
        return await infer(runtime.extract, request.text)

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest):
        return await infer(
            runtime.chat,
            [message.model_dump() for message in request.messages],
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            request.adapter_mode,
        )

    return app
