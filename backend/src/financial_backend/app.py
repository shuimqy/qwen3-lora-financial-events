import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings
from .conversation_graph import build_conversation_graph
from .intent_graph import build_intent_graph
from .model_client import ModelClient, ModelServiceError
from .schemas import ConversationRequest, ConversationResponse, IntentRequest, IntentResponse

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, client=None):
    settings = settings or Settings()
    model = client if client is not None else ModelClient(settings)
    graph = build_intent_graph(model)
    conversation_graph = build_conversation_graph(model, graph)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await model.close()

    app = FastAPI(title="金融助手业务后端", version=__version__, lifespan=lifespan,
                  description="统一聊天：基座聊天、LoRA 金融事件抽取与澄清回复。")
    app.state.intent_graph = graph
    app.state.conversation_graph = conversation_graph

    @app.exception_handler(ModelServiceError)
    async def model_error(request, error):
        return JSONResponse(status_code=error.status_code, headers={"Retry-After": "1"},
                            content={"error": {"code": error.code, "message": error.message}})

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok", "service": "financial-event-backend"}

    @app.get("/health/ready", tags=["health"])
    async def ready():
        info = await model.health()
        return JSONResponse(info, status_code=200 if info["ready"] else 503)

    @app.post("/v1/intent", response_model=IntentResponse, tags=["intent"])
    async def intent(request: IntentRequest):
        request_id = uuid4().hex
        result = await graph.ainvoke({"request": request})
        decision = result["result"]
        logger.info("Intent request_id=%s intent=%s source=%s", request_id,
                    decision["intent"], decision["source"])
        return {"request_id": request_id, **decision}

    @app.post("/v1/chat", response_model=ConversationResponse, tags=["chat"])
    async def chat(request: ConversationRequest):
        request_id = uuid4().hex
        state = await conversation_graph.ainvoke({"request": request})
        result = state["result"]
        logger.info("Chat request_id=%s route=%s status=%s", request_id, result["route"], result["status"])
        return {"request_id": request_id, **result}

    return app
