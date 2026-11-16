from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .documents import is_standalone_material, resolve_document
from .schemas import ConversationRequest, Intent, IntentRequest


class ConversationState(TypedDict, total=False):
    request: ConversationRequest
    intent: dict
    document: str | None
    result: dict


def extraction_message(result: dict) -> str:
    if result["status"] == "length_limit":
        return "这次输出达到长度上限，结果可能不完整。请缩短公告后重试。"
    if result["status"] != "ok":
        return "这次未能得到有效的抽取结果。请检查原文或重试。"
    if not result["events"]:
        return "本次未提取到企业收购或股份回购事件。"
    lines = []
    for index, event in enumerate(result["events"], 1):
        lines.append(f"{index}. {event['event_type']}")
        for name, values in event["arguments"].items():
            lines.append(f"{name}：{'、'.join(values) if values else '未提取到'}")
    if result["warnings"]:
        lines.append("部分字段需要与原文核对。")
    return "\n".join(lines)


def build_conversation_graph(client, intent_graph):
    async def identify(state):
        request = state["request"]
        effective = IntentRequest(messages=request.messages,
                                  document=request.document or request.context.document)
        located = resolve_document(effective)
        document = located.text if located else None
        if request.context.awaiting_document and is_standalone_material(request.messages[-1].content):
            intent = {"intent": Intent.EXTRACT, "source": "rule", "needs_document": False,
                      "clarification": None}
        else:
            classified = await intent_graph.ainvoke({"request": effective})
            intent = classified["result"]
        return {"intent": intent, "document": document}

    def branch(state):
        intent = state["intent"]
        if intent["intent"] == Intent.CHAT:
            return "chat"
        if intent["intent"] == Intent.EXTRACT and state["document"]:
            return "extract"
        return "clarify"

    def response(state, route, status, text, **extra):
        return {"result": {
            "message": {"role": "assistant", "content": text},
            "intent": state["intent"]["intent"], "source": state["intent"]["source"],
            "route": route, "status": status,
            "context": {"document": state["document"], "awaiting_document": status == "needs_document"},
            **extra,
        }}

    async def chat(state):
        request = state["request"]
        result = await client.chat([message.model_dump() for message in request.messages],
                                   state["document"], request.max_new_tokens)
        status = "length_limit" if result["usage"]["finish_reason"] == "length" else "ok"
        return response(state, "base_chat", status, result["content"], usage=result["usage"])

    async def extract(state):
        result = await client.extract(state["document"])
        return response(state, "lora_extract", result["status"], extraction_message(result),
                        extraction=result, usage=result["usage"])

    def clarify(state):
        intent = state["intent"]
        missing = intent["intent"] == Intent.EXTRACT and not state["document"]
        text = "请发送需要抽取的公告原文。" if missing else intent["clarification"]
        return response(state, "clarification", "needs_document" if missing else "clarification", text)

    graph = StateGraph(ConversationState)
    graph.add_node("identify", identify)
    graph.add_node("chat", chat)
    graph.add_node("extract", extract)
    graph.add_node("clarify", clarify)
    graph.add_edge(START, "identify")
    graph.add_conditional_edges("identify", branch)
    for name in ("chat", "extract", "clarify"):
        graph.add_edge(name, END)
    return graph.compile()
