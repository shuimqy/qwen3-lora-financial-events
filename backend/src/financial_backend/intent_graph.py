from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .documents import is_standalone_material, resolve_document
from .rules import VAGUE_TASK, Decision, classify_rule, has_extraction_candidate
from .schemas import Intent, IntentRequest


class IntentState(TypedDict, total=False):
    request: IntentRequest
    decision: Decision | None
    source: str
    llm_used: bool
    raw_label: str | None
    result: dict


def build_intent_graph(client):
    def rules(state):
        request = state["request"]
        decision = classify_rule(request.messages[-1].content)
        if decision is None and len(request.messages) == 1 and is_standalone_material(request.messages[-1].content):
            decision = Decision(Intent.CLARIFY, "material_without_task")
        if decision is None and VAGUE_TASK.fullmatch(request.messages[-1].content.strip()):
            decision = Decision(Intent.CLARIFY, "unspecified_task")
        if decision is None and not has_extraction_candidate(request.messages[-1].content):
            decision = Decision(Intent.CHAT, "general_conversation")
        return {"decision": decision, "source": "rule", "llm_used": False, "raw_label": None}

    async def classify(state):
        raw, finish = await client.classify(state["request"])
        label = raw.strip()
        if finish != "eos" or label not in {item.value for item in Intent}:
            return {"decision": Decision(Intent.CLARIFY, "classifier_invalid_output"),
                    "source": "fallback", "llm_used": True, "raw_label": raw[:200]}
        return {"decision": Decision(Intent(label), "contextual_model_classification"),
                "source": "llm", "llm_used": True, "raw_label": label}

    def resolve(state):
        decision = state["decision"]
        document = resolve_document(state["request"]) if decision.intent == Intent.EXTRACT else None
        needs_document = decision.intent == Intent.EXTRACT and document is None
        clarification = None
        if needs_document:
            clarification = "请发送需要抽取的公告原文。"
        elif decision.intent == Intent.CLARIFY:
            clarification = ("目前支持企业收购和股份回购的事件抽取。你需要处理这两类公告吗？"
                             if decision.reason == "unsupported_extraction" else
                             "你希望我回答问题、解释内容，还是从金融公告中抽取事件和字段？")
        return {"result": {
            "intent": decision.intent, "source": state["source"], "reason": decision.reason,
            "llm_used": state["llm_used"], "raw_label": state["raw_label"],
            "needs_document": needs_document,
            "document_source": document.source if document else "none",
            "clarification": clarification,
        }}

    graph = StateGraph(IntentState)
    graph.add_node("rules", rules)
    graph.add_node("classify", classify)
    graph.add_node("resolve", resolve)
    graph.add_edge(START, "rules")
    graph.add_conditional_edges("rules", lambda state: "resolve" if state["decision"] else "classify")
    graph.add_edge("classify", "resolve")
    graph.add_edge("resolve", END)
    return graph.compile()
