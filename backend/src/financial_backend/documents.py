"""Locate user-provided material without treating an assistant reply as source text."""
import re
from dataclasses import dataclass

from .schemas import IntentRequest

MATERIAL_MARKER = re.compile(r"(?:公告(?:原文)?|原文|材料|文本|内容|文章|新闻|如下|以下)[：:]\s*(.+)$", re.S)
INSTRUCTION = re.compile(r"^(?:请|帮我|麻烦|提取|抽取|识别|把|从|能否|可以)")
QUESTION = re.compile(
    r"^(?:什么|为什么|如何|怎么|怎样|请问|请解释|请介绍|请讲解|我想知道|想知道|能否|能不能|可否)"
    r"|[吗么呢][。！!\s]*$"
)
NEWS_CUE = re.compile(r"公告|公司|股份|收购|回购|融资|董事会")
NEWS_ACTION = re.compile(r"公告|完成|拟以|拟收购|拟回购|决定|发布|宣布|召开|收购|回购|融资")


@dataclass(frozen=True)
class Document:
    text: str
    source: str


def is_standalone_material(text: str) -> bool:
    material = material_from_message(text)
    if not material:
        return False
    marker = MATERIAL_MARKER.search(text)
    return material == text.strip() or bool(marker and not text[:marker.start()].strip())


def material_from_message(text: str) -> str | None:
    marker = MATERIAL_MARKER.search(text)
    if marker and len(marker[1].strip()) >= 8:
        return marker[1].strip()
    if INSTRUCTION.search(text):
        pieces = re.split(r"[：:]", text, maxsplit=1)
        if len(pieces) == 2:
            candidate = pieces[1].strip()
            if len(candidate) >= 8 and NEWS_CUE.search(candidate):
                return candidate
        return None
    if not QUESTION.search(text) and not re.search(r"[?？]", text) and len(text.strip()) >= 18:
        if NEWS_CUE.search(text) and NEWS_ACTION.search(text):
            return text.strip()
    return None


def resolve_document(request: IntentRequest) -> Document | None:
    current = material_from_message(request.messages[-1].content)
    if current:
        return Document(current, "current_message")
    if request.document:
        return Document(request.document, "provided")
    for message in reversed(request.messages[:-1]):
        if message.role == "user":
            material = material_from_message(message.content)
            if material:
                return Document(material, "history")
    return None
