"""Conservative command rules; finance keywords alone never trigger extraction."""
import re
from dataclasses import dataclass

from .schemas import Intent


@dataclass(frozen=True)
class Decision:
    intent: Intent
    reason: str


GREETING = re.compile(r"^(?:你好|您好|嗨|谢谢(?:你)?|再见|早上好|晚上好|hello|hi)[！!。,.，?？\s]*$", re.I)
NEGATED_EXTRACTION = re.compile(r"(?:不要|不用|不需要|无需|别)(?:再)?(?:提取|抽取|识别)")
EXPLAIN = re.compile(r"(?:解释|讲解|介绍|说明|聊天|聊聊|讨论|是什么意思|是什么|有什么区别)")
CONCEPT = re.compile(
    r"^(?:(?:请问|请|帮我|能否|可以|麻烦你)\s*)?"
    r"(?:什么是|解释|讲解|介绍|说明|如何(?:实现|进行|做|提取|抽取)|怎么(?:实现|进行|做|提取|抽取)|"
    r"怎样(?:实现|进行|做|提取|抽取))"
)
DEFINITION_END = re.compile(r"(?:是什么|是什么意思|有什么区别|有什么不同|有何区别|的区别|的原理)[？?。\s]*$")
TARGET = re.compile(r"收购方|被收购方|回购方|交易金额|回购.*数量|字段|事件|公告|原文|材料|上[一条文]|刚才|这段|这篇|下面|以下")
DIRECTIVE = re.compile(
    r"^(?:(?:请你?|帮我|帮忙|麻烦你?|劳驾)\s*)?"
    r"(?:(?:从|把|对|将).{0,60}?)?(?:提取|抽取|识别)"
)
FOLLOWUP = re.compile(r"^(?:再|继续|重新|再来一次|再做一次).{0,12}(?:提取|抽取)")
OUT_OF_SCOPE = re.compile(r"发票|税号|身份证|简历|病历|合同条款")
ORDINARY_TASK = re.compile(r"^(?:(?:请你?|帮我|麻烦你?)\s*)?(?:写(?:一|个|段|封|首|篇)|翻译|计算|讲个笑话)")
CONVERSATION = re.compile(r"陪我(?:聊|说说话)|(?:一起|和你)(?:聊聊|聊天)")
VAGUE_TASK = re.compile(r"^(?:(?:请|帮我|麻烦你)\s*)?(?:看看|看一下|处理一下|处理下)[。.!！?？\s]*$")
SHORT_EXTRACTION = re.compile(r"^(?:(?:请|帮我)\s*)?(?:提取|抽取)(?:一下|事件|字段)?[。！!\s]*$")
SHORT_CHAT = re.compile(r"^(?:聊天|聊聊|正常聊天|普通聊天|解释(?:一下)?|回答(?:问题)?|问答)[。！!\s]*$")
IDENTITY = re.compile(
    r"(?:^|[，,。！？!?])(?:我叫[^，。！？\n]{1,12}|(?:请)?(?:简单)?介绍(?:一下)?你自己|"
    r"你叫什么(?:名字)?|你是谁|我叫什么(?:名字)?)[。？！!\s]*$"
)
EXTRACTION_CANDIDATE = re.compile(
    r"提取|抽取|识别|整理|梳理|汇总|列出|列一下|列出来|表格|"
    r"收购|回购|交易|买方|卖方|出资方|字段|事件|公告|原文|材料|发票|税号|"
    r"刚才|之前|前面|上[一条段篇次个]|这[一条段篇笔份桩次个]|上述|继续|再来|重来|"
    r"谁.{0,12}(?:买|卖|支付|出资|掏钱)|(?:买|卖).{0,12}(?:谁|哪家)|"
    r"^(?:金额|数量|价格)(?:呢|是多少|多少)?[？?。\s]*$"
)


def command_text(text: str) -> str:
    return re.split(r"(?:公告(?:原文)?|原文|材料|文本|内容|文章|新闻|如下|以下)[：:]",
                    text.strip(), maxsplit=1)[0]


def has_extraction_candidate(text: str) -> bool:
    """History can resolve a reference, but cannot create a new extraction request."""
    return bool(EXTRACTION_CANDIDATE.search(command_text(text)))


def classify_rule(text: str) -> Decision | None:
    # Only inspect the command; words quoted inside an article are not user instructions.
    text = command_text(text)
    if GREETING.fullmatch(text):
        return Decision(Intent.CHAT, "greeting")
    if SHORT_EXTRACTION.fullmatch(text):
        return Decision(Intent.EXTRACT, "explicit_extraction_request")
    if SHORT_CHAT.fullmatch(text):
        return Decision(Intent.CHAT, "explicit_chat_request")
    if NEGATED_EXTRACTION.search(text):
        # A partial negative (e.g. 不要提取金额，只提取公司名) needs contextual judgement.
        remainder = NEGATED_EXTRACTION.sub("", text)
        if EXPLAIN.search(text) and not re.search(r"提取|抽取|识别", remainder):
            return Decision(Intent.CHAT, "explicit_chat_without_extraction")
        return None
    if CONCEPT.search(text) or DEFINITION_END.search(text):
        # Mixed tasks such as 解释这篇公告并提取字段 are left to the classifier.
        if not re.search(r"(?:并|同时|再|然后).{0,8}(?:提取|抽取|识别)", text):
            return Decision(Intent.CHAT, "concept_or_explanation")
    if DIRECTIVE.search(text) or FOLLOWUP.search(text):
        if OUT_OF_SCOPE.search(text):
            return Decision(Intent.CLARIFY, "unsupported_extraction")
        if TARGET.search(text):
            return Decision(Intent.EXTRACT, "explicit_extraction_request")
    if (ORDINARY_TASK.search(text) or CONVERSATION.search(text)) and not re.search(r"提取|抽取|识别", text):
        return Decision(Intent.CHAT, "ordinary_task")
    if IDENTITY.search(text) and not re.search(r"提取|抽取|识别", text):
        return Decision(Intent.CHAT, "identity_conversation")
    return None
