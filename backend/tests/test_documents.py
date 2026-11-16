import pytest

from financial_backend.documents import material_from_message


@pytest.mark.parametrize("text", [
    "能看出这份公告里哪家企业掏钱买下了对方吗",
    "我想知道这篇收购公告中的买方名称和交易金额",
    "收购公告里公司为什么选择在这个时候发布消息？",
    "请提取这篇公告的收购方和金额。",
])
def test_requests_and_questions_are_not_treated_as_source_articles(text):
    assert material_from_message(text) is None


def test_marked_article_and_news_without_instructions_are_available():
    text = "甲公司公告，拟以1亿元收购乙公司100%股权。"
    assert material_from_message(text) == text
    assert material_from_message("公告原文：" + text) == text
