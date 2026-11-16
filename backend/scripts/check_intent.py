"""Small, visible development cases; not a held-out accuracy benchmark."""
import argparse
import json
import time
from pathlib import Path

import httpx

ARTICLE = "甲公司公告，拟以1亿元收购乙公司100%股权。"
CASES = [
    ("concept", "什么是股份回购？", "CHAT", None),
    ("negative", "不要抽取，只解释这条新闻。", "CHAT", ARTICLE),
    ("explicit", "请提取这篇公告的收购方和金额。", "EXTRACT", ARTICLE),
    ("missing_document", "请提取这篇公告的收购方和金额。", "EXTRACT", None),
    ("implicit_buy", "谁买了谁，花了多少钱？", "EXTRACT", ARTICLE),
    ("implicit_fields", "我想知道这笔交易的买方和价格", "EXTRACT", ARTICLE),
    ("vague", "帮我看看", "CLARIFY", None),
    ("general_chat", "帮我写一段生日祝福", "CHAT", None),
    ("financial_concept", "收购和融资在法律上有什么不同？", "CHAT", None),
    ("unsupported", "请提取发票字段", "CLARIFY", None),
    ("bare_article", ARTICLE, "CLARIFY", None),
    ("mixed", "解释这篇公告并提取收购字段", "EXTRACT", ARTICLE),
    ("free_chat", "今天有点累，陪我聊会儿天吧", "CHAT", None),
    ("ambiguous_rephrase", "能看出这份公告里哪家企业掏钱买下了对方吗", "EXTRACT", ARTICLE),
    ("extraction_format", "把刚才的交易信息整理成表格", "EXTRACT", ARTICLE),
    ("context_vague", "帮我看看", "CLARIFY", ARTICLE),
]
HISTORY_CASES = [
    ("followup", {"messages": [
        {"role": "user", "content": ARTICLE},
        {"role": "assistant", "content": "你希望怎样处理这段公告？"},
        {"role": "user", "content": "再提取一次刚才的字段"},
    ]}, "EXTRACT"),
    ("document_arrives", {"messages": [
        {"role": "user", "content": "请提取收购公告的字段"},
        {"role": "assistant", "content": "请发送需要抽取的公告原文。"},
        {"role": "user", "content": ARTICLE},
    ]}, "EXTRACT"),
    ("context_switch", {"messages": [
        {"role": "user", "content": ARTICLE},
        {"role": "assistant", "content": "你希望怎样处理这段公告？"},
        {"role": "user", "content": "不要抽取，只解释这条新闻。"},
    ]}, "CHAT"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "run" / "intent_check.json"
    output.parent.mkdir(exist_ok=True)
    results = []
    cases = []
    for name, text, expected, document in CASES:
        body = {"messages": [{"role": "user", "content": text}]}
        if document:
            body["document"] = document
        cases.append((name, body, expected))
    with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
        for name, body, expected in cases + HISTORY_CASES:
            started = time.perf_counter()
            response = client.post("/v1/intent", json=body)
            result = {"name": name, "request": body, "expected": expected,
                      "http_status": response.status_code, "response": response.json(),
                      "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
            result["matches_expected"] = response.status_code == 200 and result["response"].get("intent") == expected
            results.append(result)
            output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f'{name}: expected={expected} response={result["response"].get("intent")} '
                  f'source={result["response"].get("source")} ms={result["elapsed_ms"]}', flush=True)
    print(f"Development cases: {sum(item['matches_expected'] for item in results)}/{len(results)}; {output}")


if __name__ == "__main__":
    main()
