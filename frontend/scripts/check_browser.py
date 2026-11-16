"""Exercise the local UI in installed Edge; does not download a browser."""
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run"
ARTICLE = "甲公司公告，拟以1亿元收购乙公司100%股权。"


def main():
    RUN.mkdir(exist_ok=True)
    observations = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.goto("http://127.0.0.1:3000", wait_until="networkidle")
        expect(page.locator("#status-text")).to_have_text("可用", timeout=10000)
        page.screenshot(path=str(RUN / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(RUN / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1000})

        # Failed requests leave the draft intact and do not contaminate history.
        page.route("**/api/v1/chat", lambda route: route.fulfill(status=503, content_type="application/json",
                   body=json.dumps({"error": {"message": "服务暂时繁忙，请稍后重试。"}})))
        page.locator("#prompt").fill("失败后保留输入")
        page.locator("#send").click()
        expect(page.locator("#error")).to_be_visible()
        expect(page.locator("#prompt")).to_have_value("失败后保留输入")
        expect(page.locator(".message")).to_have_count(0)
        page.unroute("**/api/v1/chat")
        observations["error_keeps_draft"] = True

        # Real UI -> local proxy -> backend -> model service.
        page.locator("#prompt").fill("请提取这篇公告的事件字段。")
        page.locator("#send").click()
        expect(page.locator(".assistant .message-content")).to_have_text("请发送需要抽取的公告原文。", timeout=15000)
        observations["missing_document_clarification"] = True
        page.locator("#prompt").fill(ARTICLE)
        with page.expect_response("**/api/v1/chat", timeout=150000) as response:
            page.locator("#send").click()
        result = response.value.json()
        observations["extraction_response"] = result
        assert result["route"] == "lora_extract"
        assert result["status"] == "ok", result
        expect(page.locator(".event-card").first).to_be_visible(timeout=10000)
        page.screenshot(path=str(RUN / "extraction.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(RUN / "extraction-mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})

        page.locator("#new-chat").click()
        expect(page.locator("#welcome")).to_be_visible()
        expect(page.locator(".message")).to_have_count(0)
        page.locator("#prompt").fill("你好！")
        with page.expect_response("**/api/v1/chat", timeout=150000) as response:
            page.locator("#send").click()
        result = response.value.json()
        assert result["route"] == "base_chat"
        assert result["context"]["document"] is None
        observations["base_chat_response"] = result
        expect(page.locator(".assistant .message-content")).not_to_be_empty(timeout=10000)
        observations["new_chat_clears_context"] = True

        # An assistant reply containing markup must stay text, not execute HTML.
        page.locator("#new-chat").click()
        unsafe = '<img src=x onerror="window.__injected=true">'
        page.route("**/api/v1/chat", lambda route: route.fulfill(status=200, content_type="application/json",
                   body=json.dumps({"message": {"role": "assistant", "content": unsafe},
                                    "context": {"document": None, "awaiting_document": False},
                                    "status": "ok", "route": "base_chat", "extraction": None})))
        page.locator("#prompt").fill("测试纯文本")
        page.locator("#send").click()
        expect(page.locator(".assistant .message-content")).to_have_text(unsafe)
        assert page.locator(".message img").count() == 0
        assert not page.evaluate("Boolean(window.__injected)")
        observations["html_is_rendered_as_text"] = True
        assert failures == [], failures
        observations["browser_errors"] = failures
        browser.close()
    (RUN / "browser_check.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "checks": list(observations), "screenshots": str(RUN)}), flush=True)


if __name__ == "__main__":
    main()
