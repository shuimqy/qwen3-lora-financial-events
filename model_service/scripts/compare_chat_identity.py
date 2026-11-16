"""Paired base/LoRA identity diagnostic; does not modify the running HTTP service."""
import hashlib
import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from peft.tuners.lora.layer import LoraLayer
from transformers import set_seed

from model_service.config import Settings
from model_service.engine import CHAT_SYSTEM_PROMPT, InferenceEngine

CASES = [
    {
        "name": "original_introduction",
        "expected": "助手介绍自己，不将小林称作自己的名字；不提及用户名字也可。",
        "messages": [
            {"role": "user", "content": "你好，我叫小林。请简单介绍一下你自己。"},
        ],
    },
    {
        "name": "clean_history_recall",
        "expected": "用户叫小林，回答应将小林对应到用户。",
        "messages": [
            {"role": "user", "content": "你好，我叫小林。请简单介绍一下你自己。"},
            {"role": "assistant", "content": "你好，小林！我是一个AI助手。"},
            {"role": "user", "content": "我刚才说我叫什么名字？"},
        ],
    },
    {
        "name": "contaminated_history_recall",
        "expected": "历史包含助手错误自称小林；用户仍然明确说过自己叫小林。",
        "messages": [
            {"role": "user", "content": "你好，我叫小林。请简单介绍一下你自己。"},
            {"role": "assistant", "content": "你好，我是小林，一个乐于助人的助手。有什么可以帮助你的吗？"},
            {"role": "user", "content": "我刚才说我叫什么名字？"},
        ],
    },
    {
        "name": "two_explicit_identities",
        "expected": "用户叫小林，助手叫小助手。",
        "messages": [
            {"role": "user", "content": "我叫小林，你叫小助手。请分别说出我的名字和你的名字。"},
        ],
    },
    {
        "name": "updated_user_name",
        "expected": "用户新称呼为小周，不能自称小周或仍称用户为小林。",
        "messages": [
            {"role": "user", "content": "我叫小林。"},
            {"role": "assistant", "content": "你好，小林。"},
            {"role": "user", "content": "我改用小周这个名字了。你现在应该怎么称呼我？"},
        ],
    },
]
STRATEGIES = [
    {"name": "greedy", "temperature": 0.0, "seed": 42},
    {"name": "sample_seed_7", "temperature": 0.7, "seed": 7},
    {"name": "sample_seed_42", "temperature": 0.7, "seed": 42},
    {"name": "sample_seed_2026", "temperature": 0.7, "seed": 2026},
]


def main():
    destination = Path(__file__).resolve().parents[1] / "run/chat_identity_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "results.json").exists():
        raise RuntimeError("Existing diagnostic retained; choose a new directory for a rerun")
    runtime = InferenceEngine(Settings())
    runtime.load()
    layers = [module for module in runtime.model.modules() if isinstance(module, LoraLayer)]
    assert len(layers) == 56 and all(not layer.disable_adapters for layer in layers)
    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "float32", "cpu_threads": runtime.settings.cpu_threads,
        "base_model_sha256": runtime.task["base_sha256"],
        "adapter_sha256": runtime.task["adapter_sha256"],
        "system_prompt": CHAT_SYSTEM_PROMPT, "enable_thinking": False,
        "max_new_tokens": 96, "top_p": 0.8, "sample_top_k": 20,
        "cases": CASES, "strategies": STRATEGIES, "lora_layers": len(layers),
        "method": "One verified PeftModel in a separate process. Base temporarily disables all56 LoRA layers; paired prompts/settings/seeds identical. Fixed histories including the contaminated case. No automatic semantic scoring.",
        "responses": [],
    }
    try:
        for case in CASES:
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *case["messages"]]
            prompt = runtime.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            for strategy in STRATEGIES:
                paired = []
                for mode in ("base", "lora"):
                    context = runtime.model.disable_adapter() if mode == "base" else nullcontext()
                    with context:
                        disabled = sum(layer.disable_adapters for layer in layers)
                        assert disabled == (len(layers) if mode == "base" else 0)
                        set_seed(strategy["seed"])
                        response = runtime.chat(case["messages"], max_new_tokens=96,
                                                temperature=strategy["temperature"], top_p=0.8)
                    assert all(not layer.disable_adapters for layer in layers)
                    record = {
                        "case": case["name"], "strategy": strategy["name"], "mode": mode,
                        "seed": strategy["seed"], "temperature": strategy["temperature"],
                        "prompt_token_sha256": digest, "input_tokens": len(prompt),
                        "disabled_lora_layers": disabled,
                        "answer": response["message"]["content"], "usage": response["usage"],
                    }
                    assert response["usage"]["input_tokens"] == len(prompt)
                    results["responses"].append(record)
                    paired.append(record)
                    (destination / "progress.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2)+"\n", encoding="utf8"
                    )
                    print(json.dumps({"case": case["name"], "strategy": strategy["name"],
                                      "mode": mode, "answer": record["answer"]},
                                     ensure_ascii=False), flush=True)
                assert paired[0]["prompt_token_sha256"] == paired[1]["prompt_token_sha256"]
        assert len(results["responses"]) == 40
        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["completed"] = True
        (destination / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)+"\n", encoding="utf8"
        )
        print(json.dumps({"completed": True, "generations": 40}), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
