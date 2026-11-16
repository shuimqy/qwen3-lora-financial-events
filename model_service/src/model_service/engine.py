import hashlib
import json
import logging
import threading
import time
from contextlib import nullcontext
from importlib.resources import files

from .config import Settings
from .protocol import inspect_answer

logger = logging.getLogger(__name__)
CHAT_SYSTEM_PROMPT = "你是一个乐于助人的助手。请根据对话上下文回答用户的问题。"


class BusyError(Exception):
    pass


class NotReadyError(Exception):
    pass


class InputTooLongError(Exception):
    def __init__(self, tokens: int, limit: int):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"Input has {tokens} tokens; limit is {limit}, including the prompt")


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class InferenceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.task = json.loads(files("model_service").joinpath("task.json").read_text("utf-8"))
        self.ready = False
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.generation = None

    def load(self):
        import torch
        from peft import PeftConfig, PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

        for path, expected in (
            (self.settings.base_model_path / "model.safetensors", self.task["base_sha256"]),
            (self.settings.adapter_path / "adapter_model.safetensors", self.task["adapter_sha256"]),
        ):
            if not path.is_file() or file_sha256(path) != expected:
                raise RuntimeError(f"Model file missing or SHA-256 mismatch: {path}")
        torch.set_num_threads(self.settings.cpu_threads)
        set_seed(42)
        logger.info("Loading local base and checkpoint-%s LoRA on CPU", self.task["checkpoint"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.base_model_path, local_files_only=True, padding_side="left"
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.settings.base_model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        adapter_config = PeftConfig.from_pretrained(
            self.settings.adapter_path, local_files_only=True
        )
        # Override historical cloud metadata in memory; keep the original files untouched.
        adapter_config.base_model_name_or_path = str(self.settings.base_model_path)
        self.model = PeftModel.from_pretrained(
            base, self.settings.adapter_path, config=adapter_config,
            is_trainable=False, local_files_only=True
        )
        disk = load_file(str(self.settings.adapter_path / "adapter_model.safetensors"))
        actual = get_peft_model_state_dict(self.model)
        if set(disk) != set(actual) or not all(
            torch.equal(value.float(), actual[key].detach().cpu().float())
            for key, value in disk.items()
        ):
            raise RuntimeError("Loaded adapter tensors differ from the saved weights")
        del disk, actual
        self.model.eval()
        self.generation = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.task["max_new_tokens"],
            repetition_penalty=1.0,
            eos_token_id=self.task["eos_token_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True,
        )
        self.ready = True
        logger.info("Model ready: CPU float32, %s threads", self.settings.cpu_threads)

    def close(self):
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.generation = None

    def info(self) -> dict:
        return {
            "ready": self.ready,
            "busy": self._lock.locked(),
            "model": self.task["model_name"],
            "checkpoint": self.task["checkpoint"],
            "adapter_sha256": self.task["adapter_sha256"],
            "device": "cpu",
            "dtype": "float32",
            "cpu_threads": self.settings.cpu_threads,
            "max_input_tokens": self.task["max_sequence_tokens"] - self.task["max_new_tokens"],
            "max_new_tokens": self.task["max_new_tokens"],
            "enable_thinking": False,
            "decoding": "greedy",
            "max_concurrent_generations": 1,
            "chat": {
                "endpoint": "/v1/chat",
                "lora_enabled": True,
                "default_adapter_mode": "finetuned",
                "adapter_modes": ["finetuned", "base"],
                "default_max_new_tokens": 256,
                "max_new_tokens_limit": 512,
                "default_temperature": 0.7,
                "default_top_p": 0.8,
                "top_k": 20,
                "history": "client_supplied_messages",
            },
        }

    def extract(self, article: str) -> dict:
        messages = [
            {"role": "system", "content": self.task["system_prompt"]},
            {"role": "user", "content": self.task["user_prefix"] + article},
        ]
        result = self._generate(messages, self.generation)
        parsed = inspect_answer(result["answer"], article)
        status = "ok" if parsed.error is None else "invalid_output"
        if result["usage"]["finish_reason"] == "length":
            status = "length_limit"
        return {
            "status": status,
            "valid_json_schema": parsed.error is None,
            "events": parsed.events,
            "raw_output": result["answer"],
            "validation_error": parsed.error,
            "warnings": parsed.warnings,
            "usage": result["usage"],
        }

    def chat(self, messages: list[dict], max_new_tokens=256, temperature=0.7, top_p=0.8,
             adapter_mode="finetuned"):
        from transformers import GenerationConfig

        conversation = list(messages)
        if conversation[0]["role"] != "system":
            conversation.insert(0, {"role": "system", "content": CHAT_SYSTEM_PROMPT})
        options = {
            "do_sample": temperature > 0,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
            "eos_token_id": self.task["eos_token_ids"],
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 151643,
            "bos_token_id": self.tokenizer.bos_token_id if self.tokenizer else None,
            "use_cache": True,
        }
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p, top_k=20)
        result = self._generate(conversation, GenerationConfig(**options), adapter_mode)
        return {
            "model": self.task["model_name"] if adapter_mode == "finetuned" else "Qwen3-0.6B",
            "lora_enabled": adapter_mode == "finetuned",
            "message": {"role": "assistant", "content": result["answer"]},
            "usage": result["usage"],
        }

    def _generate(self, messages: list[dict], generation, adapter_mode="finetuned") -> dict:
        if adapter_mode not in {"finetuned", "base"}:
            raise ValueError("Unsupported adapter mode")
        if not self.ready:
            raise NotReadyError("Model is not ready")
        if not self._lock.acquire(blocking=False):
            raise BusyError("Model is processing another request")
        started = time.perf_counter()
        try:
            import torch

            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            input_limit = self.task["max_sequence_tokens"] - generation.max_new_tokens
            if len(tokens) > input_limit:
                raise InputTooLongError(len(tokens), input_limit)
            batch = self.tokenizer.pad(
                {"input_ids": [tokens]},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            adapter_context = self.model.disable_adapter() if adapter_mode == "base" else nullcontext()
            # Switching adapters and generation are both inside the same lock.
            with adapter_context, torch.inference_mode():
                generated = self.model.generate(
                    **batch, generation_config=generation, use_model_defaults=False
                )
            output = generated[0, batch["input_ids"].shape[1]:].tolist()
            end = next(
                (index for index, token in enumerate(output) if token in self.task["eos_token_ids"]),
                None,
            )
            if end is not None:
                output = output[:end + 1]
            answer = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            return {
                "answer": answer,
                "usage": {
                    "input_tokens": len(tokens),
                    "output_tokens": len(output),
                    "finish_reason": "eos" if end is not None else "length",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            }
        finally:
            self._lock.release()
