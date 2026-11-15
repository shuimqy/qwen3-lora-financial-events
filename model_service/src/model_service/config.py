from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SERVICE_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEL_SERVICE_",
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1, le=65535)
    cpu_threads: int = Field(default=6, ge=1, le=64)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    base_model_path: Path = WORKSPACE_ROOT / "models/Qwen3-0.6B"
    adapter_path: Path = WORKSPACE_ROOT / "models/Qwen3-0.6B-financial-LoRA"

    @field_validator("base_model_path", "adapter_path")
    @classmethod
    def resolve_model_path(cls, value: Path) -> Path:
        return (value if value.is_absolute() else SERVICE_ROOT / value).resolve()
