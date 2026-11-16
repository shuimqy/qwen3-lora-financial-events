from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_", env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8", extra="forbid",
    )
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    model_service_url: HttpUrl = "http://127.0.0.1:8001"
    model_timeout_seconds: float = Field(default=120, gt=0, le=300)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
