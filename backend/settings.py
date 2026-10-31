"""Runtime settings from the environment and `.env` (template: `.env.sample`).

Spec: docs/03-llm-and-tool-calling.md (model sources), docs/04-interface.md (server).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.adapters.llm import LLMProfile

ROOT = Path(__file__).resolve().parents[1]
Provider = Literal["local", "openrouter"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Local model: any OpenAI-compatible server (LM Studio, Ollama, llama.cpp).
    local_llm_base_url: str = "http://127.0.0.1:1234/v1"
    local_llm_model: str = "google/gemma-3-12b"
    # External API: OpenRouter.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-4-26b-a4b-it"
    llm_api_key: SecretStr = SecretStr("")
    llm_provider_order: str = ""  # comma-separated OpenRouter providers to try first
    # Generation, shared by both sources.
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1500
    llm_timeout_s: float = 60.0
    # Trailers (optional).
    tmdb_api_key: SecretStr = SecretStr("")
    tmdb_base_url: str = "https://api.themoviedb.org/3"

    def llm_profile(self, provider: Provider) -> LLMProfile | None:
        """The connection for a model source, or None when it is not configured."""
        if provider == "local":
            return LLMProfile("local", self.local_llm_base_url, self.local_llm_model)
        key = self.llm_api_key.get_secret_value()
        if not key:
            return None
        order = [name.strip() for name in self.llm_provider_order.split(",") if name.strip()]
        extra_body = {"provider": {"order": order}} if order else {}
        return LLMProfile("openrouter", self.llm_base_url, self.llm_model, key, extra_body)
