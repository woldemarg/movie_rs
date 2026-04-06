"""Centralized application settings for MovieHarbor."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-driven settings with sane defaults for local-first usage."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = "MovieHarbor"
    data_path: str = "movies.pkl"
    max_context_rows: int = 15

    llm_provider: Literal["lmstudio", "google"] = "lmstudio"
    use_gemma: bool = False

    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_hermes_model: str = "hermes-3-llama-3.1-8b"
    lmstudio_gemma_model: str = "google/gemma-2-9b-it"

    google_primary_model: str = "gemini-2.5-flash"
    google_model_fallbacks: tuple[str, ...] = Field(
        default=("gemini-2.5-flash-lite", "gemini-2.0-flash")
    )
    google_embedding_model: str = "text-embedding-004"

    # Semantic retrieval via embeddings is Google-based in this MVP.
    enable_semantic_search: bool = True

    @property
    def active_model(self) -> str:
        if self.llm_provider == "lmstudio":
            return self.lmstudio_gemma_model if self.use_gemma else self.lmstudio_hermes_model
        return self.google_primary_model

    @property
    def provider_label(self) -> str:
        if self.llm_provider == "lmstudio":
            return "LM Studio (local)"
        return "Google AI Studio"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached settings instance."""
    return AppSettings()
