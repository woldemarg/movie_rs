"""HTTP API (FastAPI): the interface's only way into the recommender.

Request bodies are parsed and validated here, at the boundary; the core receives typed values.
Spec: docs/04-interface.md.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.adapters.embedder import load_embedder
from backend.adapters.llm import LLMClient, LLMError
from backend.adapters.tmdb import TmdbTrailers
from backend.core.agent import AgentError, AgentResult, Message, run_agent
from backend.core.catalog import Catalog, Embed
from backend.core.prompts import Gender, Viewer
from backend.core.tools import Toolbox
from backend.settings import ROOT, Provider, Settings

log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
PROVIDER_LABELS: dict[Provider, str] = {"local": "Локальна модель", "openrouter": "OpenRouter"}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ViewerIn(BaseModel):
    age: int | None = Field(default=None, ge=1, le=120)
    gender: Gender | None = None
    genre: str | None = None


class RecommendRequest(BaseModel):
    provider: Provider
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    viewer: ViewerIn = ViewerIn()

    @model_validator(mode="after")
    def ends_with_the_user(self) -> RecommendRequest:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must come from the user")
        return self


class MovieCard(BaseModel):
    id: int
    title: str
    year: int | None
    genres: list[str]
    rating: float
    overview: str
    poster_url: str | None
    why: str


class TrailerOut(BaseModel):
    movie_id: int
    title: str
    youtube_key: str
    reason: str


class StepOut(BaseModel):
    tool: str
    arguments: dict[str, object]
    outcome: str


class Trace(BaseModel):
    provider: Provider
    model: str
    llm_calls: int
    latency_ms: int
    steps: list[StepOut]


class RecommendResponse(BaseModel):
    reply: str
    picks: list[MovieCard]
    trailer: TrailerOut | None
    message: ChatMessage  # the assistant turn to send back with the next request
    trace: Trace


class ProviderOut(BaseModel):
    id: Provider
    label: str
    model: str
    configured: bool


class ConfigResponse(BaseModel):
    providers: list[ProviderOut]
    genres: list[str]
    movies: int
    trailers: bool


def create_app(settings: Settings | None = None, embed: Embed | None = None) -> FastAPI:
    """Build the app: load the catalogue and the embedding model once, then serve requests."""
    settings = settings or Settings()
    catalog = Catalog.load(DATA_DIR)
    tmdb_key = settings.tmdb_api_key.get_secret_value()
    trailers = TmdbTrailers(tmdb_key, settings.tmdb_base_url).find if tmdb_key else None
    toolbox = Toolbox(catalog, embed or load_embedder(catalog.embedding_model), trailers)
    app = FastAPI(title="MovieHarbor API")

    @app.get("/api/config")
    def config() -> ConfigResponse:
        providers = [
            ProviderOut(
                id=provider,
                label=label,
                model=settings.local_llm_model if provider == "local" else settings.llm_model,
                configured=settings.llm_profile(provider) is not None,
            )
            for provider, label in PROVIDER_LABELS.items()
        ]
        return ConfigResponse(
            providers=providers,
            genres=catalog.genres,
            movies=len(catalog.movies),
            trailers=trailers is not None,
        )

    @app.post("/api/recommend")
    def recommend(request: RecommendRequest) -> RecommendResponse:
        if request.viewer.genre and request.viewer.genre not in catalog.genres:
            raise HTTPException(422, f"Невідомий жанр: {request.viewer.genre}")
        profile = settings.llm_profile(request.provider)
        if profile is None:
            raise HTTPException(400, "OpenRouter не налаштовано: додайте LLM_API_KEY у файл .env.")
        chat = LLMClient(
            profile, settings.llm_temperature, settings.llm_max_tokens, settings.llm_timeout_s
        )
        conversation = [Message(m.role, m.content) for m in request.messages]
        started = time.perf_counter()
        try:
            result = run_agent(chat, toolbox, conversation, Viewer(**request.viewer.model_dump()))
        except (LLMError, AgentError) as exc:
            raise HTTPException(502, str(exc)) from None
        latency_ms = round((time.perf_counter() - started) * 1000)
        log.info(
            "%s/%s: %d LLM calls, %d ms, tools %s",
            profile.provider,
            profile.model,
            result.llm_calls,
            latency_ms,
            [step.tool for step in result.steps],
        )
        return to_response(
            result,
            Trace(
                provider=request.provider,
                model=profile.model,
                llm_calls=result.llm_calls,
                latency_ms=latency_ms,
                steps=[StepOut(**vars(step)) for step in result.steps],
            ),
        )

    return app


def to_response(result: AgentResult, trace: Trace) -> RecommendResponse:
    trailer = None
    if result.trailer:
        trailer = TrailerOut(
            movie_id=result.trailer.movie.id,
            title=result.trailer.movie.title,
            youtube_key=result.trailer.youtube_key,
            reason=result.trailer.reason,
        )
    return RecommendResponse(
        reply=result.reply,
        picks=[
            MovieCard(
                id=p.movie.id,
                title=p.movie.title,
                year=p.movie.year,
                genres=list(p.movie.genres),
                rating=p.movie.rating,
                overview=p.movie.overview,
                poster_url=p.movie.poster_url,
                why=p.why,
            )
            for p in result.picks
        ],
        trailer=trailer,
        message=ChatMessage(role="assistant", content=result.history_entry),
        trace=trace,
    )
