"""HTTP API (FastAPI): the interface's only way into the recommender.

Request bodies are parsed and validated here, at the boundary; the core receives typed values.
`/api/recommend/stream` reports the agent's steps while it works, which keeps a slow local model
responsive in the interface. When frontend/ is present (local development) the API also serves it;
in Docker the frontend is its own container. Spec: docs/04-interface.md.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from backend.adapters.embedder import load_embedder
from backend.adapters.llm import LLMClient, LLMError, LLMProfile, serves_model
from backend.adapters.tmdb import TmdbTrailers
from backend.core.agent import AgentError, AgentResult, Message, run_agent
from backend.core.catalog import Catalog, Embed
from backend.core.prompts import Gender, Viewer
from backend.core.tools import Step, Toolbox
from backend.settings import ROOT, Provider, Settings

log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
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
    data: dict[str, object]  # search: ranked candidates; recommend: accepted, rejected


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
    configured: bool  # the settings name this source (OpenRouter: LLM_API_KEY is set)
    ready: bool  # it can answer now (local: the server already lists the model)


class ConfigResponse(BaseModel):
    providers: list[ProviderOut]
    genres: list[str]
    movies: int
    trailers: bool


def is_ready(profile: LLMProfile | None) -> bool:
    """A configured source that can answer now; a local server must already list its model."""
    return profile is not None and (profile.provider != "local" or serves_model(profile))


@dataclass(frozen=True)
class Recommender:
    """One turn of the recommender behind the HTTP endpoints: checks, the agent run, the stream."""

    settings: Settings
    catalog: Catalog
    toolbox: Toolbox

    def config(self) -> ConfigResponse:
        providers = [
            ProviderOut(
                id=provider,
                label=label,
                model=self.settings.local_llm_model
                if provider == "local"
                else self.settings.llm_model,
                configured=self.settings.llm_profile(provider) is not None,
                ready=is_ready(self.settings.llm_profile(provider)),
            )
            for provider, label in PROVIDER_LABELS.items()
        ]
        return ConfigResponse(
            providers=providers,
            genres=self.catalog.genres,
            movies=len(self.catalog.movies),
            trailers=self.toolbox.find_trailer is not None,
        )

    def checked_profile(self, request: RecommendRequest) -> LLMProfile:
        """Reject what cannot be answered before any model call (and before a stream starts)."""
        if request.viewer.genre and request.viewer.genre not in self.catalog.genres:
            raise HTTPException(422, f"Невідомий жанр: {request.viewer.genre}")
        profile = self.settings.llm_profile(request.provider)
        if profile is None:
            raise HTTPException(400, "OpenRouter не налаштовано: додайте LLM_API_KEY у файл .env.")
        return profile

    def answer(
        self,
        request: RecommendRequest,
        profile: LLMProfile,
        on_step: Callable[[Step], None] | None = None,
    ) -> RecommendResponse:
        """Run the agent for one turn; raises LLMError or AgentError when the model fails."""
        settings = self.settings
        chat = LLMClient(
            profile, settings.llm_temperature, settings.llm_max_tokens, settings.llm_timeout_s
        )
        conversation = [Message(m.role, m.content) for m in request.messages]
        viewer = Viewer(**request.viewer.model_dump())
        started = time.perf_counter()
        result = run_agent(chat, self.toolbox, conversation, viewer, on_step)
        latency_ms = round((time.perf_counter() - started) * 1000)
        log.info(
            "%s/%s: %d LLM calls, %d ms, tools %s",
            profile.provider,
            profile.model,
            result.llm_calls,
            latency_ms,
            [step.tool for step in result.steps],
        )
        trace = Trace(
            provider=request.provider,
            model=profile.model,
            llm_calls=result.llm_calls,
            latency_ms=latency_ms,
            steps=[StepOut(**vars(step)) for step in result.steps],
        )
        return to_response(result, trace)

    def stream(self, request: RecommendRequest, profile: LLMProfile) -> Iterator[str]:
        """NDJSON lines: one {"type": "step"} per tool call, then {"type": "answer"} or "error"."""
        events: queue.Queue[dict | None] = queue.Queue()

        def report(step: Step) -> None:
            events.put({"type": "step", **StepOut(**vars(step)).model_dump()})

        def work() -> None:
            try:
                events.put({"type": "answer", **self.answer(request, profile, report).model_dump()})
            except (LLMError, AgentError) as exc:
                events.put({"type": "error", "detail": str(exc)})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"


def create_app(settings: Settings | None = None, embed: Embed | None = None) -> FastAPI:
    """Build the app: load the catalogue and the embedding model once, then serve requests."""
    settings = settings or Settings()
    catalog = Catalog.load(DATA_DIR)
    tmdb_key = settings.tmdb_api_key.get_secret_value()
    trailers = TmdbTrailers(tmdb_key, settings.tmdb_base_url).find if tmdb_key else None
    toolbox = Toolbox(catalog, embed or load_embedder(catalog.embedding_model), trailers)
    service = Recommender(settings, catalog, toolbox)
    app = FastAPI(title="MovieHarbor API")

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(map(str, e["loc"][1:])) or "body" for e in exc.errors()})
        detail = f"Некоректний запит: перевірте {', '.join(fields)}."
        return JSONResponse({"detail": detail}, status_code=422)

    @app.get("/api/config")
    def config() -> ConfigResponse:
        return service.config()

    @app.post("/api/recommend")
    def recommend(request: RecommendRequest) -> RecommendResponse:
        profile = service.checked_profile(request)
        try:
            return service.answer(request, profile)
        except (LLMError, AgentError) as exc:
            raise HTTPException(502, str(exc)) from None

    @app.post("/api/recommend/stream")
    def recommend_stream(request: RecommendRequest) -> StreamingResponse:
        stream = service.stream(request, service.checked_profile(request))
        return StreamingResponse(stream, media_type="application/x-ndjson")

    if FRONTEND_DIR.is_dir():  # local development; in Docker the frontend container serves it
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
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
