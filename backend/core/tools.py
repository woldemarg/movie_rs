"""Tool calling: the functions the LLM may call, their JSON schemas and their handlers.

The agent (core/agent.py) offers `Toolbox.specs` with every LLM request. When the model answers
with tool calls, `Toolbox.run` validates each call's arguments, executes it against the catalogue
or TMDB, records the outcome in the turn's `ToolState` and returns the JSON the model reads next.

| Tool            | The model uses it to                    | Effect on ToolState              |
|-----------------|-----------------------------------------|----------------------------------|
| search_movies   | retrieve candidates by a description    | replaces `candidates` with hits  |
| show_trailer    | show a trailer when it decides the look | sets `trailer`                   |
|                 | of the film matters (optional)          |                                  |
| recommend       | give the final, structured answer       | sets `answer` (ends the turn)    |
|                 |                                         | unless every pick is rejected    |

Grounding: show_trailer and recommend accept only movies from the latest search_movies result, so
the model cannot recommend or show a film outside the catalogue or outside the search it relies on.
show_trailer is offered only when TMDB is configured. Tuned for a small local model
(docs/03-llm-and-tool-calling.md §3.10).
Spec: docs/03-llm-and-tool-calling.md.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from backend.core.catalog import Catalog, Embed, Hit, Movie

SEARCH_K = 12  # candidates per search; docs/06-evaluation.md measures Recall@12
MAX_PICKS = 5  # with the reply and why limits below, the largest recommend fits LLM_MAX_TOKENS
OVERVIEW_CHARS = 200  # keeps 12 candidates around 1,000 prompt tokens for a small model

Handled = tuple[dict, str, dict]
"""A handler's (JSON result for the model, Ukrainian outcome, data for the trace panel)."""

FindTrailer = Callable[[int], str | None]
"""TMDB id -> YouTube key of the movie's trailer, or None (backend/adapters/tmdb.py)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text exactly as the model wrote it; the tool validates it


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...]

    def as_message(self) -> dict:
        """The turn as an OpenAI chat message, to append to the conversation."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        return message


class SearchMoviesArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=400,
        description="Description of the wanted film in the viewer's language (do not translate): "
        "plot, themes, mood, style; for 'like film X', its traits plus its title.",
    )


class Pick(BaseModel):
    movie_id: int = Field(description="id of a film from the search_movies results")
    why: str = Field(
        min_length=3,
        max_length=300,
        description="One sentence, in the viewer's language: why the film fits this request.",
    )


class ShowTrailerArgs(BaseModel):
    movie_id: int = Field(description="id of your best film from the search_movies results")
    reason: str = Field(
        min_length=3,
        max_length=300,
        description="One sentence, in the viewer's language: what exactly to watch for.",
    )


class RecommendArgs(BaseModel):
    reply: str = Field(
        min_length=1,
        max_length=600,
        description="One or two lively sentences in the viewer's language that introduce the "
        "picks without listing them.",
    )
    picks: list[Pick] = Field(
        max_length=MAX_PICKS,
        description="Usually 3 films (up to 5 if asked), best first, ids from the latest search "
        "results; empty if the request is not about films.",
    )


@dataclass(frozen=True)
class Recommendation:
    movie: Movie
    why: str


@dataclass(frozen=True)
class Answer:
    reply: str
    picks: tuple[Recommendation, ...]


@dataclass(frozen=True)
class Trailer:
    movie: Movie
    youtube_key: str
    reason: str


@dataclass(frozen=True)
class Step:
    """One executed tool call, for the trace the interface shows."""

    tool: str
    arguments: dict[str, object]
    outcome: str  # short Ukrainian summary
    data: dict[str, object] = field(default_factory=dict)  # what the trace panel explains


@dataclass
class ToolState:
    """What the tools gathered during one user turn; the agent turns it into the result."""

    genre: str | None
    candidates: dict[int, Hit] = field(default_factory=dict)  # the grounding set
    trailer: Trailer | None = None
    answer: Answer | None = None
    steps: list[Step] = field(default_factory=list)


@dataclass(frozen=True)
class Toolbox:
    catalog: Catalog
    embed: Embed
    find_trailer: (
        FindTrailer | None
    )  # None when TMDB is not configured: trailer requests are ignored

    @property
    def available(self) -> dict[str, Tool]:
        """The tools this toolbox offers: show_trailer only when trailers are configured."""
        return {t.name: t for t in TOOLS if t.name != "show_trailer" or self.find_trailer}

    @property
    def specs(self) -> list[dict]:
        """The tool definitions sent to the model (OpenAI `tools` format)."""
        return [tool.spec() for tool in self.available.values()]

    def run(self, call: ToolCall, state: ToolState) -> str:
        """Execute one tool call and return its JSON result for the model."""
        tool = self.available.get(call.name)
        if tool is None:
            return self._fail(state, call, f"unknown tool {call.name!r}")
        try:
            args = tool.args_model.model_validate_json(call.arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            return self._fail(state, call, f"invalid arguments: {problems}")
        result, outcome, data = tool.handler(self, args, state)
        state.steps.append(Step(call.name, args.model_dump(), outcome, data))
        return json.dumps(result, ensure_ascii=False)

    def search_movies(self, args: SearchMoviesArgs, state: ToolState) -> Handled:
        hits = self.catalog.search(self.embed([args.query])[0], k=SEARCH_K, genre=state.genre)
        state.candidates = {
            hit.movie.id: hit for hit in hits
        }  # the latest search grounds the answer
        nearest = ", ".join(f"{h.movie.title} ({h.score:.2f})" for h in hits[:3])
        outcome = f"знайдено {len(hits)} фільмів" + (f"; найближчі: {nearest}" if hits else "")
        ranked = [
            {
                "id": h.movie.id,
                "title": h.movie.title,
                "year": h.movie.year,
                "score": round(h.score, 3),
            }
            for h in hits
        ]
        return {"movies": [candidate(hit.movie) for hit in hits]}, outcome, {"candidates": ranked}

    def show_trailer(self, args: ShowTrailerArgs, state: ToolState) -> Handled:
        hit = state.candidates.get(args.movie_id)
        if hit is None:
            error = {"error": "movie_id is not among the latest search results"}
            return error, "помилка: фільм не з пошуку", {}
        key = self.find_trailer(hit.movie.id) if self.find_trailer else None  # offered only with it
        status = "shown" if key else "not_found"
        data = {"id": hit.movie.id, "title": hit.movie.title, "status": status}
        if key is None:
            return {"status": status}, f"трейлер «{hit.movie.title}» не знайдено", data
        state.trailer = Trailer(hit.movie, key, args.reason)
        return {"status": status}, f"трейлер «{hit.movie.title}» показано", data

    def recommend(self, args: RecommendArgs, state: ToolState) -> Handled:
        accepted: dict[int, Recommendation] = {}
        rejected: list[int] = []
        for pick in args.picks:
            hit = state.candidates.get(pick.movie_id)
            if hit is None or pick.movie_id in accepted:
                rejected.append(pick.movie_id)
            else:
                accepted[pick.movie_id] = Recommendation(hit.movie, pick.why)
        if args.picks and not accepted:  # nothing grounded: the model must choose again
            error = "none of the movie_ids are among the latest search results; choose from them"
            note = f"помилка: відхилено всі {len(rejected)}"
            return (
                {"error": error, "rejected": rejected},
                note,
                {"accepted": [], "rejected": rejected},
            )
        state.answer = Answer(args.reply, tuple(accepted.values()))
        result = {"status": "ok", "accepted": list(accepted), "rejected": rejected}
        titles = ", ".join(r.movie.title for r in accepted.values())
        outcome = f"прийнято {len(accepted)}" + (f": {titles}" if titles else "")
        outcome += f"; відхилено {len(rejected)}" + (
            f" (id {', '.join(map(str, rejected))})" if rejected else ""
        )
        return result, outcome, {"accepted": list(accepted), "rejected": rejected}

    @staticmethod
    def _fail(state: ToolState, call: ToolCall, error: str) -> str:
        state.steps.append(Step(call.name, {}, f"помилка: {error}"))
        return json.dumps({"error": error})


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Toolbox, BaseModel, ToolState], Handled]

    def spec(self) -> dict:
        """The OpenAI tool definition; its parameters are the args model's JSON schema."""
        return tool_spec(self.name, self.description, self.args_model)


TOOLS = (
    Tool(
        "search_movies",
        "Semantic search over the film catalogue. Returns up to 12 candidates "
        "(id, title, year, genres, rating, overview).",
        SearchMoviesArgs,
        Toolbox.search_movies,
    ),
    Tool(
        "show_trailer",
        "Show the viewer the trailer of your best candidate film, when its look, atmosphere or "
        "spectacle matters. Optional; call it before recommend.",
        ShowTrailerArgs,
        Toolbox.show_trailer,
    ),
    Tool(
        "recommend",
        "Give the final answer: a short introduction and the chosen films with reasons. "
        "Call it exactly once, at the end.",
        RecommendArgs,
        Toolbox.recommend,
    ),
)


def candidate(movie: Movie) -> dict[str, object]:
    """A movie as the model sees it in search results."""
    return {
        "id": movie.id,
        "title": movie.title,
        "year": movie.year,
        "genres": list(movie.genres),
        "rating": movie.rating,
        "overview": textwrap.shorten(movie.overview, OVERVIEW_CHARS, placeholder=" …"),
    }


def tool_spec(name: str, description: str, args_model: type[BaseModel]) -> dict:
    """An OpenAI tool definition whose parameters are the pydantic model's JSON schema.

    The same model validates the arguments, so schema and validation cannot drift apart.
    `$ref`s are inlined, optional fields lose their `null` alternative (absence from
    `required` already says "optional") and titles are dropped: plain JSON Schema that every
    server, and a small model, reads without surprises.
    """
    schema = args_model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(definitions[node["$ref"].rsplit("/", 1)[-1]])
            options = [o for o in node.get("anyOf", []) if o != {"type": "null"}]
            if len(options) == 1:  # `X | None`: keep X and the field's own keys, drop null
                rest = {k: v for k, v in node.items() if k not in ("anyOf", "default")}
                return inline({**options[0], **rest})
            return {
                k: inline(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))
            }
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": inline(schema)},
    }
