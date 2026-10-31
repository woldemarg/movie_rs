"""Deterministic test doubles: tests never load the real embedding model or touch the network."""

import json
import re
import zlib
from collections.abc import Sequence

import numpy as np

from backend.core.catalog import Movie
from backend.core.tools import AssistantTurn, ToolCall

FAKE_DIM = 384  # the shipped vectors' dimension, so the fake also works against data/


def fake_embed(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-words hashing embedder: texts sharing words get a higher cosine."""
    vectors = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % FAKE_DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def make_movie(movie_id: int, title: str, genres: tuple[str, ...], overview: str) -> Movie:
    return Movie(
        id=movie_id,
        title=title,
        year=2000 + movie_id,
        genres=genres,
        rating=6.5,
        votes=1000,
        overview=overview,
        poster_path=f"/poster{movie_id}.jpg",
    )


def tool_call(name: str, call_id: str = "", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=json.dumps(arguments))


def assistant(*calls: ToolCall, content: str = "") -> AssistantTurn:
    return AssistantTurn(content=content, tool_calls=calls)


class ScriptedChat:
    """An LLM that replays prepared turns and records every request it receives."""

    def __init__(self, *turns: AssistantTurn) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[list[dict], list[dict]]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.requests.append(([dict(m) for m in messages], tools))
        return self.turns.pop(0)


class FakeTrailers:
    """TMDB stand-in: a trailer key for the listed movie ids, None for the rest."""

    def __init__(self, *movie_ids: int) -> None:
        self.movie_ids = set(movie_ids)

    def find(self, movie_id: int) -> str | None:
        return f"yt-{movie_id}" if movie_id in self.movie_ids else None
