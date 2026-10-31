"""The movie catalogue: what the recommender may suggest, and semantic search over it.

Spec: docs/desc/02-data-and-embeddings.md.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import BinaryIO

import numpy as np

Embed = Callable[[Sequence[str]], np.ndarray]
"""Texts -> unit-length vectors, one row per text (implemented in backend/adapters/embedder.py)."""

MOVIES_FILE = "movies.json"
VECTORS_FILE = "movie_vectors.npy"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"


@dataclass(frozen=True)
class Movie:
    id: int  # TMDB id
    title: str
    year: int | None
    genres: tuple[str, ...]
    rating: float  # weighted rating, docs/desc/02-data-and-embeddings.md
    votes: int
    overview: str
    poster_path: str | None

    @property
    def poster_url(self) -> str | None:
        return f"{POSTER_BASE_URL}{self.poster_path}" if self.poster_path else None

    def embedding_text(self) -> str:
        """The text whose vector represents the movie in search."""
        return f"{self.title}. {', '.join(self.genres)}. {self.overview}"


@dataclass(frozen=True)
class Hit:
    movie: Movie
    score: float  # cosine similarity to the query


@dataclass(frozen=True)
class Catalog:
    movies: tuple[Movie, ...]
    vectors: np.ndarray  # (len(movies), dim) unit rows; row i represents movies[i]
    embedding_model: str  # the model that made the vectors; queries must use the same one

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.movies):
            raise ValueError(
                f"vectors {self.vectors.shape} are not aligned with {len(self.movies)} movies"
            )

    @classmethod
    def build(cls, movies: Sequence[Movie], embed: Embed, embedding_model: str) -> Catalog:
        return cls(tuple(movies), embed([m.embedding_text() for m in movies]), embedding_model)

    @classmethod
    def load(cls, data_dir: Path) -> Catalog:
        meta = json.loads((data_dir / MOVIES_FILE).read_text(encoding="utf-8"))
        movies = tuple(Movie(**{**m, "genres": tuple(m["genres"])}) for m in meta["movies"])
        return cls(movies, np.load(data_dir / VECTORS_FILE), meta["embedding_model"])

    def save(self, data_dir: Path, provenance: dict[str, object]) -> None:
        meta = {
            "embedding_model": self.embedding_model,
            **provenance,
            "movies": [asdict(movie) for movie in self.movies],
        }
        write_catalog_json(data_dir / MOVIES_FILE, meta)
        vectors = self.vectors.astype(np.float32)
        replace_atomically(data_dir / VECTORS_FILE, lambda file: np.save(file, vectors))

    @cached_property
    def genres(self) -> list[str]:
        return sorted({genre for movie in self.movies for genre in movie.genres})

    @cached_property
    def _by_id(self) -> dict[int, Movie]:
        return {movie.id: movie for movie in self.movies}

    def get(self, movie_id: int) -> Movie | None:
        return self._by_id.get(movie_id)

    def search(self, query_vector: np.ndarray, k: int, genre: str | None = None) -> list[Hit]:
        """The k movies closest to a unit query vector, optionally only those of one genre."""
        scores = self.vectors @ query_vector
        if genre:
            in_genre = np.fromiter((genre in m.genres for m in self.movies), dtype=bool)
            scores = np.where(in_genre, scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.movies[i], float(scores[i])) for i in top if np.isfinite(scores[i])]


def write_catalog_json(path: Path, meta: dict) -> None:
    """The one writer of movies.json, so every tool keeps the same diff-friendly format."""
    text = json.dumps(meta, ensure_ascii=False, indent=1) + "\n"
    replace_atomically(path, lambda file: file.write(text.encode("utf-8")))


def replace_atomically(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a temporary sibling, then rename it over `path`: an interrupted write keeps the old file."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            write(file)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
