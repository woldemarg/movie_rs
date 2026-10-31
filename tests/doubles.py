"""Deterministic test doubles: tests never load the real embedding model or touch the network."""

import re
import zlib
from collections.abc import Sequence

import numpy as np

from backend.core.catalog import Movie

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
