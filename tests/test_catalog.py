from pathlib import Path

import numpy as np
import pytest

from backend.core.catalog import Catalog
from tests.doubles import fake_embed

SHIPPED_DATA = Path("data")


def search_titles(catalog: Catalog, query: str, k: int = 4, genre: str | None = None) -> list[str]:
    return [hit.movie.title for hit in catalog.search(fake_embed([query])[0], k=k, genre=genre)]


def test_search_ranks_movies_by_similarity_to_the_query(movies):
    catalog = Catalog.build(movies, fake_embed, embedding_model="fake")
    assert search_titles(catalog, "ghosts haunt a house")[:2] == ["The Haunting", "Ghost Ship"]


def test_search_keeps_only_the_requested_genre_and_at_most_k_hits(movies):
    catalog = Catalog.build(movies, fake_embed, embedding_model="fake")
    assert search_titles(catalog, "ghosts haunt a house", k=2, genre="Science Fiction") == [
        "Ghost Ship",
        "Star Voyage",
    ]
    assert search_titles(catalog, "love", genre="Western") == []


def test_saved_catalogue_loads_back_unchanged(movies, tmp_path):
    catalog = Catalog.build(movies, fake_embed, embedding_model="fake")
    catalog.save(tmp_path, provenance={"source": "test"})

    loaded = Catalog.load(tmp_path)

    assert loaded.movies == catalog.movies
    assert loaded.embedding_model == "fake"
    np.testing.assert_array_equal(loaded.vectors, catalog.vectors)
    assert loaded.get(3).title == "Ghost Ship"
    assert loaded.get(99) is None
    assert loaded.genres == ["Comedy", "Horror", "Romance", "Science Fiction"]


def test_catalogue_rejects_vectors_not_aligned_with_movies(movies):
    with pytest.raises(ValueError, match="aligned"):
        Catalog(tuple(movies), np.zeros((2, 3), dtype=np.float32), "fake")


def test_shipped_catalogue_is_consistent():
    catalog = Catalog.load(SHIPPED_DATA)
    ids = [movie.id for movie in catalog.movies]

    assert len(catalog.movies) > 2000
    assert len(set(ids)) == len(ids)
    assert catalog.vectors.shape == (len(ids), 384)
    np.testing.assert_allclose(np.linalg.norm(catalog.vectors, axis=1), 1.0, atol=1e-4)
    assert all(movie.genres and movie.overview and movie.poster_url for movie in catalog.movies)


def test_an_interrupted_write_leaves_the_previous_file_intact(tmp_path):
    from backend.core.catalog import replace_atomically

    target = tmp_path / "movies.json"
    target.write_text("previous", encoding="utf-8")

    def crash_midway(file) -> None:
        file.write(b"half a catal")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        replace_atomically(target, crash_midway)

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [target]
