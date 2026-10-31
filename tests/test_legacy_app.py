"""Characterization tests of the Streamlit prototype, pinned before the rewrite replaces it."""

import pandas as pd

import movieharbor as app


def test_normalize_genres_accepts_lists_and_stringified_lists():
    assert app.normalize_genres(["Drama", " "]) == ["Drama"]
    assert app.normalize_genres("['Comedy', 'Family']") == ["Comedy", "Family"]
    assert app.normalize_genres(None) == []


def test_semantic_search_never_ranks_the_shipped_catalogue():
    """The pickle keeps vectors under `embedding`, the search reads `vector`: it returns the head."""
    movies = pd.read_pickle("movies.pkl")
    assert "vector" not in movies.columns
    result = app.semantic_search(client=object(), movies=movies, query="space", top_k=3)
    assert result.index.tolist() == movies.index[:3].tolist()
