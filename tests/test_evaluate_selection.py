import json

import numpy as np
import pytest

from scripts.evaluate_selection import BENCHMARK, density, jaccard, mean_rank, share_with


def test_density_is_the_mean_and_the_sum_of_cosines_to_the_query():
    vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.6, 0.8]), 3: np.array([0.0, 1.0])}
    query = np.array([1.0, 0.0])

    assert density(vectors, query, [1, 2]) == pytest.approx((0.8, 1.6))
    assert density(vectors, query, []) == (0.0, 0.0)


def test_jaccard_rank_and_shares():
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert jaccard([], []) == 1.0
    assert mean_rank([7, 3], [3, 5, 7]) == pytest.approx(2.0)  # positions 3 and 1
    genres = {1: ("Horror",), 2: ("Comedy", "Family"), 3: ("Thriller",)}
    assert share_with(genres, [1, 2, 3], {"Horror", "Thriller"}) == pytest.approx(2 / 3)
    assert share_with(genres, [], {"Horror"}) == 0.0


def test_the_benchmark_uses_genres_of_the_catalogue():
    from pathlib import Path

    from backend.core.catalog import Catalog

    genres = set(Catalog.load(Path("data")).genres)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert {q["genre"] for q in benchmark["queries"]} <= genres
    assert [c["name"] for c in benchmark["configs"]][0] == "open"
