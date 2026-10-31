import json

import numpy as np
import pytest

from scripts.evaluate_selection import (
    BENCHMARK,
    copies_words,
    density,
    jaccard,
    mean_rank,
    paired_ci,
    rating_scores,
    share_with,
    top_share,
)


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


def test_rating_scores_compare_the_picks_with_the_shortlist_and_with_plain_search():
    ratings = {1: 8.0, 2: 7.0, 3: 6.0, 4: 5.0}
    votes = {1: 100_000, 2: 1_000, 3: 1_000, 4: 100}

    scores = rating_scores(ratings, votes, picks=[2, 3], shortlist=[4, 3, 2, 1], search=[4, 3, 2])

    assert scores["rating_sum"] == pytest.approx(13.0)
    assert scores["rating_mean"] == pytest.approx(6.5)
    assert scores["lift"] == pytest.approx(0.0)  # the shortlist's mean is 6.5 as well
    assert scores["efficiency"] == pytest.approx(13.0 / 15.0)  # the best two would sum to 15
    assert scores["vs_search"] == pytest.approx(13.0 - 11.0)  # search's first two: 5 + 6
    assert scores["popularity_lift"] == pytest.approx(3.0 - 3.25)  # mean log10 votes: 3 vs 13/4
    assert rating_scores(ratings, votes, [], [1], [1])["rating_sum"] == 0.0


def test_paired_ci_brackets_the_mean_difference_and_is_reproducible():
    diffs = [0.5, 1.0, 1.5, 2.0]

    mean, low, high = paired_ci(diffs)

    assert mean == pytest.approx(1.25)
    assert low <= mean <= high
    assert paired_ci(diffs) == (mean, low, high)
    assert paired_ci([1.0, 1.0]) == pytest.approx((1.0, 1.0, 1.0))


def test_query_copying_and_the_share_of_picks_from_the_top_three():
    assert copies_words("Історія про виживання.", "історія про виживання")
    assert not copies_words("виживання в тайзі, дика природа", "Історія про виживання")
    assert top_share([5, 1, 9], [1, 2, 5, 7, 9]) == pytest.approx(2 / 3)
    assert top_share([], [1]) == 0.0
