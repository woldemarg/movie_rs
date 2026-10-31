import json
from pathlib import Path

import pytest

from backend.core.catalog import Catalog
from scripts.evaluate_retrieval import BENCHMARK, genre_precision, known_item_metrics, rank_of


def test_rank_of_is_one_based_and_none_when_missing():
    assert rank_of(7, [3, 7, 9]) == 2
    assert rank_of(4, [3, 7, 9]) is None


def test_known_item_metrics_average_recall_at_each_cutoff_and_reciprocal_rank():
    metrics = known_item_metrics([1, 3, None, 12], cutoffs=(1, 5, 12))
    assert metrics == pytest.approx(
        {"R@1": 0.25, "R@5": 0.5, "R@12": 0.75, "MRR": (1 + 1 / 3 + 0 + 1 / 12) / 4}
    )


def test_genre_precision_is_the_share_of_hits_with_any_target_genre():
    hits = [("Horror",), ("Comedy", "Horror"), ("Drama",), ("Romance",)]
    assert genre_precision(hits, ["Horror", "Romance"]) == 0.75


def test_benchmark_refers_to_movies_and_genres_of_the_shipped_catalogue():
    catalog = Catalog.load(Path("data"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    for item in benchmark["known_item"]:
        assert catalog.get(item["id"]).title == item["title"]
    for item in benchmark["thematic"]:
        assert set(item["genres"]) <= set(catalog.genres)
