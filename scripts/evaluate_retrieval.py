"""Measure retrieval quality on the curated benchmark (data/eval/retrieval_queries.json).

Run from the repository root: `python -m scripts.evaluate_retrieval`. It prints the Markdown
tables quoted in docs/06-evaluation.md (the spec of every metric below).
"""

import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from backend.adapters.embedder import load_embedder
from backend.core.catalog import Catalog, Hit
from backend.core.tools import SEARCH_K

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "eval" / "retrieval_queries.json"
CUTOFFS = (1, 5, SEARCH_K)  # SEARCH_K: the candidates one search hands to the LLM
QUERY_FORMS = {
    "title_uk": "назва українською",
    "description_uk": "опис українською",
    "description_en": "опис англійською (як формулює LLM)",
}


def rank_of(movie_id: int, ranked_ids: Sequence[int]) -> int | None:
    """1-based position of the movie in the ranking, None when absent."""
    return ranked_ids.index(movie_id) + 1 if movie_id in ranked_ids else None


def known_item_metrics(ranks: Sequence[int | None], cutoffs: Sequence[int]) -> dict[str, float]:
    """Recall@k for each cutoff (share of targets ranked within k) and the mean reciprocal rank."""
    metrics = {f"R@{k}": statistics.fmean(r is not None and r <= k for r in ranks) for k in cutoffs}
    metrics["MRR"] = statistics.fmean(1 / r if r else 0.0 for r in ranks)
    return metrics


def genre_precision(hit_genres: Sequence[Sequence[str]], target: Sequence[str]) -> float:
    """Share of hits that have at least one of the target genres."""
    return statistics.fmean(bool(set(genres) & set(target)) for genres in hit_genres)


def format_row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(f"{c:.3f}" if isinstance(c, float) else str(c) for c in cells) + " |"


def main() -> None:
    catalog = Catalog.load(ROOT / "data")
    started = time.perf_counter()
    embed = load_embedder(catalog.embedding_model)
    load_seconds = time.perf_counter() - started
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    n = len(catalog.movies)
    query_ms: list[float] = []

    def search(query: str, k: int) -> list[Hit]:
        started = time.perf_counter()
        hits = catalog.search(embed([query])[0], k=k)
        query_ms.append((time.perf_counter() - started) * 1000)
        return hits

    labels = [*(f"R@{k}" for k in CUTOFFS), "MRR"]
    print(f"Known-item: {len(benchmark['known_item'])} movies, catalogue of {n}\n")
    print(format_row(["Форма запиту", *labels]))
    print(format_row(["---"] * (len(labels) + 1)))
    for form, label in QUERY_FORMS.items():
        ranks = [
            rank_of(item["id"], [hit.movie.id for hit in search(item[form], k=n)])
            for item in benchmark["known_item"]
        ]
        print(format_row([label, *known_item_metrics(ranks, CUTOFFS).values()]))
    random_mrr = sum(1 / r for r in range(1, n + 1)) / n
    print(format_row(["випадковий порядок", *(k / n for k in CUTOFFS), random_mrr]))

    k = max(CUTOFFS)
    print(f"\nThematic: precision@{k} by genre\n")
    print(format_row(["Запит", "Жанри", f"P@{k}", "Частка в каталозі", "Lift"]))
    print(format_row(["---"] * 5))
    precisions = []
    for item in benchmark["thematic"]:
        hits = search(item["query"], k=k)
        precision = genre_precision([hit.movie.genres for hit in hits], item["genres"])
        share = genre_precision([movie.genres for movie in catalog.movies], item["genres"])
        precisions.append(precision)
        genres = ", ".join(item["genres"])
        print(format_row([item["query"], genres, precision, share, f"×{precision / share:.1f}"]))
    print(format_row(["**середнє**", "", statistics.fmean(precisions), "", ""]))

    p95 = statistics.quantiles(query_ms, n=20)[18]
    print(f"\nLatency: model load {load_seconds:.1f} s; embed + search median ", end="")
    print(f"{statistics.median(query_ms):.0f} ms, p95 {p95:.0f} ms over {len(query_ms)} queries")


if __name__ == "__main__":
    main()
