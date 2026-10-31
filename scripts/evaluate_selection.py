"""How the viewer's parameters (genre filter, age) change the selection, against open requests.

Level 1, retrieval (deterministic, no LLM): the 12-film shortlist for each request with and without
its genre filter. Level 2, end to end (the live model through the running API): the agent's picks
for the same request open and with parameters, repeated. Selection density is the cosine
similarity of the shortlist and of the picks to the viewer's own words (mean and sum); the rating
analysis scores the picks' weighted ratings against the shortlist and against plain search.

Run from the repository root with the stack up: `python -m scripts.evaluate_selection
--provider local|openrouter` (`--retrieval-only` skips level 2, `--rescore` re-analyses the saved
run without calling the model). Writes data/eval/selection_run_<provider>.json and prints the
tables of docs/09-selection-parameters.md, which defines every metric.
"""

import argparse
import json
import math
import statistics
import time
from collections.abc import Collection, Mapping, Sequence
from datetime import date
from pathlib import Path

import httpx
import numpy as np

from backend.adapters.embedder import load_embedder
from backend.core.catalog import Catalog
from backend.core.tools import SEARCH_K

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "eval" / "selection_queries.json"
MATURE = {"Horror", "Thriller", "Crime", "War"}  # genres a child's selection should avoid
RATING_KEYS = ("rating_sum", "rating_mean", "lift", "efficiency", "vs_search", "popularity_lift")


def density(
    vectors: Mapping[int, np.ndarray], query: np.ndarray, ids: Sequence[int]
) -> tuple[float, float]:
    """(mean, sum) of the cosines between the query and the films; unit vectors, so dot products."""
    cosines = [float(vectors[i] @ query) for i in ids]
    return (statistics.fmean(cosines), sum(cosines)) if cosines else (0.0, 0.0)


def jaccard(a: Collection[int], b: Collection[int]) -> float:
    """Overlap of two selections; two empty selections are identical."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def mean_rank(picks: Sequence[int], shortlist: Sequence[int]) -> float:
    """Mean 1-based position of the picks in the similarity-ranked shortlist."""
    ranks = [shortlist.index(p) + 1 for p in picks if p in shortlist]
    return statistics.fmean(ranks) if ranks else 0.0


def share_with(
    genres: Mapping[int, Sequence[str]], ids: Sequence[int], wanted: Collection[str]
) -> float:
    """Share of the films that have at least one of the wanted genres."""
    return statistics.fmean(bool(set(genres[i]) & set(wanted)) for i in ids) if ids else 0.0


def copies_words(model_query: str, query: str) -> bool:
    """Whether the model searched with the viewer's own words instead of describing the film."""
    return model_query.strip().rstrip(".").lower() == query.strip().rstrip(".").lower()


def top_share(picks: Sequence[int], shortlist: Sequence[int], k: int = 3) -> float:
    """Share of the picks that are among the k candidates closest to the model's query."""
    return statistics.fmean(p in shortlist[:k] for p in picks) if picks else 0.0


def mean_log_votes(votes: Mapping[int, float], ids: Sequence[int]) -> float:
    return statistics.fmean(math.log10(votes[i]) for i in ids)


def rating_scores(
    ratings: Mapping[int, float],
    votes: Mapping[int, float],
    picks: Sequence[int],
    shortlist: Sequence[int],
    search: Sequence[int],
) -> dict[str, float]:
    """One turn's picks against its shortlist and against plain search for as many films.

    rating_sum: the picks' weighted ratings summed; efficiency: that sum over the best sum the
    shortlist allowed; vs_search: that sum minus plain search's first films; popularity_lift: mean
    log10 votes of the picks minus the shortlist's.
    """
    n = len(picks)
    if n == 0:
        return dict.fromkeys(RATING_KEYS, 0.0)
    total = sum(ratings[i] for i in picks)
    best = sum(sorted((ratings[i] for i in shortlist), reverse=True)[:n])
    return {
        "rating_sum": total,
        "rating_mean": total / n,
        "lift": total / n - statistics.fmean(ratings[i] for i in shortlist),
        "efficiency": total / best,
        "vs_search": total - sum(ratings[i] for i in search[:n]),
        "popularity_lift": mean_log_votes(votes, picks) - mean_log_votes(votes, shortlist),
    }


def paired_ci(
    diffs: Sequence[float], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of paired differences with its 95 % percentile bootstrap interval (fixed seed)."""
    d = np.asarray(diffs, dtype=float)
    means = np.random.default_rng(seed).choice(d, size=(resamples, len(d))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(low), float(high)


def row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(f"{c:.3f}" if isinstance(c, float) else str(c) for c in cells) + " |"


def retrieval_level(catalog: Catalog, embed, queries: list[dict]) -> list[dict]:
    """Shortlists for the viewer's own words, open and genre-filtered (no LLM involved)."""
    vectors = dict(zip((m.id for m in catalog.movies), catalog.vectors, strict=True))
    genres = {m.id: m.genres for m in catalog.movies}
    results = []
    print(f"\n### Retrieval: {SEARCH_K} candidates for the viewer's words\n")
    print(
        row(
            [
                "Запит",
                "Жанр",
                "щільність відкрито",
                "щільність з жанром",
                "Δ",
                "перетин",
                "частка жанру: відкрито → з жанром",
            ]
        )
    )
    print(row(["---"] * 7))
    for q in queries:
        u = embed([q["text"]])[0]
        open_ids = [h.movie.id for h in catalog.search(u, k=SEARCH_K)]
        genre_ids = [h.movie.id for h in catalog.search(u, k=SEARCH_K, genre=q["genre"])]
        d_open, d_genre = density(vectors, u, open_ids)[0], density(vectors, u, genre_ids)[0]
        r = {
            "query": q["text"],
            "genre": q["genre"],
            "density_open": d_open,
            "density_genre": d_genre,
            "overlap": jaccard(open_ids, genre_ids),
            "genre_share_open": share_with(genres, open_ids, {q["genre"]}),
        }
        results.append(r)
        share = f"{r['genre_share_open']:.2f} → 1.00"
        print(row([q["text"], q["genre"], d_open, d_genre, d_genre - d_open, r["overlap"], share]))
    return results


def ask(http: httpx.Client, api: str, provider: str, text: str, viewer: dict) -> dict:
    started = time.perf_counter()
    body = {"provider": provider, "messages": [{"role": "user", "content": text}], "viewer": viewer}
    response = http.post(f"{api}/api/recommend", json=body, timeout=900)
    data = response.json()
    if response.status_code != 200:
        return {"error": data.get("detail"), "seconds": time.perf_counter() - started}
    steps = data["trace"]["steps"]
    searches = [s for s in steps if s["tool"] == "search_movies" and s["data"]]
    return {
        "picks": [p["id"] for p in data["picks"]],
        "shortlist": [c["id"] for c in searches[-1]["data"]["candidates"]] if searches else [],
        "model_query": searches[-1]["arguments"]["query"] if searches else "",
        "model_query_scores": [c["score"] for c in searches[-1]["data"]["candidates"]]
        if searches
        else [],
        "tools": [s["tool"] for s in steps],
        "llm_calls": data["trace"]["llm_calls"],
        "seconds": time.perf_counter() - started,
    }


def end_to_end_level(catalog: Catalog, embed, benchmark: dict, args) -> list[dict]:
    """The agent's picks for every request under every configuration, repeated."""
    runs = []
    with httpx.Client() as http:
        for q in benchmark["queries"]:
            for config in benchmark["configs"]:
                viewer = {"genre": q["genre"] if config["genre"] else None, "age": config["age"]}
                for repeat in range(args.repeats):
                    run = {
                        "query": q["text"],
                        "target_genre": q["genre"],
                        "config": config["name"],
                        "repeat": repeat,
                    }
                    run |= ask(http, args.api, args.provider, q["text"], viewer)
                    runs.append(run)
                    print(
                        f"  {config['name']:15} #{repeat} {run.get('seconds', 0):5.1f}s "
                        f"{run.get('error') or [catalog.get(i).title for i in run['picks']]}"
                    )
    return runs


def summarise(catalog: Catalog, embed, runs: list[dict], configs: list[str]) -> list[dict]:
    """Per configuration: means over requests and repeats (definitions: docs/09 §9.3)."""
    vectors = dict(zip((m.id for m in catalog.movies), catalog.vectors, strict=True))
    genres = {m.id: m.genres for m in catalog.movies}
    query_vectors = {text: embed([text])[0] for text in {r["query"] for r in runs}}
    ok = [r for r in runs if "picks" in r]
    open_picks = {(r["query"], r["repeat"]): r["picks"] for r in ok if r["config"] == "open"}
    summary = []
    for name in configs:
        rs = [r for r in ok if r["config"] == name]
        if not rs:
            continue
        by_query: dict[str, list[list[int]]] = {}
        for r in rs:
            by_query.setdefault(r["query"], []).append(r["picks"])
        summary.append(
            {
                "config": name,
                "turns": f"{len(rs)}/{sum(r['config'] == name for r in runs)}",
                "pick_density": statistics.fmean(
                    density(vectors, query_vectors[r["query"]], r["picks"])[0] for r in rs
                ),
                "pick_density_sum": statistics.fmean(
                    density(vectors, query_vectors[r["query"]], r["picks"])[1] for r in rs
                ),
                "shortlist_density": statistics.fmean(
                    density(vectors, query_vectors[r["query"]], r["shortlist"])[0] for r in rs
                ),
                "model_query_density": statistics.fmean(
                    statistics.fmean(r["model_query_scores"]) for r in rs if r["model_query_scores"]
                ),
                "pick_rank": statistics.fmean(mean_rank(r["picks"], r["shortlist"]) for r in rs),
                "top3_share": statistics.fmean(top_share(r["picks"], r["shortlist"]) for r in rs),
                "copied_query": statistics.fmean(
                    copies_words(r["model_query"], r["query"]) for r in rs
                ),
                "genre_share": statistics.fmean(
                    share_with(genres, r["picks"], {r["target_genre"]}) for r in rs
                ),
                "mature_share": statistics.fmean(
                    share_with(genres, r["picks"], MATURE) for r in rs
                ),
                "overlap_with_open": statistics.fmean(
                    jaccard(r["picks"], open_picks.get((r["query"], r["repeat"]), [])) for r in rs
                ),
                "stability": statistics.fmean(
                    jaccard(p[0], p[1]) for p in by_query.values() if len(p) > 1
                )
                if any(len(p) > 1 for p in by_query.values())
                else float("nan"),
                "trailer_share": statistics.fmean("show_trailer" in r["tools"] for r in rs),
                "seconds": statistics.median(r["seconds"] for r in rs),
            }
        )
    print("\n### End to end: the agent's picks by configuration\n")
    columns = [
        "config",
        "turns",
        "pick_density",
        "pick_density_sum",
        "shortlist_density",
        "model_query_density",
        "pick_rank",
        "top3_share",
        "copied_query",
        "genre_share",
        "mature_share",
        "overlap_with_open",
        "stability",
        "trailer_share",
        "seconds",
    ]
    print(row(columns))
    print(row(["---"] * len(columns)))
    for s in summary:
        print(row([s[c] for c in columns]))
    return summary


def score_runs(catalog: Catalog, embed, runs: list[dict], configs: list[dict]) -> list[dict]:
    """Rating scores per answered turn; plain search takes the viewer's words and the filter."""
    ratings = {m.id: m.rating for m in catalog.movies}
    votes = {m.id: m.votes for m in catalog.movies}
    filtered = {c["name"]: c["genre"] for c in configs}
    searches: dict[tuple[str, str | None], list[int]] = {}
    scored = []
    for r in runs:
        if "picks" not in r:
            continue
        key = (r["query"], r["target_genre"] if filtered[r["config"]] else None)
        if key not in searches:
            hits = catalog.search(embed([key[0]])[0], k=SEARCH_K, genre=key[1])
            searches[key] = [h.movie.id for h in hits]
        scores = rating_scores(ratings, votes, r["picks"], r["shortlist"], searches[key])
        scored.append(r | {"films": len(r["picks"])} | scores)
    return scored


def ci_text(estimate: Sequence[float]) -> str:
    mean, low, high = estimate
    return f"{mean:+.2f} [{low:+.2f}; {high:+.2f}]"


def rating_level(catalog: Catalog, embed, runs: list[dict], configs: list[dict]) -> list[dict]:
    """Per configuration: mean rating scores, and paired 95 % intervals against search and open."""
    scored = score_runs(catalog, embed, runs, configs)
    opened = {(s["query"], s["repeat"]): s for s in scored if s["config"] == "open"}
    summary = []
    for config in configs:
        rs = [s for s in scored if s["config"] == config["name"]]
        pairs = [(s, opened[k]) for s in rs if (k := (s["query"], s["repeat"])) in opened]
        if not pairs:
            continue
        summary.append(
            {"config": config["name"], "turns": len(rs)}
            | {k: statistics.fmean(s[k] for s in rs) for k in ("films", *RATING_KEYS)}
            | {
                "sum_vs_search": paired_ci([s["vs_search"] for s in rs]),
                "sum_vs_open": paired_ci([a["rating_sum"] - b["rating_sum"] for a, b in pairs]),
                "mean_vs_open": paired_ci([a["rating_mean"] - b["rating_mean"] for a, b in pairs]),
                "popularity": paired_ci([s["popularity_lift"] for s in rs]),
            }
        )
    print_rating(summary, statistics.fmean(m.rating for m in catalog.movies))
    return summary


def print_rating(summary: list[dict], catalogue_mean: float) -> None:
    print(f"\n### Rating: weighted rating of the picks (catalogue mean {catalogue_mean:.2f})\n")
    columns = ["config", "turns", "films", *RATING_KEYS]
    print(row(columns))
    print(row(["---"] * len(columns)))
    for s in summary:
        print(row([s[c] for c in columns]))
    print("\n### Paired differences: mean [95 % bootstrap interval]\n")
    print(row(["config", "Σ − plain search", "Σ − open", "per film − open", "popularity lift"]))
    print(row(["---"] * 5))
    for s in summary:
        keys = ("sum_vs_search", "sum_vs_open", "mean_vs_open", "popularity")
        cells = [ci_text(s[k]) for k in keys]
        print(row([s["config"], *cells]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="re-analyse the saved run")
    args = parser.parse_args()
    run_file = ROOT / "data" / "eval" / f"selection_run_{args.provider}.json"

    catalog = Catalog.load(ROOT / "data")
    embed = load_embedder(catalog.embedding_model)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    retrieval = retrieval_level(catalog, embed, benchmark["queries"])
    if args.retrieval_only:
        return
    if args.rescore:
        record = json.loads(run_file.read_text(encoding="utf-8"))
    else:
        print("\nEnd-to-end runs:")
        record = {
            "date": date.today().isoformat(),
            "provider": args.provider,
            "providers": httpx.get(f"{args.api}/api/config").json()["providers"],
            "repeats": args.repeats,
            "runs": end_to_end_level(catalog, embed, benchmark, args),
        }
    configs = benchmark["configs"]
    record |= {
        "retrieval": retrieval,
        "summary": summarise(catalog, embed, record["runs"], [c["name"] for c in configs]),
        "rating": rating_level(catalog, embed, record["runs"], configs),
    }
    run_file.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved {run_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
