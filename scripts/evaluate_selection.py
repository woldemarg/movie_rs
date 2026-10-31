"""How the viewer's parameters (genre filter, age) change the selection, against open requests.

Level 1, retrieval (deterministic, no LLM): the 12-film shortlist for each request with and without
its genre filter. Level 2, end to end (the live model through the running API): the agent's picks
for the same request open and with parameters, repeated. Selection density is the cosine
similarity of the shortlist and of the picks to the viewer's own words (mean and sum).

Run from the repository root with the stack up: `python -m scripts.evaluate_selection`
(`--retrieval-only` skips level 2). Writes data/eval/selection_run.json and prints the tables of
docs/09-selection-parameters.md, which defines every metric.
"""

import argparse
import json
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
RUN_FILE = ROOT / "data" / "eval" / "selection_run.json"
MATURE = {"Horror", "Thriller", "Crime", "War"}  # genres a child's selection should avoid


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    catalog = Catalog.load(ROOT / "data")
    embed = load_embedder(catalog.embedding_model)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    retrieval = retrieval_level(catalog, embed, benchmark["queries"])
    if args.retrieval_only:
        return
    print("\nEnd-to-end runs:")
    runs = end_to_end_level(catalog, embed, benchmark, args)
    summary = summarise(catalog, embed, runs, [c["name"] for c in benchmark["configs"]])
    model = httpx.get(f"{args.api}/api/config").json()["providers"]
    record = {
        "date": date.today().isoformat(),
        "provider": args.provider,
        "providers": model,
        "repeats": args.repeats,
        "retrieval": retrieval,
        "summary": summary,
        "runs": runs,
    }
    RUN_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved {RUN_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
