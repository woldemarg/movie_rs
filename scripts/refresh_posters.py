"""Refresh poster paths in data/movies.json from the TMDB API.

The 2017 dataset's poster paths no longer resolve (TMDB rotates its images), so after
`python -m scripts.build_catalog` run `python -m scripts.refresh_posters` (needs TMDB_API_KEY;
about 2,300 requests, under a minute). Spec: docs/02-data-and-embeddings.md.
"""

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx

from backend.core.catalog import MOVIES_FILE, write_catalog_json
from backend.settings import ROOT, Settings

WORKERS = 8  # parallel requests, well under TMDB's rate limit


def refresh_posters(movies: list[dict], fetch: Callable[[int], str | None]) -> tuple[int, int]:
    """Set each movie's `poster_path` to TMDB's current one (None: TMDB has no poster).

    A failed lookup keeps the old value, so a rerun fills the gaps. Returns (updated, failed).
    """

    def lookup(movie: dict) -> tuple[bool, str | None]:
        try:
            return True, fetch(movie["id"])
        except httpx.HTTPError:
            return False, None

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lookup, movies))
    for movie, (ok, path) in zip(movies, results, strict=True):
        if ok:
            movie["poster_path"] = path
    succeeded = sum(ok for ok, _ in results)
    return succeeded, len(movies) - succeeded


def main() -> None:
    settings = Settings()
    key = settings.tmdb_api_key.get_secret_value()
    if not key:
        sys.exit("TMDB_API_KEY is not set (see .env.sample)")
    path = ROOT / "data" / MOVIES_FILE
    catalog = json.loads(path.read_text(encoding="utf-8"))
    with httpx.Client(timeout=10) as http:

        def fetch(movie_id: int) -> str | None:
            response = http.get(
                f"{settings.tmdb_base_url}/movie/{movie_id}", params={"api_key": key}
            )
            response.raise_for_status()
            return response.json().get("poster_path")

        updated, failed = refresh_posters(catalog["movies"], fetch)
    catalog["posters_refreshed"] = date.today().isoformat()
    write_catalog_json(path, catalog)
    print(f"posters: {updated} refreshed, {failed} lookups failed (kept)")


if __name__ == "__main__":
    main()
