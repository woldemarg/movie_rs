"""First-run data preparation: build the catalogue from data/raw when it is missing.

The catalogue (data/movies.json, data/movie_vectors.npy) is generated, not versioned: only the raw
dataset lives in git. `python -m scripts.prepare_data` (run by the backend container before the API starts) builds it once
and refreshes the posters when TMDB_API_KEY is set; afterwards it does nothing.
Spec: docs/desc/02-data-and-embeddings.md §2.2.
"""

from collections.abc import Callable
from pathlib import Path

from backend.core.catalog import MOVIES_FILE, VECTORS_FILE
from backend.settings import ROOT, Settings


def prepare(
    data_dir: Path, build: Callable[[], None], refresh_posters: Callable[[], None], tmdb_key: bool
) -> str:
    """Build the catalogue unless both files exist; refresh posters after a build when possible."""
    if (data_dir / MOVIES_FILE).exists() and (data_dir / VECTORS_FILE).exists():
        return "ready"
    build()
    if not tmdb_key:
        return "built; posters keep the dataset's paths (set TMDB_API_KEY to refresh them)"
    refresh_posters()
    return "built, posters refreshed"


def main() -> None:
    from scripts import build_catalog, refresh_posters  # the embedding model loads only if needed

    tmdb_key = bool(Settings().tmdb_api_key.get_secret_value())
    print("catalogue:", prepare(ROOT / "data", build_catalog.main, refresh_posters.main, tmdb_key))


if __name__ == "__main__":
    main()
