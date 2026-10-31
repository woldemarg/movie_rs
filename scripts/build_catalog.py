"""Build the catalogue (data/movies.json + data/movie_vectors.npy) from the raw dataset.

Run from the repository root: `python -m scripts.build_catalog` (about 3 minutes on a CPU;
the first run downloads the embedding model into models/embeddings/). Spec: docs/02-data-and-embeddings.md.
"""

import ast
import csv
import statistics
from collections.abc import Iterable
from pathlib import Path

from backend.adapters.embedder import DEFAULT_MODEL, load_embedder
from backend.core.catalog import Catalog, Movie

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "movies_metadata.csv"
SOURCE = "The Movies Dataset (Kaggle, CC0): data/raw/movies_metadata.csv"
VOTES_PERCENTILE = 95  # a movie needs this share of the dataset below it in votes to be credible
PRIOR_RATING = 5.0  # the rating a movie with no votes is assumed to have


def parse_number(text: str | None) -> float | None:
    """A float, or None for an empty, malformed or missing cell (short rows give None)."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def credibility_threshold(rows: Iterable[dict[str, str]]) -> float:
    """The vote count at the VOTES_PERCENTILE of all rows with a numeric vote count."""
    votes = [v for v in (parse_number(row["vote_count"]) for row in rows) if v is not None]
    return statistics.quantiles(votes, n=100, method="inclusive")[VOTES_PERCENTILE - 1]


def weighted_rating(average: float, votes: float, min_votes: float) -> float:
    """The IMDb-style Bayesian average: few votes pull the rating towards PRIOR_RATING."""
    return (votes * average + min_votes * PRIOR_RATING) / (votes + min_votes)


def select_movies(rows: Iterable[dict[str, str]], min_votes: float) -> list[Movie]:
    """Credible (votes >= min_votes), non-adult movies with a title, genres and an overview."""
    movies: dict[int, Movie] = {}
    for row in rows:
        votes = parse_number(row["vote_count"])
        average = parse_number(row["vote_average"])
        if votes is None or average is None or votes < min_votes or row["adult"] != "False":
            continue
        genres = tuple(genre["name"] for genre in ast.literal_eval(row["genres"] or "[]"))
        title, overview = row["title"].strip(), row["overview"].strip()
        if not (genres and title and overview and row["id"].isdigit()):
            continue
        year = row["release_date"][:4]
        movies.setdefault(
            int(row["id"]),
            Movie(
                id=int(row["id"]),
                title=title,
                year=int(year) if year.isdigit() else None,
                genres=genres,
                rating=round(weighted_rating(average, votes, min_votes), 2),
                votes=int(votes),
                overview=overview,
                poster_path=row["poster_path"] or None,
            ),
        )
    return list(movies.values())


def main() -> None:
    with RAW_CSV.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    min_votes = credibility_threshold(rows)
    movies = select_movies(rows, min_votes)
    catalog = Catalog.build(movies, load_embedder(DEFAULT_MODEL), DEFAULT_MODEL)
    catalog.save(ROOT / "data", provenance={"source": SOURCE, "min_votes": min_votes})
    print(f"{len(rows)} rows -> {len(movies)} movies (min votes {min_votes:.0f})")


if __name__ == "__main__":
    main()
