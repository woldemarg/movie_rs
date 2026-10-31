import pytest

from scripts.build_catalog import credibility_threshold, select_movies, weighted_rating

GENRES = "[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}]"


def row(movie_id="1", votes: str | None = "500", **overrides) -> dict[str, str | None]:
    base = {
        "id": movie_id,
        "title": f"Movie {movie_id}",
        "adult": "False",
        "genres": GENRES,
        "overview": "A story.",
        "release_date": "1999-05-01",
        "vote_average": "7.0",
        "vote_count": votes,
        "poster_path": f"/p{movie_id}.jpg",
    }
    return base | overrides


def test_credibility_threshold_is_the_95th_percentile_of_numeric_vote_counts():
    rows = [row(votes=str(v)) for v in range(101)] + [row(votes="n/a"), row(votes=None)]
    assert credibility_threshold(rows) == pytest.approx(95.0)


def test_weighted_rating_shrinks_towards_the_prior_when_votes_are_few():
    assert weighted_rating(average=9.0, votes=100, min_votes=100) == pytest.approx(7.0)
    assert weighted_rating(average=9.0, votes=10_000, min_votes=100) == pytest.approx(
        8.96, abs=0.01
    )


def test_select_movies_keeps_credible_complete_movies_only():
    rows = [
        row("1"),
        row("2", votes="10"),
        row("3", adult="True"),
        row("4", overview="  "),
        row("5", genres="[]"),
        row("6", votes="oops"),
        row("8", votes=None),  # csv.DictReader fills the missing cells of a short row with None
        row("1", title="Duplicate id"),
        row("7", release_date="", poster_path=""),
    ]

    movies = select_movies(rows, min_votes=100)

    assert [movie.id for movie in movies] == [1, 7]
    first, undated = movies
    assert first.title == "Movie 1"
    assert first.year == 1999
    assert first.genres == ("Drama", "Comedy")
    assert first.votes == 500
    assert first.rating == pytest.approx(round(weighted_rating(7.0, 500, 100), 2))
    assert undated.year is None
    assert undated.poster_path is None
