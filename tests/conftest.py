"""Shared fixtures."""

import pytest

from backend.core.catalog import Movie
from tests.doubles import make_movie


@pytest.fixture
def movies() -> list[Movie]:
    return [
        make_movie(
            1, "Star Voyage", ("Science Fiction",), "astronauts travel through a black hole"
        ),
        make_movie(2, "The Haunting", ("Horror",), "ghosts haunt an old family house"),
        make_movie(3, "Ghost Ship", ("Horror", "Science Fiction"), "ghosts haunt a lost spaceship"),
        make_movie(4, "Paris Love", ("Romance", "Comedy"), "two strangers fall in love in paris"),
    ]
