"""Shared fixtures."""

import pytest

from backend.core.catalog import Movie
from backend.settings import Settings
from tests.doubles import make_movie
from tests.servers import ServerThread, stub_app


@pytest.fixture(scope="session")
def stub():
    """The stub LLM and TMDB server; `stub.app.state.requests` records every completion request."""
    app = stub_app()
    with ServerThread(app) as server:
        server.app = app
        yield server


def stub_settings(stub, **overrides) -> Settings:
    """Settings wired to the stub server; never reads the developer's .env."""
    values = {
        "local_llm_base_url": f"{stub.url}/v1",
        "local_llm_model": "stub-local",
        "llm_base_url": f"{stub.url}/v1",
        "llm_model": "stub-remote",
        "llm_api_key": "",
        "tmdb_api_key": "tmdb-test",
        "tmdb_base_url": f"{stub.url}/3",
    }
    return Settings(_env_file=None, **values | overrides)


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
