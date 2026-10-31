import httpx

from scripts.refresh_posters import refresh_posters


def test_posters_take_tmdbs_current_path_and_keep_theirs_when_the_lookup_fails():
    movies = [
        {"id": 1, "poster_path": "/stale1.jpg"},
        {"id": 2, "poster_path": "/stale2.jpg"},
        {"id": 3, "poster_path": "/stale3.jpg"},
    ]
    current = {1: "/fresh1.jpg", 2: None}

    def fetch(movie_id: int) -> str | None:
        if movie_id not in current:
            raise httpx.ConnectError("down")
        return current[movie_id]

    assert refresh_posters(movies, fetch) == (2, 1)
    assert [m["poster_path"] for m in movies] == ["/fresh1.jpg", None, "/stale3.jpg"]
