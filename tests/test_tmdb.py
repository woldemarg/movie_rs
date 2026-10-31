import logging

import httpx

from backend.adapters.tmdb import TmdbTrailers

KEY = "tmdb-secret"


def trailers(handler) -> TmdbTrailers:
    return TmdbTrailers(KEY, "https://tmdb.test/3", transport=httpx.MockTransport(handler))


def videos(*items: dict) -> httpx.Response:
    return httpx.Response(200, json={"id": 78, "results": list(items)})


def test_prefers_an_official_youtube_trailer():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/3/movie/78/videos"
        assert request.url.params["api_key"] == KEY
        return videos(
            {"site": "YouTube", "type": "Teaser", "key": "teaser", "official": True},
            {"site": "Vimeo", "type": "Trailer", "key": "vimeo", "official": True},
            {"site": "YouTube", "type": "Trailer", "key": "fan", "official": False},
            {"site": "YouTube", "type": "Trailer", "key": "official", "official": True},
            {"site": "YouTube", "type": "Featurette", "key": "featurette", "official": True},
        )

    assert trailers(handler).find(78) == "official"


def test_falls_back_to_a_teaser_and_to_none():
    teaser = {"site": "YouTube", "type": "Teaser", "key": "teaser"}
    assert trailers(lambda r: videos(teaser)).find(1) == "teaser"
    assert trailers(lambda r: videos()).find(1) is None


def test_tmdb_failure_means_no_trailer_and_the_key_stays_out_of_the_log(caplog):
    caplog.set_level(logging.WARNING)

    assert trailers(lambda r: httpx.Response(401, json={})).find(1) is None

    assert "TMDB" in caplog.text
    assert KEY not in caplog.text
