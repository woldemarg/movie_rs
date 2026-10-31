import json

import pytest

from backend.core.catalog import Catalog
from backend.core.tools import Toolbox, ToolCall, ToolState
from tests.doubles import FakeTrailers, fake_embed, tool_call


@pytest.fixture
def toolbox(movies) -> Toolbox:
    return Toolbox(Catalog.build(movies, fake_embed, "fake"), fake_embed, FakeTrailers(3).find)


def run(toolbox: Toolbox, state: ToolState, name: str, **arguments) -> dict:
    return json.loads(toolbox.run(tool_call(name, **arguments), state))


def test_specs_are_plain_json_schema_for_every_available_tool(toolbox, movies):
    names = [spec["function"]["name"] for spec in toolbox.specs]
    assert names == ["search_movies", "show_trailer", "recommend"]
    text = json.dumps(toolbox.specs)
    assert "$ref" not in text and "$defs" not in text
    recommend = toolbox.specs[2]["function"]["parameters"]
    assert recommend["required"] == ["reply", "picks"]
    assert recommend["properties"]["picks"]["items"]["required"] == ["movie_id", "why"]

    without_trailers = Toolbox(toolbox.catalog, fake_embed, find_trailer=None)
    assert [s["function"]["name"] for s in without_trailers.specs] == ["search_movies", "recommend"]


def test_search_returns_candidates_within_the_viewers_genre_and_remembers_them(toolbox):
    state = ToolState(genre="Horror")

    result = run(toolbox, state, "search_movies", query="ghosts haunt a house")

    assert [m["title"] for m in result["movies"]] == ["The Haunting", "Ghost Ship"]
    assert set(result["movies"][0]) == {"id", "title", "year", "genres", "rating", "overview"}
    assert set(state.candidates) == {2, 3}
    assert state.steps[0].tool == "search_movies"
    assert state.steps[0].arguments == {"query": "ghosts haunt a house"}


def test_recommend_keeps_only_searched_movies_once_each(toolbox):
    state = ToolState(genre=None)
    run(toolbox, state, "search_movies", query="ghosts")
    picks = [
        {"movie_id": 3, "why": "spooky"},
        {"movie_id": 404, "why": "invented"},
        {"movie_id": 3, "why": "again"},
    ]

    result = run(toolbox, state, "recommend", reply="Ось добірка.", picks=picks)

    assert result == {"status": "ok", "accepted": [3], "rejected": [404, 3]}
    assert state.answer.reply == "Ось добірка."
    assert [(r.movie.title, r.why) for r in state.answer.picks] == [("Ghost Ship", "spooky")]


def test_show_trailer_needs_a_searched_movie_with_a_trailer(toolbox):
    state = ToolState(genre=None)
    assert "error" in run(toolbox, state, "show_trailer", movie_id=3, reason="look")

    run(toolbox, state, "search_movies", query="ghosts")
    assert run(toolbox, state, "show_trailer", movie_id=2, reason="look") == {"status": "not_found"}
    assert state.trailer is None
    assert run(toolbox, state, "show_trailer", movie_id=3, reason="Неонове світло") == {
        "status": "shown"
    }
    assert (state.trailer.movie.id, state.trailer.youtube_key) == (3, "yt-3")
    assert state.trailer.reason == "Неонове світло"


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("search_movies", "not json"),
        ("search_movies", "{}"),
        ("recommend", '{"reply": "hi", "picks": [{"movie_id": "x"}]}'),
        ("delete_everything", "{}"),
    ],
)
def test_bad_calls_return_an_error_the_model_can_read(toolbox, name, arguments):
    state = ToolState(genre=None)
    result = json.loads(toolbox.run(ToolCall("c1", name, arguments), state))

    assert "error" in result
    assert state.answer is None
    assert state.steps[0].outcome.startswith("помилка")
