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


def test_specs_are_plain_json_schema_for_every_available_tool(toolbox):
    names = [spec["function"]["name"] for spec in toolbox.specs]
    assert names == ["search_movies", "show_trailer", "recommend"]
    text = json.dumps(toolbox.specs)
    assert "$ref" not in text and "$defs" not in text and "anyOf" not in text and "null" not in text
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
    shown = run(toolbox, state, "show_trailer", movie_id=3, reason="Неонове світло")
    assert shown == {"status": "shown"}
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


def test_a_tool_that_is_not_offered_cannot_be_called(movies):
    toolbox = Toolbox(Catalog.build(movies, fake_embed, "fake"), fake_embed, find_trailer=None)
    state = ToolState(genre=None)
    run(toolbox, state, "search_movies", query="ghosts")

    result = run(toolbox, state, "show_trailer", movie_id=3, reason="look")

    assert result == {"error": "unknown tool 'show_trailer'"}
    assert state.trailer is None


def test_recommend_with_only_unsearched_ids_is_an_error_and_keeps_the_turn_open(toolbox):
    state = ToolState(genre=None)

    result = run(
        toolbox, state, "recommend", reply="Ось.", picks=[{"movie_id": 2, "why": "моторошно"}]
    )

    assert result["rejected"] == [2] and "error" in result
    assert state.answer is None


def test_recommend_with_no_picks_ends_the_turn(toolbox):
    state = ToolState(genre=None)

    assert (
        run(toolbox, state, "recommend", reply="Я раджу лише фільми.", picks=[])["status"] == "ok"
    )
    assert state.answer.picks == ()


def test_a_new_search_replaces_the_grounding_set(toolbox, monkeypatch):
    monkeypatch.setattr("backend.core.tools.SEARCH_K", 1)
    state = ToolState(genre=None)
    run(toolbox, state, "search_movies", query="ghosts haunt an old family house")
    assert set(state.candidates) == {2}

    run(toolbox, state, "search_movies", query="strangers fall in love in paris")

    assert set(state.candidates) == {4}
    result = run(
        toolbox, state, "recommend", reply="Ось.", picks=[{"movie_id": 2, "why": "моторошно"}]
    )
    assert result["rejected"] == [2]
    assert state.answer is None


def test_the_largest_valid_recommend_fits_the_default_token_budget():
    from backend.core.tools import MAX_PICKS, Pick, RecommendArgs
    from backend.settings import Settings

    reply_max = RecommendArgs.model_fields["reply"].metadata[-1].max_length
    why_max = Pick.model_fields["why"].metadata[-1].max_length
    largest = RecommendArgs(
        reply="ж" * reply_max,
        picks=[Pick(movie_id=123456, why="ж" * why_max) for _ in range(MAX_PICKS)],
    ).model_dump_json()

    # Cyrillic averages above 2 characters per token; 2 keeps the bound conservative.
    assert len(largest) / 2 <= Settings.model_fields["llm_max_tokens"].default


def test_steps_carry_the_ranked_candidates_and_the_choice_for_the_trace(toolbox):
    state = ToolState(genre=None)
    run(toolbox, state, "search_movies", query="ghosts haunt a house")
    run(toolbox, state, "recommend", reply="Ось.", picks=[{"movie_id": 3, "why": "атмосфера"}])

    ranked = state.steps[0].data["candidates"]
    assert [c["title"] for c in ranked][:2] == ["The Haunting", "Ghost Ship"]
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert state.steps[1].data == {"accepted": [3], "rejected": []}
