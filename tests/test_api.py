import json

import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from tests.conftest import stub_settings
from tests.doubles import fake_embed

SECRET = "sk-or-secret-123"


def make_client(stub, **overrides) -> TestClient:
    return TestClient(create_app(stub_settings(stub, **overrides), embed=fake_embed))


def ask(client: TestClient, text: str, provider: str = "local", **viewer) -> dict:
    body = {"provider": provider, "messages": [{"role": "user", "content": text}], "viewer": viewer}
    return client.post("/api/recommend", json=body)


def test_config_lists_both_sources_and_never_exposes_keys(stub):
    response = make_client(stub, llm_api_key=SECRET).get("/api/config")

    assert response.status_code == 200
    config = response.json()
    assert [(p["id"], p["model"], p["configured"], p["ready"]) for p in config["providers"]] == [
        ("local", "stub-local", True, True),
        ("openrouter", "stub-remote", True, True),
    ]
    assert config["movies"] > 2000 and "Horror" in config["genres"]
    assert config["trailers"] is True
    assert SECRET not in response.text and "tmdb-test" not in response.text

    unconfigured = make_client(stub, tmdb_api_key="").get("/api/config").json()
    assert unconfigured["providers"][1]["configured"] is False
    assert unconfigured["providers"][1]["ready"] is False
    assert unconfigured["trailers"] is False


def test_the_local_model_is_ready_only_once_its_server_lists_it(stub):
    pulling = make_client(stub, local_llm_model="gemma4:e4b").get("/api/config").json()
    local = pulling["providers"][0]
    assert (local["configured"], local["ready"]) == (True, False)

    stub.app.state.models.append("gemma4:e4b")  # ollama-init finished the download
    try:
        ready = make_client(stub, local_llm_model="gemma4:e4b").get("/api/config").json()
    finally:
        stub.app.state.models.remove("gemma4:e4b")
    assert ready["providers"][0]["ready"] is True


def test_recommend_runs_the_agent_through_the_selected_model(stub):
    response = ask(make_client(stub), "ghosts in a haunted house", genre="Horror", age=30)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Ось що я раджу."
    assert len(data["picks"]) == 3
    assert all("Horror" in card["genres"] and card["why"] for card in data["picks"])
    assert data["picks"][0]["poster_url"].startswith("https://image.tmdb.org/")
    assert data["trailer"] is None
    assert data["message"]["role"] == "assistant"
    assert data["picks"][0]["title"] in data["message"]["content"]
    trace = data["trace"]
    assert (trace["provider"], trace["model"], trace["llm_calls"]) == ("local", "stub-local", 2)
    assert [step["tool"] for step in trace["steps"]] == ["search_movies", "recommend"]
    assert stub.app.state.requests[-1]["auth"] is None


def test_openrouter_requests_carry_the_key_and_trailers_come_from_tmdb(stub):
    response = ask(
        make_client(stub, llm_api_key=SECRET), "покажи трейлер космічного фільму", "openrouter"
    )

    data = response.json()
    assert data["trailer"]["youtube_key"] == f"stub{data['trailer']['movie_id']}"
    assert data["trailer"]["reason"] == "Зверніть увагу на світло."
    assert data["trace"]["model"] == "stub-remote"
    assert stub.app.state.requests[-1]["auth"] == f"Bearer {SECRET}"
    assert SECRET not in response.text


def test_unconfigured_openrouter_is_a_clear_400(stub):
    response = ask(make_client(stub), "щось смішне", "openrouter")

    assert response.status_code == 400
    assert "LLM_API_KEY" in response.json()["detail"]


def test_an_unreachable_model_server_is_a_502_with_a_ukrainian_message(stub):
    client = make_client(stub, local_llm_base_url="http://127.0.0.1:9/v1", llm_timeout_s=2)

    response = ask(client, "щось смішне")

    assert response.status_code == 502
    assert "недоступний" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "local", "messages": []},
        {"provider": "local", "messages": [{"role": "assistant", "content": "hi"}]},
        {"provider": "cloud", "messages": [{"role": "user", "content": "hi"}]},
        {
            "provider": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "viewer": {"age": 0},
        },
        {
            "provider": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "viewer": {"genre": "Opera"},
        },
    ],
)
def test_invalid_requests_are_rejected_before_any_model_call(stub, body):
    before = len(stub.app.state.requests)

    response = make_client(stub).post("/api/recommend", json=body)

    assert response.status_code == 422
    assert len(stub.app.state.requests) == before


def test_the_web_ui_is_served_at_the_root(stub):
    client = make_client(stub)

    page = client.get("/")

    assert page.status_code == 200
    assert '<script src="app.js"' in page.text
    assert client.get("/app.js").status_code == 200


def test_validation_errors_carry_one_ukrainian_message_naming_the_field(stub):
    body = {
        "provider": "local",
        "messages": [{"role": "user", "content": "hi"}],
        "viewer": {"age": 0},
    }

    detail = make_client(stub).post("/api/recommend", json=body).json()["detail"]

    assert isinstance(detail, str)
    assert detail.startswith("Некоректний запит") and "viewer.age" in detail


def test_the_stream_reports_each_step_then_the_answer(stub):
    body = {"provider": "local", "messages": [{"role": "user", "content": "ghosts in a house"}]}

    with make_client(stub).stream("POST", "/api/recommend/stream", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert [e["type"] for e in events] == ["step", "step", "answer"]
    assert events[0]["tool"] == "search_movies" and events[0]["arguments"]["query"]
    assert len(events[-1]["picks"]) == 3 and events[-1]["message"]["role"] == "assistant"


def test_the_stream_validates_before_streaming_and_reports_model_failures_as_an_event(stub):
    client = make_client(stub, local_llm_base_url="http://127.0.0.1:9/v1")
    bad = {"provider": "local", "messages": []}
    assert client.post("/api/recommend/stream", json=bad).status_code == 422

    body = {"provider": "local", "messages": [{"role": "user", "content": "щось смішне"}]}
    with client.stream("POST", "/api/recommend/stream", json=body) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert events[-1]["type"] == "error" and "недоступний" in events[-1]["detail"]
