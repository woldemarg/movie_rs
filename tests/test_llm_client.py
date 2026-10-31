import json

import httpx
import pytest

from backend.adapters.llm import LLMClient, LLMError, LLMProfile

SECRET = "sk-or-secret-123"
TOOLS = [{"type": "function", "function": {"name": "search_movies", "parameters": {}}}]
MESSAGES = [{"role": "user", "content": "Привіт"}]


def client(handler, api_key: str = "", extra_body: dict | None = None) -> LLMClient:
    profile = LLMProfile(
        "openrouter", "https://llm.test/api/v1/", "google/gemma", api_key, extra_body or {}
    )
    return LLMClient(profile, 0.7, 500, 5.0, transport=httpx.MockTransport(handler))


def reply(message: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": message}]})


def test_sends_an_openai_tools_request_and_parses_tool_calls():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "search_movies", "arguments": '{"query": "x"}'},
        }
        return reply({"role": "assistant", "content": None, "tool_calls": [call]})

    order = {"provider": {"order": ["a", "b"]}}
    turn = client(handler, api_key=SECRET, extra_body=order).complete(MESSAGES, TOOLS)

    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://llm.test/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "google/gemma"
    assert body["messages"] == MESSAGES and body["tools"] == TOOLS
    assert body["tool_choice"] == "auto"
    assert (body["temperature"], body["max_tokens"]) == (0.7, 500)
    assert body["provider"] == {"order": ["a", "b"]}
    assert turn.content == ""
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("c1", "search_movies", '{"query": "x"}')
    ]
    assert turn.as_message()["tool_calls"][0]["function"]["name"] == "search_movies"


def test_local_server_gets_no_authorization_header_and_plain_answers_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return reply({"role": "assistant", "content": "Привіт!"})

    turn = client(handler).complete(MESSAGES, TOOLS)

    assert (turn.content, turn.tool_calls) == ("Привіт!", ())


@pytest.mark.parametrize(
    "response, expected",
    [
        (httpx.Response(401, json={"error": {"message": "bad key"}}), "LLM_API_KEY"),
        (httpx.Response(402, json={}), "коштів"),
        (httpx.Response(404, text="nope"), "google/gemma"),
        (httpx.Response(429, json={}), "ліміт"),
        (
            httpx.Response(400, json={"error": {"message": "tools are not supported"}}),
            "tools are not supported",
        ),
        (
            httpx.Response(200, json={"error": {"message": "upstream overloaded"}}),
            "upstream overloaded",
        ),
        (httpx.Response(200, text="<html>"), "неочікуваному форматі"),
        (httpx.Response(200, json={"choices": []}), "неочікуваному форматі"),
    ],
)
def test_server_errors_become_ukrainian_messages_without_the_key(response, expected):
    with pytest.raises(LLMError) as error:
        client(lambda request: response, api_key=SECRET).complete(MESSAGES, TOOLS)

    assert expected in str(error.value)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    "exception, expected",
    [
        (httpx.ConnectError("refused"), "недоступний (https://llm.test/api/v1/)"),
        (httpx.ConnectTimeout("no answer to SYN"), "недоступний (https://llm.test/api/v1/)"),
        (httpx.ReadTimeout("slow"), "5 с"),
    ],
)
def test_network_failures_become_ukrainian_messages(exception, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    with pytest.raises(LLMError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        client(handler, api_key=SECRET).complete(MESSAGES, TOOLS)
