"""Live local HTTP servers for the API and end-to-end tests (no network beyond 127.0.0.1).

`stub_app` plays the external services: an OpenAI-compatible model that uses the tools the way
a real one should, and TMDB's videos endpoint. `ServerThread` runs any ASGI app on a free port.
"""

from __future__ import annotations

import json
import threading
import time

import uvicorn
from fastapi import FastAPI, Request


def stub_app() -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        app.state.requests.append({"body": body, "auth": request.headers.get("authorization")})
        return {"choices": [{"message": scripted_reply(body["messages"])}]}

    @app.get("/3/movie/{movie_id}/videos")
    def videos(movie_id: int) -> dict:
        return {"results": [{"site": "YouTube", "type": "Trailer", "key": f"stub{movie_id}"}]}

    return app


def scripted_reply(messages: list[dict]) -> dict:
    """Search with the user's words, show a trailer when asked for one, recommend the top three."""
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    user = messages[last_user]["content"]
    since = messages[last_user + 1 :]
    results = [json.loads(m["content"]) for m in since if m["role"] == "tool"]
    found = next((r["movies"] for r in reversed(results) if "movies" in r), None)
    called = {c["function"]["name"] for m in since for c in m.get("tool_calls") or []}
    if found is None:
        return tool_reply("search_movies", query=user)
    if "трейлер" in user.lower() and found and "show_trailer" not in called:
        return tool_reply(
            "show_trailer", movie_id=found[0]["id"], reason="Зверніть увагу на світло."
        )
    picks = [{"movie_id": m["id"], "why": f"Пасує до запиту: {user}"} for m in found[:3]]
    return tool_reply("recommend", reply="Ось що я раджу.", picks=picks)


def tool_reply(name: str, **arguments: object) -> dict:
    call = {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


class ServerThread:
    """Run an ASGI app with uvicorn in a background thread; `url` is set once it listens."""

    def __init__(self, app: object) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = ""

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("test server did not start")
            time.sleep(0.02)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
