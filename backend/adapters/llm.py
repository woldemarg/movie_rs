"""OpenAI-compatible chat-completions client: one protocol for a local server and for OpenRouter.

LM Studio, Ollama, llama.cpp and OpenRouter all accept the same `POST {base_url}/chat/completions`
with `tools`, so switching between a local model and the external API is only a different
profile (URL, model, key). Spec: docs/03-llm-and-tool-calling.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

from backend.core.agent import ToolChoice
from backend.core.tools import AssistantTurn, ToolCall

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 5.0  # a server that does not accept the connection by then is not running


class LLMError(Exception):
    """A failure to show the user as is: Ukrainian text that never carries a secret."""


@dataclass(frozen=True)
class LLMProfile:
    provider: str  # "local" | "openrouter"
    base_url: str
    model: str
    api_key: str = ""
    extra_body: dict[str, object] = field(default_factory=dict)  # e.g. OpenRouter routing


@dataclass(frozen=True)
class LLMClient:
    profile: LLMProfile
    temperature: float
    max_tokens: int
    timeout_s: float
    transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport

    def complete(
        self, messages: list[dict], tools: list[dict], tool_choice: ToolChoice = "auto"
    ) -> AssistantTurn:
        """One chat-completion round trip: the model either answers or asks for tool calls."""
        body = {
            "model": self.profile.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.profile.extra_body,
        }
        headers = (
            {"Authorization": f"Bearer {self.profile.api_key}"} if self.profile.api_key else {}
        )
        url = f"{self.profile.base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(self.timeout_s, connect=CONNECT_TIMEOUT_S)
        try:
            with httpx.Client(transport=self.transport, timeout=timeout) as http:
                response = http.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            if isinstance(exc, httpx.ConnectTimeout):  # nobody listening: same as refused
                raise self.unreachable(exc) from None
            raise LLMError(
                f"Модель не відповіла за {self.timeout_s:.0f} с. Спробуйте ще раз."
            ) from None
        except httpx.TransportError as exc:
            raise self.unreachable(exc) from None
        if response.status_code >= 400:
            raise LLMError(describe_http_error(response, self.profile))
        return parse_turn(response)

    def unreachable(self, exc: httpx.TransportError) -> LLMError:
        log.warning("LLM server %s unreachable: %s", self.profile.base_url, type(exc).__name__)
        return LLMError(
            f"Сервер моделі недоступний ({self.profile.base_url}). "
            "Перевірте, що він запущений, або оберіть інше джерело моделі."
        )


def describe_http_error(response: httpx.Response, profile: LLMProfile) -> str:
    """A Ukrainian explanation of an HTTP error from the model server."""
    known = {
        401: "Провайдер відхилив ключ API. Перевірте LLM_API_KEY у файлі .env.",
        402: "На рахунку провайдера недостатньо коштів.",
        403: "Провайдер відхилив ключ API. Перевірте LLM_API_KEY у файлі .env.",
        404: f"Модель «{profile.model}» не знайдено на сервері. Перевірте назву моделі.",
        429: "Перевищено ліміт запитів до моделі. Спробуйте за хвилину.",
    }
    if response.status_code in known:
        return known[response.status_code]
    detail = provider_message(response)
    return f"Сервер моделі повернув помилку {response.status_code}" + (
        f": {detail}" if detail else "."
    )


def provider_message(response: httpx.Response) -> str:
    """The server's own error text (OpenAI shape `{"error": {"message": ...}}`), shortened."""
    try:
        error = response.json().get("error")
    except ValueError:
        return ""
    message = error.get("message", "") if isinstance(error, dict) else str(error or "")
    return message[:200]


def parse_turn(response: httpx.Response) -> AssistantTurn:
    """Parse the assistant message: its text and its tool calls (arguments stay raw JSON text)."""
    try:
        data = response.json()
        if data.get("error"):
            raise LLMError(f"Сервер моделі повернув помилку: {provider_message(response)}")
        message = data["choices"][0]["message"]
        calls = tuple(
            ToolCall(
                id=call.get("id") or f"call_{i}",
                name=call["function"]["name"],
                arguments=arguments_text(call["function"].get("arguments")),
            )
            for i, call in enumerate(message.get("tool_calls") or [])
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise LLMError("Сервер моделі повернув відповідь у неочікуваному форматі.") from None
    return AssistantTurn(content=message.get("content") or "", tool_calls=calls)


def arguments_text(arguments: object) -> str:
    """Tool arguments as JSON text: the spec says a string, some local servers send an object."""
    if arguments is None or arguments == "":
        return "{}"
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
