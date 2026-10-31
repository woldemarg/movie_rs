"""The recommendation agent: an LLM in a loop with the tools of core/tools.py.

One user turn = up to MAX_LLM_CALLS chat completions. Each completion requests tool calls
(executed, their results appended, loop continues). A turn ends in exactly one way: a `recommend`
that the tools accept. Prose instead of tool calls makes the next completion require a tool; prose
again is an error. Spec: docs/03-llm-and-tool-calling.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from backend.core.prompts import Viewer, build_system_prompt
from backend.core.tools import (
    AssistantTurn,
    Recommendation,
    Step,
    Toolbox,
    ToolState,
    Trailer,
)

MAX_LLM_CALLS = 6  # search, a second search, trailer, recommend, and room for one correction
HISTORY_MESSAGES = 10  # recent conversation sent to the model; older turns are dropped


ToolChoice = Literal["auto", "required"]


class Chat(Protocol):
    def complete(
        self, messages: list[dict], tools: list[dict], tool_choice: ToolChoice = "auto"
    ) -> AssistantTurn: ...


class AgentError(Exception):
    """The model did not finish the turn; Ukrainian text for the user."""


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class AgentResult:
    reply: str
    picks: tuple[Recommendation, ...]
    trailer: Trailer | None
    steps: tuple[Step, ...]
    llm_calls: int

    @property
    def history_entry(self) -> str:
        """How this answer is remembered in the next request: the reply plus the picks' ids."""
        if not self.picks:
            return self.reply
        films = "; ".join(f"{p.movie.title} ({p.movie.year}, id {p.movie.id})" for p in self.picks)
        return f"{self.reply}\n\n[recommended: {films}]"


def run_agent(
    chat: Chat, toolbox: Toolbox, conversation: Sequence[Message], viewer: Viewer
) -> AgentResult:
    """Answer the conversation's last user message with grounded recommendations."""
    state = ToolState(genre=viewer.genre)
    system = build_system_prompt(viewer, trailers=toolbox.find_trailer is not None)
    messages = [{"role": "system", "content": system}, *recent(conversation)]
    tool_choice: ToolChoice = "auto"
    for calls in range(1, MAX_LLM_CALLS + 1):
        turn = chat.complete(messages, toolbox.specs, tool_choice)
        messages.append(turn.as_message())
        if not turn.tool_calls:
            if tool_choice == "required":
                raise AgentError(
                    "Модель відповіла текстом замість виклику інструментів. "
                    "Спробуйте ще раз або оберіть модель, що підтримує інструменти."
                )
            tool_choice = "required"  # the turn may end only through recommend
            continue
        for call in turn.tool_calls:
            result = toolbox.run(call, state)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if state.answer is not None:
            return AgentResult(
                state.answer.reply, state.answer.picks, state.trailer, tuple(state.steps), calls
            )
    raise AgentError(
        f"Модель не завершила відповідь за {MAX_LLM_CALLS} звернень. Спробуйте переформулювати запит."
    )


def recent(conversation: Sequence[Message]) -> list[dict[str, str]]:
    """The last HISTORY_MESSAGES messages, starting with a user turn as chat templates require."""
    window = list(conversation[-HISTORY_MESSAGES:])
    while window and window[0].role != "user":
        window.pop(0)
    return [{"role": m.role, "content": m.content} for m in window]
