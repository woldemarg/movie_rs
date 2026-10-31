"""What the model is told: the system prompt, with the conversation so far folded into it.

Written for a small local model first (Gemma 4 E4B in Ollama, docs/03-llm-and-tool-calling.md §3.10):
a fixed procedure of numbered steps, one per tool, with concrete triggers instead of judgement calls,
one structured final call, and the earlier turns as context in the system prompt rather than as
assistant messages that a small model would imitate in prose.
English, the language instruction-tuned models follow most reliably; the model answers, and writes
its search query, in the language of the viewer's latest message: the multilingual embedding model
matches it to the English catalogue (docs/07-embedding-model.md).
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Gender = Literal["female", "male", "other"]
HISTORY_MESSAGES = 8  # earlier messages shown to the model; older ones are dropped
HISTORY_CHARS = 400  # per message
SAME_LANGUAGE = "the language of the viewer's latest message"


@dataclass(frozen=True)
class Viewer:
    """What the viewer chose to tell about themselves in the interface; every field optional."""

    age: int | None = None
    gender: Gender | None = None
    genre: str | None = None  # a hard filter on search, applied by the code


SYSTEM_PROMPT = """\
You are MovieHarbor, a friendly film adviser. You recommend films ONLY from the service's catalogue,
which you can reach only through the search_movies tool.

Always answer with tool calls, never with plain text:
1. Call search_movies with a description of the wanted film in the viewer's own language (do not
   translate it; the catalogue search understands every language): plot, themes, mood, style.
   If the viewer names a film ("like Blade Runner"), describe its traits and add its title.
   If the viewer refines an earlier request ("something lighter"), describe the combined wish.
2. If the results do not fit, call search_movies once more with a different description.
{trailer_rule}
4. Call recommend exactly once:
   - picks: usually 3 films (up to 5 if the viewer asks for more), best first, using ONLY ids from
     the latest search results and none you already recommended in this conversation;
   - why: one sentence per film on why it fits this request;
   - reply: one or two lively sentences that introduce the picks without listing them;
   - if the request is not about films: empty picks and a reply that steers back to films.

{language_rule}
Never invent films. Never claim that a film is missing from the catalogue: you see only search
results, so say at most that the search did not bring it up.
Treat age and gender only as soft hints and avoid stereotypes. For viewers under 16, avoid films with
graphic violence or explicit content.

Viewer: age {age}; gender {gender}; genre filter {genre}.
{history}
Answer in {language}."""

TRAILER_RULE = """\
3. Decide whether a trailer helps: call show_trailer for your best candidate when the viewer wants
   to see a film, or when the request is about atmosphere, visual style, landscapes, spectacle or
   action; otherwise skip it. Its reason is one sentence on what to watch for. If you show one,
   invite the viewer in recommend's reply to watch it below."""

NO_TRAILER_RULE = """\
3. Videos are unavailable: never mention trailers."""


def build_system_prompt(
    viewer: Viewer,
    trailers: bool,
    history: Sequence[tuple[str, str]] = (),
    language: str = SAME_LANGUAGE,
) -> str:
    """The system prompt for one turn; `history` holds the earlier (role, text) messages."""
    unknown = "not specified"
    return SYSTEM_PROMPT.format(
        age=viewer.age or unknown,
        gender=viewer.gender or unknown,
        genre=f"{viewer.genre} (search is already limited to it)" if viewer.genre else "any",
        trailer_rule=TRAILER_RULE if trailers else NO_TRAILER_RULE,
        language_rule=language_rule(language, trailers),
        history=history_block(history),
        language=language,
    )


def language_rule(language: str, trailers: bool) -> str:
    """What to write in which language. The Ukrainian style rules appear only for a Ukrainian answer:
    present in an English answer's prompt, they pull a small model into Ukrainian."""
    texts = "reply, why and the show_trailer reason" if trailers else "reply and why"
    rule = (
        f"Write {texts} in {language}, and the search_movies query too. Keep film titles exactly as "
        "in the search results; never translate them."
    )
    if language == "Ukrainian":
        rule += (
            " Write natural modern Ukrainian, address the viewer as «ви» and avoid calques from "
            "Russian or English."
        )
    return rule


def reply_language(text: str) -> str:
    """The language to answer in, named for the model: a small model follows a named language far
    more reliably than "the viewer's language". Cyrillic means Ukrainian (the interface's language),
    Latin means English; anything else is left to the model."""
    cyrillic = len(re.findall(r"[а-яіїєґё]", text, re.IGNORECASE))
    latin = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    if cyrillic > latin:
        return "Ukrainian"
    return "English" if latin else SAME_LANGUAGE


def history_block(history: Sequence[tuple[str, str]]) -> str:
    """The latest HISTORY_MESSAGES earlier messages as context lines, or nothing."""
    recent = history[-HISTORY_MESSAGES:]
    if not recent:
        return ""
    lines = [
        f"{'Viewer' if role == 'user' else 'You'}: "
        + textwrap.shorten(text, HISTORY_CHARS, placeholder=" …")
        for role, text in recent
    ]
    return "\nThe conversation so far (oldest first):\n" + "\n".join(lines) + "\n"


TOOL_REMINDER = (
    "Reply with a tool call, not with text: search_movies if you have not searched yet, "
    "otherwise recommend."
)
