import pytest

from backend.core.agent import MAX_LLM_CALLS, AgentError, Message, run_agent
from backend.core.catalog import Catalog
from backend.core.prompts import Viewer
from backend.core.tools import Toolbox
from tests.doubles import FakeTrailers, ScriptedChat, assistant, fake_embed, tool_call

ASK = [Message("user", "Хочу щось страшне про привидів")]


@pytest.fixture
def toolbox(movies) -> Toolbox:
    return Toolbox(Catalog.build(movies, fake_embed, "fake"), fake_embed, FakeTrailers(3).find)


def search_then_recommend(*picks: dict, reply: str = "Тримайте добірку.") -> ScriptedChat:
    return ScriptedChat(
        assistant(tool_call("search_movies", query="ghosts haunt a house")),
        assistant(tool_call("recommend", reply=reply, picks=list(picks))),
    )


def test_agent_searches_then_returns_grounded_picks(toolbox):
    chat = search_then_recommend({"movie_id": 2, "why": "класика жанру"})

    result = run_agent(chat, toolbox, ASK, Viewer(age=30, gender="female", genre="Horror"))

    assert result.reply == "Тримайте добірку."
    assert [(p.movie.title, p.why) for p in result.picks] == [("The Haunting", "класика жанру")]
    assert result.trailer is None
    assert [step.tool for step in result.steps] == ["search_movies", "recommend"]
    assert result.llm_calls == 2
    first_messages, tools = chat.requests[0]
    assert first_messages[0]["role"] == "system"
    system = first_messages[0]["content"]
    assert "30" in system and "Horror" in system
    assert "language of the viewer's latest message" in system
    assert first_messages[1:] == [{"role": "user", "content": ASK[0].content}]
    assert [t["function"]["name"] for t in tools] == ["search_movies", "show_trailer", "recommend"]
    second_messages, _ = chat.requests[1]
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "search_movies"
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call_search_movies"


def test_agent_shows_a_trailer_when_the_model_asks_for_one(toolbox):
    chat = ScriptedChat(
        assistant(tool_call("search_movies", query="ghost spaceship")),
        assistant(
            tool_call("show_trailer", movie_id=3, reason="Подивіться на темні коридори."),
            tool_call("recommend", reply="Ось.", picks=[{"movie_id": 3, "why": "атмосфера"}]),
        ),
    )

    result = run_agent(chat, toolbox, ASK, Viewer())

    assert result.trailer.youtube_key == "yt-3"
    assert result.trailer.reason == "Подивіться на темні коридори."
    assert [p.movie.id for p in result.picks] == [3]


def test_prose_makes_the_next_call_require_a_tool_without_a_fake_user_turn(toolbox):
    chat = ScriptedChat(
        assistant(tool_call("search_movies", query="ghosts")),
        assistant(content="Раджу The Haunting!"),
        assistant(
            tool_call("recommend", reply="Раджу.", picks=[{"movie_id": 2, "why": "страшно"}])
        ),
    )

    result = run_agent(chat, toolbox, ASK, Viewer())

    assert [p.movie.id for p in result.picks] == [2]
    assert chat.tool_choices == ["auto", "auto", "required"]
    sent = chat.requests[2][0]
    assert [m["role"] for m in sent].count("user") == 1
    assert sent[-1] == {"role": "assistant", "content": "Раджу The Haunting!"}


def test_a_turn_never_ends_in_prose(toolbox):
    chat = ScriptedChat(assistant(content="Привіт!"), assistant(content="Чим допомогти?"))

    with pytest.raises(AgentError, match="інструмент"):
        run_agent(chat, toolbox, ASK, Viewer())


def test_rejected_picks_give_the_model_another_chance(toolbox):
    chat = ScriptedChat(
        assistant(tool_call("search_movies", query="ghosts haunt a house")),
        assistant(
            tool_call("recommend", reply="Ось.", picks=[{"movie_id": 404, "why": "вигадка"}])
        ),
        assistant(
            tool_call("recommend", reply="Ось.", picks=[{"movie_id": 2, "why": "моторошно"}])
        ),
    )

    result = run_agent(chat, toolbox, ASK, Viewer())

    assert [p.movie.id for p in result.picks] == [2]
    assert result.llm_calls == 3
    assert "error" in chat.requests[2][0][-1]["content"]


def test_agent_gives_up_after_the_call_budget(toolbox):
    loop = [assistant(tool_call("search_movies", query="ghosts")) for _ in range(MAX_LLM_CALLS)]
    with pytest.raises(AgentError, match=str(MAX_LLM_CALLS)):
        run_agent(ScriptedChat(*loop), toolbox, ASK, Viewer())


def test_agent_sends_recent_history_starting_with_a_user_turn(toolbox):
    history = [Message("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(15)]
    chat = search_then_recommend()

    run_agent(chat, toolbox, history, Viewer())

    sent = chat.requests[0][0][1:]
    assert sent[0] == {"role": "user", "content": "m6"}
    assert sent[-1] == {"role": "user", "content": "m14"}
    assert len(sent) <= 10


def test_history_entry_remembers_the_picks_for_the_next_turn(toolbox):
    chat = search_then_recommend({"movie_id": 2, "why": "атмосферно"}, reply="Ось.")

    result = run_agent(chat, toolbox, ASK, Viewer())

    assert result.history_entry.startswith("Ось.")
    assert "The Haunting (2002, id 2)" in result.history_entry
