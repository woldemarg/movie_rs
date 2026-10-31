"""The prompt's guardrails are requirements (docs/08-hallucinations.md): none may disappear."""

from backend.core.prompts import Viewer, build_system_prompt

GUARDRAILS = (
    "Always answer with tool calls, never with plain text",
    "ONLY ids from",  # grounding, enforced again in code
    "Never invent films",
    "Never claim that a film is missing from the catalogue",  # no claims beyond what it saw
    "do not\n   translate it",  # the multilingual embedder reads the viewer's language
    "never translate them",  # translated titles were a live hallucination
    "avoid stereotypes",
    "under 16",
)


def test_every_guardrail_is_in_the_prompt_with_and_without_trailers():
    for trailers in (True, False):
        prompt = build_system_prompt(Viewer(), trailers=trailers)
        assert [rule for rule in GUARDRAILS if rule not in prompt] == []


def test_the_ukrainian_style_rules_appear_only_for_a_ukrainian_answer():
    ukrainian = build_system_prompt(Viewer(), trailers=True, language="Ukrainian")
    english = build_system_prompt(Viewer(), trailers=True, language="English")

    assert "address the viewer as «ви»" in ukrainian
    assert "«ви»" not in english and "Ukrainian" not in english
    assert english.rstrip().endswith("Answer in English.")


def test_the_trailer_step_matches_whether_trailers_are_available():
    with_trailers = build_system_prompt(Viewer(), trailers=True)
    without = build_system_prompt(Viewer(), trailers=False)

    assert "3. Decide whether a trailer helps: call show_trailer" in with_trailers
    assert "show_trailer" not in without and "never mention trailers" in without


def test_the_conversation_so_far_appears_only_when_there_is_one():
    assert "conversation so far" not in build_system_prompt(Viewer(), trailers=True)
    prompt = build_system_prompt(
        Viewer(), trailers=True, history=[("user", "щось страшне"), ("assistant", "Ось.")]
    )
    assert "Viewer: щось страшне\nYou: Ось." in prompt


def test_the_reply_language_is_named_from_the_script_of_the_viewers_message():
    from backend.core.prompts import reply_language

    assert reply_language("Порадь страшний фільм") == "Ukrainian"
    assert reply_language("A whimsical romance like Amélie") == "English"
    assert reply_language("Щось як «Blade Runner», але легше") == "Ukrainian"
    assert reply_language("1984?") == "the language of the viewer's latest message"

    prompt = build_system_prompt(Viewer(), trailers=True, language="English")
    assert "Write reply, why and the show_trailer reason in English" in prompt
