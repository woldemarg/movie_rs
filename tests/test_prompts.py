"""The prompt's guardrails are requirements (docs/08-hallucinations.md): none may disappear."""

from backend.core.prompts import Viewer, build_system_prompt

GUARDRAILS = (
    "ONLY from the search results",  # grounding, enforced again in code
    "Never invent films or change ids",
    "Never claim that a film is missing from the catalogue",  # no claims beyond what it saw
    "language of the viewer's latest message",
    "avoid stereotypes",
    "under 16",
)


def test_every_guardrail_is_in_the_prompt_with_and_without_trailers():
    for trailers in (True, False):
        prompt = build_system_prompt(Viewer(), trailers=trailers)
        assert [rule for rule in GUARDRAILS if rule not in prompt] == []


def test_the_trailer_rule_matches_what_the_toolbox_offers():
    with_trailers = build_system_prompt(Viewer(), trailers=True)
    without = build_system_prompt(Viewer(), trailers=False)

    assert "Call show_trailer only when" in with_trailers
    assert "show_trailer" not in without
    assert "do not mention trailers" in without
