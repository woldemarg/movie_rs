"""End to end: Chromium -> web UI -> API -> agent -> (stub) LLM and TMDB -> rendered answer.

The backend is the real app on a local port; only the external services are stubbed, and the
page may not reach anything but 127.0.0.1. Spec: docs/05-e2e-workflow.md.
"""

import re

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, expect, sync_playwright

from backend.api import create_app
from tests.conftest import stub_settings
from tests.doubles import fake_embed
from tests.servers import ServerThread

LOCAL_ONLY = re.compile(r"^(?!https?://127\.0\.0\.1[:/])")


@pytest.fixture(scope="module")
def app_url(stub):
    settings = stub_settings(stub, llm_api_key="sk-or-e2e")
    with ServerThread(create_app(settings, embed=fake_embed)) as server:
        yield server.url


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except PlaywrightError as exc:  # reported as a skip, never as a pass
            pytest.skip(f"Chromium for Playwright is missing (playwright install chromium): {exc}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser, app_url) -> Page:
    page = browser.new_page()
    page.route(LOCAL_ONLY, lambda route: route.abort())  # posters, thumbnails: no network in tests
    page.goto(app_url)
    expect(page.locator("#catalog-info")).to_contain_text("фільмів у каталозі")
    yield page
    page.close()


def send(page: Page, text: str) -> None:
    page.fill("#question", text)
    page.press("#question", "Enter")


def last_request(stub) -> dict:
    return stub.app.state.requests[-1]


def test_full_conversation_with_cards_trace_follow_up_and_trailer(page, stub):
    source = page.locator("#source-switch")
    expect(source.locator('[data-provider="local"]')).to_have_attribute("aria-checked", "true")
    page.select_option("#genre", "Horror")

    send(page, "ghosts haunt an old house")

    expect(page.locator(".msg.user").last).to_have_text("ghosts haunt an old house")
    cards = page.locator(".msg.assistant").last.locator(".card")
    expect(cards).to_have_count(3)
    for i in range(3):
        expect(cards.nth(i).locator(".tag", has_text="Жахи")).to_have_count(1)
        expect(cards.nth(i).locator(".why")).to_contain_text("Пасує до запиту")
    expect(cards.first.locator(".poster-fallback")).to_be_visible()  # the blocked poster falls back
    trace = page.locator("#trace-panel .trace").last  # the side panel, one entry per turn
    expect(trace).to_be_visible()
    expect(trace).to_contain_text("stub-local")
    expect(trace.locator(".stage")).to_have_count(3)  # query, candidates, choice
    expect(trace).to_contain_text("«ghosts haunt an old house»")
    expect(trace.locator(".candidates li")).to_have_count(12)  # the search's shortlist
    expect(trace.locator(".candidates .chosen")).to_have_count(3)  # ✓ the model's picks
    expect(trace).to_contain_text("Обрано 3 з 12 · місця за схожістю: 1, 2, 3")
    expect(trace).not_to_contain_text("undefined")
    expect(trace).not_to_contain_text("false")
    assert last_request(stub)["auth"] is None

    send(page, "А покажи трейлер")

    trailer = page.locator(".trailer")
    expect(trailer).to_contain_text("Зверніть увагу на світло.")
    expect(page.locator("#trace-panel .trace")).to_have_count(1)  # refreshed, not appended
    expect(page.locator("#trace-panel .trace")).to_contain_text("show_trailer")
    system = last_request(stub)["body"]["messages"][0]["content"]
    assert "[recommended:" in system, "the follow-up must carry the previous answer"
    trailer.locator(".video-thumb").click()
    expect(trailer.locator("iframe")).to_have_attribute(
        "src", re.compile(r"youtube-nocookie\.com/embed/stub\d+")
    )


def test_openrouter_switch_sends_the_key_and_the_choice_survives_a_reload(page, stub):
    page.locator('[data-provider="openrouter"]').click()
    page.select_option("#genre", "Comedy")

    send(page, "щось смішне")

    expect(page.locator(".card")).to_have_count(3)
    assert last_request(stub)["auth"] == "Bearer sk-or-e2e"
    assert last_request(stub)["body"]["model"] == "stub-remote"
    page.reload()
    expect(page.locator('[data-provider="openrouter"]')).to_have_attribute("aria-checked", "true")
    expect(page.locator("#genre")).to_have_value("Comedy")


def test_a_model_error_shows_a_message_and_retry_recovers(page, stub):
    stub.app.state.fail_next = True

    send(page, "space adventure")

    error = page.locator(".msg.error")
    expect(error).to_contain_text("Сервер моделі повернув помилку 500: stub overloaded")
    error.get_by_role("button", name="Спробувати ще раз").click()
    expect(page.locator(".msg.error")).to_have_count(0)
    expect(page.locator(".card")).to_have_count(3)
    expect(page.locator(".msg.user")).to_have_count(1)


def test_a_suggestion_starts_the_chat_and_new_chat_resets_it(page):
    expect(page.locator("#welcome")).to_be_visible()

    page.locator(".chip").first.click()

    expect(page.locator(".msg.user")).to_have_count(1)
    expect(page.locator(".card")).to_have_count(3)
    expect(page.locator("#welcome")).to_be_hidden()
    expect(page.locator("#trace-panel .trace")).to_have_count(1)
    page.get_by_role("button", name="Нова розмова").click()
    expect(page.locator("#welcome")).to_be_visible()
    expect(page.locator(".msg")).to_have_count(0)
    expect(page.locator("#trace-panel .trace")).to_have_count(0)


def test_a_backend_still_starting_disables_asking_until_it_answers(browser, app_url):
    page = browser.new_page()
    page.route(LOCAL_ONLY, lambda route: route.abort())
    page.route("**/api/config", lambda route: route.fulfill(status=502, body="Bad Gateway"))

    page.goto(app_url)

    expect(page.locator(".msg.error")).to_contain_text("ще запускається")
    expect(page.locator("#send")).to_be_disabled()
    page.unroute("**/api/config")  # the backend comes up
    expect(page.locator(".msg.error")).to_have_count(0, timeout=10_000)
    expect(page.locator("#send")).to_be_enabled()
    expect(page.locator("#welcome")).to_be_visible()
    page.close()


def test_an_unexpected_answer_body_is_reported_as_such_not_as_a_lost_connection(page):
    malformed = '{"type": "answer", "reply": "x"}\n'  # no picks, message, trace
    page.route(
        "**/api/recommend/stream",
        lambda route: route.fulfill(
            status=200, body=malformed, content_type="application/x-ndjson"
        ),
    )

    send(page, "space adventure")

    expect(page.locator(".msg.error")).to_contain_text("неочікуваному форматі")
    expect(page.locator(".msg.error")).not_to_contain_text("Немає зв'язку")


def test_a_local_model_still_downloading_disables_asking_until_it_is_ready(browser, app_url):
    def local_not_ready(route) -> None:
        response = route.fetch()
        config = response.json()
        config["providers"] = [{**p, "ready": False} for p in config["providers"]]
        route.fulfill(response=response, json=config)

    page = browser.new_page()
    page.route(LOCAL_ONLY, lambda route: route.abort())
    page.route("**/api/config", local_not_ready)

    page.goto(app_url)

    expect(page.locator(".msg.error")).to_contain_text("ще завантажується")
    expect(page.locator("#send")).to_be_disabled()
    page.unroute("**/api/config")  # the download has finished
    expect(page.locator(".msg.error")).to_have_count(0, timeout=10_000)
    expect(page.locator("#send")).to_be_enabled()
    expect(page.locator('[data-provider="local"]')).to_have_attribute("aria-checked", "true")
    page.close()
