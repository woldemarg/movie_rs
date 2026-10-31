# AGENTS.md — working rules for coding agents

MovieHarbor is a master-thesis proof of concept: an LLM-powered movie recommender (semantic retrieval over a movie catalogue + a chat model that picks and explains). It lives in a public repository. The bar is a clean, honest, reproducible PoC that is easy to explain in the thesis, not a production service. Prefer the simplest design that is correct and tested.

`python` below means the project venv created by `uv sync`: `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere.

## Rule 0 — the gate passes before you report done

```text
python scripts/check.py
```

The gate is `ruff check` + `ruff format --check` + the whole pytest suite. Every change to Python, `pyproject.toml` or `uv.lock` ends with it passing; CI runs the same script.

- Do not silence it (`# noqa`, new ruff ignores, skipped or loosened tests). Fix the cause.
- If a stage cannot run, say which one and why. Never imply it passed.
- Docs-only changes may skip it.

## Layout

| Path | Role |
|---|---|
| `movieharbor.py`, `config.py`, `prompts.py`, `styles.css` | the Streamlit prototype (being replaced) |
| `movies.pkl`, `movies_metadata.csv` | the prototype's catalogue and its raw source |
| `Chapter 7 - Building recommendation systems with LLMs.ipynb` | the notebook the project started from; reference only |
| `tests/` | pytest suite |
| `scripts/check.py` | the gate (Rule 0) |
| `pyproject.toml`, `uv.lock`, `.python-version` | dependencies (exact versions in the lock), tool config, Python 3.12 |

## Boundaries

- **Secrets.** `.env` holds the user's keys. Never print, log, copy or commit its values; edit it only when asked. `.env.sample` is the template.
- **Paid and external calls.** The LLM provider and TMDB accounts belong to the user. Tests never touch the network: the LLM and TMDB are local stubs. Make live calls only when the user asks.
- **Data.** Only the public dataset ships in the repository. Its source and licence are documented next to it.
- **Dependencies.** Add one only when the standard library or an installed package cannot do the job in a few lines; add it to `pyproject.toml`, then `uv lock`.

## Standards

- Python 3.12, type hints, `X | None`, dataclasses or pydantic models for any shape that crosses a module boundary.
- Parse outside data once, where it enters (HTTP request, LLM reply, CSV row), then trust it inside. Keep business logic in pure functions.
- One module, one job. No dead code, speculative flags, wrappers around a single call, or section banners. Prefer deleting a mechanism to adding a flag.
- User-facing errors are clear Ukrainian messages. They never carry secrets, so never echo raw HTTP-client exceptions (their URLs can hold keys).
- Language: English for code, comments and docstrings. Ukrainian for the UI, the prompts, `README.md`, `docs/` and PR descriptions.

## Testing

- Test behaviour through the public entry point (function, HTTP endpoint, page), not internals. A test that would still pass if the code returned nothing is not a test.
- A bug fix starts with a test that fails for the reason you expect.
- Order: the targeted test, then the gate.

## Docs

- `README.md` and `docs/` describe the current state in the present tense. History belongs in commits and PRs.
- A change to behaviour updates the owning doc in the same change.

## Workflow

- Small, reversible changes; root causes, not symptoms. Ask one focused question before a risky change; otherwise decide and state the decision.
- One concern per PR. Branch `<type>/<topic>`. PR description in Ukrainian: what changed, why, how it was verified (real command output).
- Concise reports: failures with their output, skipped steps named.
