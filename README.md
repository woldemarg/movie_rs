# MovieHarbor

Діалоговий сервіс рекомендацій фільмів на основі LLM — proof of concept магістерської роботи.

## Запуск

Потрібні [uv](https://docs.astral.sh/uv/) і Python 3.12.

```bash
uv sync
cp .env.sample .env        # заповніть LLM_API_KEY (OpenRouter) і, за бажання, TMDB_API_KEY
uv run python -m backend   # http://127.0.0.1:8000 — вебінтерфейс, /docs — API
```

Модель працює або локально (LM Studio, Ollama — будь-який OpenAI-сумісний сервер), або через OpenRouter (Gemma 4). Джерело обирається перемикачем у шапці вебінтерфейсу. Змінні описано в `.env.sample` і в [розділі 3](docs/03-llm-and-tool-calling.md).

## Дані

Каталог — 2 274 фільми з відкритого набору The Movies Dataset (CC0) з багатомовними ембедінгами для семантичного пошуку. Він уже зібраний у `data/`. Перебудувати його (≈ 3 хв на CPU) і виміряти якість пошуку:

```bash
uv run python -m scripts.build_catalog
uv run python -m scripts.refresh_posters     # потрібен TMDB_API_KEY
uv run python -m scripts.evaluate_retrieval
```

## Документація

Специфікація проєкту — у [`docs/`](docs/):

- [2. Дані та ембедінги](docs/02-data-and-embeddings.md)
- [3. LLM і виклик інструментів](docs/03-llm-and-tool-calling.md)
- [4. Інтерфейс](docs/04-interface.md)
- [5. Наскрізний сценарій](docs/05-e2e-workflow.md)
- [6. Оцінювання якості пошуку](docs/06-evaluation.md)

## Перевірка якості

```bash
uv run python scripts/check.py
```

Скрипт запускає `ruff check`, `ruff format --check` і весь набір тестів pytest, зокрема E2E-сценарії в браузері (потрібен `uv run playwright install chromium`). Те саме виконує CI для кожного PR.

Правила роботи з кодом (для людей і AI-агентів) описані в [AGENTS.md](AGENTS.md).

Метадані, постери й трейлери надає [TMDB](https://www.themoviedb.org/). Цей продукт використовує TMDB API, але не схвалений і не сертифікований TMDB.
