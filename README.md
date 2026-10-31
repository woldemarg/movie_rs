# MovieHarbor

Діалоговий сервіс рекомендацій фільмів на основі LLM — proof of concept магістерської роботи.

## Запуск

Потрібні [uv](https://docs.astral.sh/uv/) і Python 3.12.

```bash
uv sync
cp .env.sample .env        # заповніть LLM_API_KEY (OpenRouter) і, за бажання, TMDB_API_KEY
uv run python -m backend   # API: http://127.0.0.1:8000/docs
```

Модель працює або локально (LM Studio, Ollama — будь-який OpenAI-сумісний сервер), або через OpenRouter (Gemma 4). Джерело обирається в кожному запиті. Змінні описано в `.env.sample` і в [розділі 3](docs/03-llm-and-tool-calling.md).

Прототип на Streamlit (`uv run streamlit run movieharbor.py`) працює, доки його не замінить новий вебінтерфейс.

## Дані

Каталог — 2 274 фільми з відкритого набору The Movies Dataset (CC0) з багатомовними ембедінгами для семантичного пошуку. Він уже зібраний у `data/`. Перебудувати його (≈ 3 хв на CPU) і виміряти якість пошуку:

```bash
uv run python -m scripts.build_catalog
uv run python -m scripts.evaluate_retrieval
```

## Документація

Специфікація проєкту — у [`docs/`](docs/):

- [2. Дані та ембедінги](docs/02-data-and-embeddings.md)
- [3. LLM і виклик інструментів](docs/03-llm-and-tool-calling.md)
- [6. Оцінювання якості пошуку](docs/06-evaluation.md)

## Перевірка якості

```bash
uv run python scripts/check.py
```

Скрипт запускає `ruff check`, `ruff format --check` і весь набір тестів pytest. Те саме виконує CI для кожного PR.

Правила роботи з кодом (для людей і AI-агентів) описані в [AGENTS.md](AGENTS.md).
