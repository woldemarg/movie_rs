# MovieHarbor

Діалоговий сервіс рекомендацій фільмів на основі LLM — proof of concept магістерської роботи.

## Запуск

Потрібні [uv](https://docs.astral.sh/uv/) і Python 3.12.

```bash
uv sync
uv run streamlit run movieharbor.py
```

Налаштування читаються з `.env` у корені проєкту.

## Перевірка якості

```bash
uv run python scripts/check.py
```

Скрипт запускає `ruff check`, `ruff format --check` і весь набір тестів pytest. Те саме виконує CI для кожного PR.

Правила роботи з кодом (для людей і AI-агентів) описані в [AGENTS.md](AGENTS.md).
