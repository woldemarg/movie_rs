# Специфікація MovieHarbor

Документи описують поточний стан коду й оновлюються разом із ним (правило в [AGENTS.md](../AGENTS.md#docs)). Почніть з огляду: він описує систему за моделлю C4 і веде до решти розділів.

1. [Огляд системи (C4)](desc/01-overview.md)
2. [Дані та ембедінги](desc/02-data-and-embeddings.md)
3. [LLM і виклик інструментів](desc/03-llm-and-tool-calling.md)
4. [Інтерфейс: вебсторінка й HTTP API](desc/04-interface.md)
5. [Наскрізний сценарій: модулі й зовнішні сервіси](desc/05-e2e-workflow.md)
6. [Оцінювання якості пошуку](desc/06-evaluation.md)
7. [Вибір моделі ембедінгів](desc/07-embedding-model.md)
8. [Ризики галюцинацій LLM](desc/08-hallucinations.md)
9. [Вплив параметрів користувача на вибір](desc/09-selection-parameters.md)
10. [Стратегія тестування](desc/10-testing.md)
11. [Аналогічні системи та порівняння](desc/11-related-systems.md)
12. [Вимоги до системи](desc/12-requirements.md)

Діаграми — Mermaid. GitHub рендерить їх у переглядачі. Для тексту роботи їх можна експортувати у SVG ([Mermaid Live Editor](https://mermaid.live) або `mmdc`). Знімки екрана — у [`images/`](images/).
