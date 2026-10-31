"""`python -m backend`: serve MovieHarbor on http://127.0.0.1:8000 (API docs at /docs)."""

import logging

import uvicorn

from backend.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
settings = Settings()
uvicorn.run("backend.api:create_app", factory=True, host=settings.host, port=settings.port)
