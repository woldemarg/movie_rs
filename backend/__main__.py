"""`python -m backend`: serve MovieHarbor on http://127.0.0.1:8000 (API docs at /docs)."""

import logging

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
uvicorn.run("backend.api:create_app", factory=True, host="127.0.0.1", port=8000)
