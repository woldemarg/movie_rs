# MovieHarbor backend: the HTTP API, the agent, the embedding model and the catalogue.
# Built from the repository root (compose.yaml: context ., dockerfile backend/Dockerfile).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0

RUN useradd --create-home --uid 1000 app && mkdir /app && chown app /app
USER app
WORKDIR /app

# Dependencies first (cached until the lock changes); the project itself is not a package.
COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# The embedding model comes from the project cache (models/embeddings, filled once on the host by
# `python -m backend.adapters.embedder`); the RUN only downloads it if that cache was empty. The
# container then starts in seconds and needs no Hugging Face access (docs/desc/01-overview.md §1.8).
COPY --chown=app backend backend
COPY --chown=app models/embeddings models/embeddings
RUN python -m backend.adapters.embedder
ENV HF_HUB_OFFLINE=1

# data/ is mounted from the project (compose.yaml): the first start builds the catalogue from
# data/raw into it, once; later starts find it there (scripts/prepare_data.py does nothing then).
COPY --chown=app scripts scripts

EXPOSE 8000
CMD ["sh", "-c", "python -m scripts.prepare_data && exec python -m backend"]
