"""Sentence embeddings with fastembed (ONNX on CPU, no torch).

Spec: docs/desc/02-data-and-embeddings.md.
"""

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from backend.core.catalog import Embed

# The project-local cache (gitignored): downloaded once, then reused by every run and copied
# into the Docker image (Dockerfile).
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedder(model_name: str) -> Embed:
    """Load the model (downloaded into models/embeddings/ on first use); return the embed function."""
    model = TextEmbedding(model_name, cache_dir=str(MODELS_DIR))
    lock = threading.Lock()  # requests run in a thread pool; one ONNX run at a time

    def embed(texts: Sequence[str]) -> np.ndarray:
        with lock:
            vectors = np.array(list(model.embed(list(texts))), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return embed


if __name__ == "__main__":  # python -m backend.adapters.embedder: fill the cache once
    load_embedder(DEFAULT_MODEL)
    print(f"{DEFAULT_MODEL} is cached in {MODELS_DIR}")
