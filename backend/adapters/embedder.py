"""Sentence embeddings with fastembed (ONNX on CPU, no torch).

Spec: docs/02-data-and-embeddings.md.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from backend.core.catalog import Embed

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedder(model_name: str) -> Embed:
    """Load the model once (downloaded into models/ on first use) and return the embed function."""
    model = TextEmbedding(model_name, cache_dir=str(MODELS_DIR))

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = np.array(list(model.embed(list(texts))), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return embed
