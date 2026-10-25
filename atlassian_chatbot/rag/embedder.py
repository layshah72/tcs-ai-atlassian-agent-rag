from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding

# Suppress HuggingFace Hub unauthenticated warning if no token is configured.
if not os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str) -> Any:
    """Load and reuse a FastEmbed model, caching downloaded weights in app data."""
    cache_dir = os.getenv("FASTEMBED_CACHE_DIR")
    if not cache_dir:
        app_data = (
            os.getenv("LOCALAPPDATA")
            or os.getenv("XDG_DATA_HOME")
            or str(Path.home() / ".cache")
        )
        cache_dir = str(Path(app_data) / "TCS-Confluent-chatbot" / "models" / "fastembed")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedding_model(model_name: str | None = None) -> Any:
    """Get the process-cached FastEmbed model, defaulting to FASTEMBED_MODEL env var."""
    return load_embedding_model(model_name or os.getenv("FASTEMBED_MODEL") or DEFAULT_EMBED_MODEL)
