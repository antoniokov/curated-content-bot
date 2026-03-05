"""Embedding model management (lazy-loaded singleton with auto-unload)."""

import gc
import logging
import time

# Suppress "UNEXPECTED key" warnings from transformers model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_embed_model = None
_last_used = 0
MODEL_IDLE_TIMEOUT = 300  # 5 minutes


def get_embed_model():
    global _embed_model, _last_used
    _last_used = time.time()
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        try:
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except OSError:
            logger.info("Model not cached locally, downloading (~80MB)...")
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded.")
    return _embed_model


def maybe_unload_model():
    """Unload the model if it hasn't been used for MODEL_IDLE_TIMEOUT seconds."""
    global _embed_model
    if _embed_model is not None and time.time() - _last_used > MODEL_IDLE_TIMEOUT:
        _embed_model = None
        gc.collect()
        logger.info("Embedding model unloaded (idle > %ds).", MODEL_IDLE_TIMEOUT)


def update_embed_model():
    """Re-download the embedding model from HuggingFace and reload it."""
    global _embed_model, _last_used
    from sentence_transformers import SentenceTransformer
    logger.info("Updating embedding model from HuggingFace...")
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    _last_used = time.time()
    logger.info("Model updated.")
    return _embed_model
