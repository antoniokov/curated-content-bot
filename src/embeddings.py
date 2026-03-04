"""Embedding model management (lazy-loaded singleton)."""

import logging

# Suppress "UNEXPECTED key" warnings from transformers model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (first time may download ~80MB)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded.")
    return _embed_model
