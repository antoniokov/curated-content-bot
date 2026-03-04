"""Embedding model management (lazy-loaded singleton)."""

import logging
import sys

# Suppress "UNEXPECTED key" warnings from transformers model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (first time may download ~80MB)...", file=sys.stderr)
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Model loaded.", file=sys.stderr)
    return _embed_model
