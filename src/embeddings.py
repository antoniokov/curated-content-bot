"""Embedding model management (lazy-loaded singleton with auto-unload)."""

import gc
import logging
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_embed_model = None
_last_used = 0
MODEL_IDLE_TIMEOUT = 300  # 5 minutes
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class ONNXEmbedder:
    """Lightweight ONNX-based text embedder (replaces SentenceTransformer)."""

    def __init__(self, model_id, local_files_only=False):
        model_path = hf_hub_download(
            model_id, "onnx/model.onnx", local_files_only=local_files_only
        )
        tokenizer_path = hf_hub_download(
            model_id, "tokenizer.json", local_files_only=local_files_only
        )
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in self._input_names

    def encode(self, texts, batch_size=64):
        """Encode texts into embeddings (matches SentenceTransformer.encode interface)."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )

            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self._needs_token_type_ids:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            token_embeddings = self.session.run(None, feeds)[0]

            # Mean pooling with attention mask
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)

            # L2 normalize (matches SentenceTransformer's Normalize module)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1
            pooled = pooled / norms

            all_embeddings.append(pooled)

        return np.concatenate(all_embeddings, axis=0)


def get_embed_model():
    global _embed_model, _last_used
    _last_used = time.time()
    if _embed_model is None:
        logger.info("Loading embedding model...")
        try:
            _embed_model = ONNXEmbedder(MODEL_ID, local_files_only=True)
        except OSError:
            logger.info("Model not cached locally, downloading (~25MB)...")
            _embed_model = ONNXEmbedder(MODEL_ID)
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
    logger.info("Updating embedding model from HuggingFace...")
    _embed_model = ONNXEmbedder(MODEL_ID)
    _last_used = time.time()
    logger.info("Model updated.")
    return _embed_model
