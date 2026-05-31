"""
Dense retrieval over code chunks using a multilingual sentence embedder.

Why dense in addition to BM25:
- BM25 is lexical: it cannot match a Russian question ("какой класс сравнивает файлы")
  to English code, nor a symbol name (DIFFRANGE) to its declaring file (DiffList.h)
  when the tokens don't literally overlap.
- A multilingual embedder maps question and code into a shared semantic space,
  closing both gaps. This mirrors the production system's BM25 + dense (FAISS, BGE) hybrid.

Embeddings are computed once and cached to disk (keyed by model + corpus hash) so that
repeated runs skip the expensive encode step.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from .chunker import Chunk

log = logging.getLogger(__name__)

# e5-family models require "query:" / "passage:" prefixes for asymmetric search.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _corpus_hash(chunks: list[Chunk], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(str(len(chunks)).encode())
    # Hash file paths + line ranges (cheap, stable proxy for corpus identity)
    for c in chunks[:: max(1, len(chunks) // 200)]:  # sample for speed
        h.update(f"{c.file_path}:{c.start_line}:{c.end_line}".encode())
    return h.hexdigest()[:16]


class DenseIndex:
    """Cosine-similarity dense index over chunk embeddings (numpy, no FAISS dependency)."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        self._chunks = chunks
        self._model_name = model_name
        self._device = device
        self._is_e5 = _is_e5(model_name)

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        # Code chunks are short (~40 lines); cap sequence length to avoid the
        # model's large default (e.g. bge-m3 = 8192), which wastes compute.
        self._model.max_seq_length = min(getattr(self._model, "max_seq_length", 512) or 512, 512)
        if device.startswith("cuda"):
            self._model = self._model.half()

        self._embeddings = self._load_or_compute(cache_dir)

    def _load_or_compute(self, cache_dir: Path | None) -> np.ndarray:
        cache_dir = cache_dir or Path("./reports/.emb_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _corpus_hash(self._chunks, self._model_name)
        cache_file = cache_dir / f"emb_{key}.npy"

        if cache_file.exists():
            log.info("Loading cached embeddings: %s", cache_file)
            emb = np.load(cache_file)
            if emb.shape[0] == len(self._chunks):
                return emb
            log.warning("Cache size mismatch (%d != %d); recomputing",
                        emb.shape[0], len(self._chunks))

        log.info("Encoding %d chunks with %s on %s (one-time; cached afterwards)...",
                 len(self._chunks), self._model_name, self._device)
        texts = [
            (_E5_PASSAGE_PREFIX + c.text) if self._is_e5 else c.text
            for c in self._chunks
        ]
        total = len(texts)
        batch_size = 64
        batches: list[np.ndarray] = []
        done = 0
        for start in range(0, total, batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_emb = self._model.encode(
                batch_texts, batch_size=64,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            batches.append(batch_emb)
            done += len(batch_texts)
            log.info("embedded %d/%d chunks", done, total)
        emb = np.vstack(batches).astype(np.float32)
        np.save(cache_file, emb)
        log.info("Saved embeddings cache: %s", cache_file)
        return emb

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_score) for ALL chunks, sorted descending."""
        q = (_E5_QUERY_PREFIX + query) if self._is_e5 else query
        q_emb = self._model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)[0]
        # embeddings are L2-normalized → dot product == cosine similarity
        scores = self._embeddings @ q_emb
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]
