import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunker import Chunk, build_chunks


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # camelCase → spaces
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)          # underscores/punct → spaces
    return [t for t in text.lower().split() if len(t) >= 2]


# Project-name tokens appear in (nearly) every question and every file path.
# "WinMerge" tokenizes to {"win","merge"}; "merge" collides with the Merge.* file
# stems and would grant them the full definition boost on every single query.
# Strip these before stem matching — they carry no discriminative signal.
_STEM_STOPWORDS = frozenset({"win", "merge", "winmerge"})


def _stem_bonus(file_path: str, query_tokens: set[str]) -> float:
    """Multiplicative score bonus when file stem tokens match query tokens.

    Rationale: BM25 rewards *usage* frequency (large .cpp files mention a class
    many times) over *definition* (the .h file has fewer occurrences).
    A stem match means this file is likely the definition, warranting a boost.

    Handles C++ class naming conventions:
    - MFC prefix: CDiffContext → query token "cdiff" ends with stem token "diff".
    - Interface prefix: IUnknown → query token "iunknown" ends with "unknown".
    """
    stem = Path(file_path).stem          # "DiffList" from "Src/DiffList.h"
    stem_tokens = set(_tokenize(stem)) - _STEM_STOPWORDS
    query_tokens = query_tokens - _STEM_STOPWORDS
    if not stem_tokens:
        return 1.0

    # A stem token matches a query token if equal, OR if the query token is the
    # stem token with a single-char C++ class prefix removed (C/I convention):
    #   "cdiff"  → "diff"   ✓ (prefix "c", 1 char)
    #   "winmerge" → "merge" ✗ (prefix "win", 3 chars — spurious; every question
    #                            mentions "WinMerge", so this must NOT boost Merge* files)
    def _matches(st: str) -> bool:
        if st in query_tokens:
            return True
        return any(qt.endswith(st) and len(qt) - len(st) == 1 for qt in query_tokens)

    matched = sum(1 for st in stem_tokens if _matches(st))
    coverage = matched / len(stem_tokens)
    if coverage == 1.0:
        return 2.5 + matched * 0.25   # full match: strong boost
    elif coverage >= 0.5:
        return 1.0 + coverage          # partial match: moderate boost
    return 1.0


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _select_results(
    chunks: list[Chunk],
    chunk_scores: list[tuple[int, float]],
    query_tokens: set[str],
    top_k: int,
    chunks_per_file: int,
) -> list[SearchResult]:
    """Shared post-ranking: per-file dedup (top-N chunks/file) + stem-match file boost.

    chunk_scores: (chunk_idx, fused_score) for candidate chunks, any order.
    Returns up to top_k files × chunks_per_file chunks, files ranked by
    best-chunk score × stem bonus.
    """
    file_chunks: dict[str, list[tuple[int, float]]] = {}
    for i, score in chunk_scores:
        fp = chunks[i].file_path
        file_chunks.setdefault(fp, []).append((i, score))

    for fp in file_chunks:
        file_chunks[fp].sort(key=lambda t: t[1], reverse=True)
        file_chunks[fp] = file_chunks[fp][:chunks_per_file]

    file_best: dict[str, float] = {
        fp: cs[0][1] * _stem_bonus(fp, query_tokens)
        for fp, cs in file_chunks.items()
    }
    top_files = sorted(file_best, key=file_best.__getitem__, reverse=True)[:top_k]

    results: list[SearchResult] = []
    for fp in top_files:
        bonus = _stem_bonus(fp, query_tokens)
        for idx, raw in file_chunks[fp]:
            results.append(SearchResult(chunk=chunks[idx], score=raw * bonus))
    return results


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Return (chunk_idx, bm25_score) for ALL chunks, sorted descending."""
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order]

    def search(self, query: str, top_k: int = 5,
               chunks_per_file: int = 2) -> list[SearchResult]:
        """BM25-only search with per-file dedup + stem boost."""
        return _select_results(
            self._chunks, self.rank(query), set(_tokenize(query)),
            top_k, chunks_per_file,
        )

    @property
    def size(self) -> int:
        return len(self._chunks)


def _rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    rrf_k: int = 60,
    candidate_pool: int = 200,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    score(chunk) = sum_r 1 / (rrf_k + rank_r(chunk))

    RRF is used (not score averaging) because BM25 scores and cosine similarities
    live on incomparable scales; rank position is the only common currency.
    Only the top `candidate_pool` of each ranking contribute (tail is noise).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking[:candidate_pool]):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda t: t[1], reverse=True)


class HybridIndex:
    """BM25 + dense retrieval fused with Reciprocal Rank Fusion.

    Mirrors the production system's BM25 + dense (FAISS, BGE) hybrid (spec §1):
    BM25 nails exact symbol names; dense embeddings bridge language gaps
    (RU question → EN code) and name gaps (DIFFRANGE → DiffList.h).
    """

    def __init__(self, chunks: list[Chunk], dense, bm25_min_score: float = 15.0) -> None:
        self._chunks = chunks
        self._bm25 = BM25Index(chunks)
        self._dense = dense
        # BM25 joins fusion only when its top score clears this floor — i.e. when it
        # has a real lexical match. Below it, BM25's ranking is corpus-frequency noise
        # (e.g. a Russian question whose only ASCII tokens are the project name), which
        # would otherwise dilute the dense signal under rank fusion. Empirically stable
        # across floor ∈ [12, 18] on the eval set; not a tuned magic constant.
        self._bm25_min_score = bm25_min_score

    def search(self, query: str, top_k: int = 5,
               chunks_per_file: int = 2) -> list[SearchResult]:
        bm25_rank = self._bm25.rank(query)
        dense_rank = self._dense.rank(query)

        bm25_top1 = bm25_rank[0][1] if bm25_rank else 0.0
        if bm25_top1 >= self._bm25_min_score:
            rankings = [bm25_rank, dense_rank]   # confident lexical match → true hybrid
        else:
            rankings = [dense_rank]              # BM25 is noise for this query → dense-only

        fused = _rrf_fuse(rankings)
        return _select_results(
            self._chunks, fused, set(_tokenize(query)),
            top_k, chunks_per_file,
        )

    @property
    def size(self) -> int:
        return len(self._chunks)


def build_index(repo_root: Path, settings=None):
    """Build retrieval index. Hybrid (BM25+dense) when settings.use_dense, else BM25-only."""
    chunks = build_chunks(repo_root)
    if settings is not None and getattr(settings, "use_dense", False):
        from .dense import DenseIndex
        dense = DenseIndex(
            chunks,
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        return HybridIndex(chunks, dense,
                           bm25_min_score=getattr(settings, "bm25_min_score", 15.0))
    return BM25Index(chunks)
