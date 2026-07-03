"""
Cross-encoder reranking.

Bi-encoders (used for the initial vector search) embed the query and each
chunk independently, then compare vectors — fast, but coarse, because the
model never actually sees the query and chunk together. A cross-encoder
takes the (query, chunk) pair as joint input and scores relevance directly,
which is far more accurate — too slow to run over a whole corpus, but
cheap once the candidate set is already narrowed to ~10 by hybrid search.
That two-stage pattern (fast recall, then precise rerank) is the standard
production shape.
"""
from sentence_transformers import CrossEncoder

from src import config

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int = None):
    """candidates: list of {"text": ..., "metadata": ..., "score": ...}
    Returns the same shape, re-scored and truncated to top_k."""
    if not candidates:
        return []

    top_k = top_k or config.TOP_K_RERANK
    model = _get_model()

    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
