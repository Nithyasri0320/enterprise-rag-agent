"""
Hybrid retrieval: combine dense (vector/semantic) and sparse (BM25/keyword)
search results via Reciprocal Rank Fusion (RRF).

Why hybrid: vector search is great at "meaning" but blurs exact terms,
IDs, acronyms, and rare vocabulary. BM25 is the opposite. Fusing both is
the production baseline most enterprises actually run in 2026 — full
Graph/Agentic retrieval is reserved for cases that genuinely need
multi-hop reasoning.
"""
import pickle

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config


class HybridRetriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embeddings,
        )

        with open(config.BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_texts = bm25_data["texts"]
        self.bm25_metadatas = bm25_data["metadatas"]

    def _vector_search(self, query: str, k: int):
        results = self.vector_store.similarity_search(query, k=k)
        return [(r.page_content, r.metadata) for r in results]

    def _bm25_search(self, query: str, k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.bm25_texts[i], self.bm25_metadatas[i]) for i in ranked_idx]

    def search(self, query: str, k: int = None):
        """Returns a fused, deduplicated list of {text, metadata, score}
        dicts ranked by Reciprocal Rank Fusion across both retrievers."""
        k = k or config.TOP_K_RETRIEVE

        vector_results = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)

        rrf_scores = {}
        doc_lookup = {}

        for rank, (text, meta) in enumerate(vector_results):
            key = text
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (config.RRF_K + rank + 1)
            doc_lookup[key] = meta

        for rank, (text, meta) in enumerate(bm25_results):
            key = text
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (config.RRF_K + rank + 1)
            doc_lookup[key] = meta

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {"text": text, "metadata": doc_lookup[text], "score": score}
            for text, score in fused[:k]
        ]
