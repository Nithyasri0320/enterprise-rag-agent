"""
Basic tests for the pieces that don't need a Groq API key — ingestion,
chunking, and hybrid retrieval. Run: pytest tests/

Note: reranker and agent tests need real model downloads / API calls and
are intentionally left out of this offline test file. Test those manually
via `streamlit run app.py` once you have a GROQ_API_KEY set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingestion, config


def test_load_documents():
    docs = ingestion.load_documents(config.DATA_DIR)
    assert len(docs) >= 3, "Expected at least 3 sample documents"


def test_chunking():
    docs = ingestion.load_documents(config.DATA_DIR)
    chunks = ingestion.chunk_documents(docs)
    assert len(chunks) > len(docs), "Chunking should produce more chunks than source docs"
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) <= config.CHUNK_SIZE + 200  # some slack for separators


def test_bm25_index_build_and_query():
    docs = ingestion.load_documents(config.DATA_DIR)
    chunks = ingestion.chunk_documents(docs)
    bm25 = ingestion.build_bm25_index(chunks)

    tokenized_query = "refund window days".lower().split()
    scores = bm25.get_scores(tokenized_query)
    assert len(scores) == len(chunks)
    assert max(scores) > 0, "Expected at least one chunk to match 'refund'"


if __name__ == "__main__":
    test_load_documents()
    test_chunking()
    test_bm25_index_build_and_query()
    print("All offline tests passed.")
