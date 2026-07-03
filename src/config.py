"""
Central configuration. Every tunable knob lives here so the rest of the
codebase never reads os.environ directly — makes the architecture easy to
explain in an interview ("here's the one place that controls retrieval
behavior") and easy to tune without touching logic.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data/sample_docs")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "data/index")
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 10))  # candidates per retriever, pre-fusion
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))        # final chunks sent to the LLM
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", 2))

RRF_K = 60  # standard reciprocal-rank-fusion smoothing constant

GROUNDEDNESS_THRESHOLD = 0.5  # fraction of answer sentences that must trace back to context

INDEX_DIR.mkdir(parents=True, exist_ok=True)
