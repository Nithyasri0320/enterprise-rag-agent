"""
Ingestion pipeline: load raw documents -> chunk -> build both indexes
(Chroma vector store + BM25 keyword index) that the hybrid retriever
in retrieval.py reads from.

Run directly:  python -m src.ingestion
"""
import pickle
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src import config


def load_documents(data_dir: Path):
    """Load .txt/.md/.pdf files from data_dir. Each source file becomes one
    or more LangChain Document objects with `source` metadata attached —
    this metadata is what lets us show citations later."""
    docs = []

    txt_loader = DirectoryLoader(
        str(data_dir), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )
    docs.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        str(data_dir), glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )
    docs.extend(md_loader.load())

    for pdf_path in Path(data_dir).rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf_path)).load())

    if not docs:
        raise ValueError(
            f"No .txt, .md, or .pdf files found in {data_dir}. "
            "Add documents there before running ingestion."
        )
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Give every chunk a stable id — used to cross-reference between the
    # vector store and the BM25 index, and for citation display.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    # persist_directory alone is enough — langchain-chroma auto-persists to
    # disk on write, there's no separate .persist() call anymore.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    return vector_store


def build_bm25_index(chunks):
    """BM25 needs tokenized text. We keep it simple (lowercase + split) —
    good enough for keyword/acronym/ID matching, which is exactly the case
    vector search tends to miss."""
    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "texts": [c.page_content for c in chunks],
                "metadatas": [c.metadata for c in chunks],
            },
            f,
        )
    return bm25


def run_ingestion():
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = load_documents(config.DATA_DIR)
    print(f"Loaded {len(docs)} source document(s).")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print("Building vector index (Chroma)...")
    build_vector_store(chunks)

    print("Building keyword index (BM25)...")
    build_bm25_index(chunks)

    print(f"Done. Indexes persisted to {config.INDEX_DIR}")


if __name__ == "__main__":
    run_ingestion()
