# Enterprise RAG Agent

An agentic Retrieval-Augmented Generation system built to demonstrate **enterprise-grade RAG architecture** — not just "embed and retrieve," but the full pattern enterprises actually ship in 2026: hybrid retrieval, reranking, adaptive/agentic control flow, groundedness guardrails, and evaluation.

Built for interview prep on Agentic AI Engineer roles — every component below maps directly to a common interview question.

---

## Architecture

```
                     ┌─────────────┐
   question ───────► │   Router    │  "does this even need retrieval?"
                     └──────┬──────┘
                            │
                 needs_retrieval? ──No──► direct_answer ──► END
                            │Yes
                     ┌──────▼──────┐
                     │  Retrieve   │  Hybrid search: vector (Chroma) + BM25 (keyword)
                     │  (Hybrid)   │  merged via Reciprocal Rank Fusion
                     └──────┬──────┘
                     ┌──────▼──────┐
                     │   Rerank    │  Cross-encoder reranks top-N → top-k
                     └──────┬──────┘
                     ┌──────▼──────┐
                     │Grade Docs   │  LLM grades each chunk: relevant or not
                     └──────┬──────┘
                irrelevant? │ (and retries left)
              ┌─────Yes─────┴─────No──────┐
      ┌───────▼──────┐             ┌──────▼──────┐
      │Rewrite Query │             │  Generate    │  Answer + inline citations
      └───────┬──────┘             └──────┬──────┘
              │ (loop back to Retrieve)    │
              └─────────────────────►┌─────▼──────┐
                                      │Check Ground │  Are the answer's claims
                                      │  -edness    │  actually in the context?
                                      └──────┬──────┘
                              not grounded?  │ (and retries left)
                          loop back to Rewrite / END with answer
```

This is implemented as a **LangGraph** state machine in `src/agent.py` — not a linear script. That distinction ("RAG pipeline vs. agentic loop") is one of the most common interview questions right now: a plain RAG system embeds → retrieves → generates in one fixed pass; an agent decides whether/what to retrieve, can retry, and evaluates its own output.

---

## Why each piece exists (interview cheat sheet)

| Component | File | Interview question it answers |
|---|---|---|
| Hybrid retrieval (vector + BM25, RRF fusion) | `src/retrieval.py` | "Why not just use vector search?" — keyword search catches exact terms/IDs/acronyms that embeddings blur. |
| Cross-encoder reranking | `src/reranker.py` | "How do you improve precision after retrieval?" — bi-encoders are fast but coarse; a cross-encoder re-scores the actual (query, chunk) pair. |
| Router node | `src/agent.py: route_query` | "Does every query need retrieval?" — no, adaptive retrieval avoids wasted latency/cost on greetings or general questions. |
| Document grading + query rewrite loop | `src/agent.py` | "What happens when retrieval returns garbage?" — self-correction: grade relevance, rewrite the query, retry, instead of hallucinating an answer from noise. |
| Groundedness guardrail | `src/guardrails.py` | "How do you know it's not hallucinating?" — check that generated claims are traceable to retrieved chunks before returning the answer. |
| Evaluation harness (RAGAS) | `eval/evaluate.py` | "How do you measure RAG quality?" — faithfulness, answer relevancy, context precision, not vibes. |

---

## Stack

- **Orchestration:** LangGraph (agentic control flow) + LangChain (loaders, splitters)
- **LLM:** Groq (Llama 3.3 70B via `langchain-groq`) — fast + generous free tier
- **Vector store:** ChromaDB (local, persisted to disk)
- **Keyword search:** `rank_bm25`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free)
- **Eval:** RAGAS
- **UI:** Streamlit (shows the agent's reasoning trace — great for demos/interviews)

Everything runs on free tiers. No paid infra required.

---

## Setup

```bash
# 1. Clone and enter
cd enterprise-rag-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# edit .env and paste your key from https://console.groq.com/keys

# 5. Ingest the sample documents (builds the vector store + BM25 index)
python -m src.ingestion

# 6. Run the app
streamlit run app.py
```

To use your own documents, drop `.txt`, `.md`, or `.pdf` files into `data/sample_docs/` (or point `DATA_DIR` in `.env` elsewhere) and re-run ingestion.

---

## Running the evaluation harness

```bash
python -m eval.evaluate
```

This runs the eval questions in `eval/eval_dataset.json` through the agent and scores faithfulness, answer relevancy, and context precision with RAGAS.

---

## Deployment note

This won't deploy cleanly to Vercel — Chroma, sentence-transformers, and the cross-encoder are too heavy for serverless functions with cold-start limits. For a portfolio demo, use one of:

- **Streamlit Community Cloud** (free, easiest — connects directly to your GitHub repo)
- **Hugging Face Spaces** (free, good if you want to show ML-heavy projects specifically)

Record a short demo GIF/video of the reasoning trace in the UI for your LinkedIn post — the visible agent loop (router → retrieve → grade → generate → groundedness check) is the part that actually differentiates this from a basic PDF bot, so make sure it's visible on screen.

---

## What to say about this project in an interview

"I built an agentic RAG system rather than a naive retrieve-and-generate pipeline. It routes queries adaptively, combines vector and keyword search with reciprocal rank fusion, reranks with a cross-encoder, grades retrieved documents for relevance before generating, and checks the final answer's groundedness against the retrieved context — rewriting the query and retrying when retrieval or grounding fails. I evaluate it with RAGAS on faithfulness and context precision rather than eyeballing outputs."

That's a 20-second answer that hits retrieval architecture, agentic design, and evaluation rigor in one breath.
