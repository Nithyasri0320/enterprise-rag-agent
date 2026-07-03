"""
Streamlit demo UI. Deliberately shows the agent's reasoning trace
(router -> retrieve -> rerank -> grade -> generate -> groundedness check)
so a viewer — or an interviewer — can see it's an agentic loop, not a
black-box call. Run with: streamlit run app.py
"""
import streamlit as st

from src import config, agent

st.set_page_config(page_title="Enterprise RAG Agent", page_icon="🔎", layout="wide")

st.title("🔎 Enterprise RAG Agent")
st.caption(
    "Agentic RAG: adaptive routing, hybrid retrieval (vector + BM25), "
    "cross-encoder reranking, document grading, groundedness checking."
)

if not config.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key from "
        "https://console.groq.com/keys, then restart the app."
    )
    st.stop()

if not config.BM25_PATH.exists():
    with st.spinner("First run: building the search index from data/sample_docs (vector store + BM25)... this takes a minute."):
        from src import ingestion
        ingestion.run_ingestion()
    st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the ingested documents...")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            st.caption("Sources:\n" + turn["sources"])
        with st.expander("Agent reasoning trace"):
            for step in turn["trace"]:
                st.text(step)
            badge = "✅ Grounded" if turn["grounded"] else "⚠️ Not fully grounded"
            st.caption(f"{badge} (score: {turn['groundedness_score']})")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Running agent..."):
            result = agent.run(question)

        st.write(result["answer"])
        if result["sources"]:
            st.caption("Sources:\n" + result["sources"])

        with st.expander("Agent reasoning trace", expanded=True):
            for step in result["trace"]:
                st.text(step)
            badge = "✅ Grounded" if result["grounded"] else "⚠️ Not fully grounded"
            st.caption(f"{badge} (score: {result['groundedness_score']})")

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "trace": result["trace"],
            "grounded": result["grounded"],
            "groundedness_score": result["groundedness_score"],
        }
    )

with st.sidebar:
    st.subheader("Architecture")
    st.markdown(
        "1. **Route** — does this need retrieval?\n"
        "2. **Retrieve** — hybrid vector + BM25 search\n"
        "3. **Rerank** — cross-encoder re-scores candidates\n"
        "4. **Grade** — LLM filters irrelevant chunks\n"
        "5. **Rewrite & retry** — if nothing relevant was found\n"
        "6. **Generate** — answer with inline citations\n"
        "7. **Check groundedness** — verify claims trace to context"
    )
    st.divider()
    st.caption(f"Model: {config.GROQ_MODEL}")
    st.caption(f"Embedding: {config.EMBEDDING_MODEL.split('/')[-1]}")
    st.caption(f"Reranker: {config.RERANKER_MODEL.split('/')[-1]}")
