"""
The agentic control loop, built with LangGraph.

This is the piece that separates this project from a basic RAG pipeline:
the LLM sits *inside* the loop and can decide whether to retrieve, rewrite
a bad query and retry, and evaluate its own output before returning it —
rather than a single fixed embed -> retrieve -> generate pass.

Graph shape (see README for the diagram):

    route -> [retrieve -> rerank -> grade] -> generate -> check_grounding -> END
                 ^                     |                        |
                 └────── rewrite <─────┴────────────────────────┘
    route -> direct_answer -> END   (when retrieval isn't needed at all)
"""
from typing import TypedDict, List, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src import config, retrieval, reranker, guardrails

_llm = None
_retriever = None


def get_llm():
    global _llm
    if _llm is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://console.groq.com/keys"
            )
        _llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = retrieval.HybridRetriever()
    return _retriever


class AgentState(TypedDict):
    question: str
    original_question: str
    needs_retrieval: bool
    retries: int
    candidates: List[dict]
    graded: List[dict]
    answer: str
    grounded: bool
    groundedness_score: float
    sources: str
    trace: List[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> AgentState:
    question = state["question"]

    # cheap heuristic first — no LLM call needed for obvious small talk
    greeting_words = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
    if question.strip().lower().rstrip("!.") in greeting_words:
        state["needs_retrieval"] = False
        state["trace"] = state.get("trace", []) + ["route: heuristic -> no retrieval (greeting)"]
        return state

    llm = get_llm()
    prompt = (
        "Does answering the following question require looking up specific "
        "documents/knowledge, or can it be answered with general conversation? "
        "Reply with exactly one word: RETRIEVE or DIRECT.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt).content.strip().upper()
    needs_retrieval = "RETRIEVE" in response

    state["needs_retrieval"] = needs_retrieval
    state["trace"] = state.get("trace", []) + [f"route: LLM -> {'retrieve' if needs_retrieval else 'direct answer'}"]
    return state


def retrieve(state: AgentState) -> AgentState:
    retriever = get_retriever()
    candidates = retriever.search(state["question"])
    state["candidates"] = candidates
    state["trace"] = state["trace"] + [f"retrieve: hybrid search -> {len(candidates)} candidates"]
    return state


def rerank_node(state: AgentState) -> AgentState:
    reranked = reranker.rerank(state["question"], state["candidates"])
    state["candidates"] = reranked
    state["trace"] = state["trace"] + [f"rerank: cross-encoder -> top {len(reranked)}"]
    return state


def grade_documents(state: AgentState) -> AgentState:
    """Single LLM call: for each candidate chunk, is it actually relevant
    to the question? Filters out noise before it ever reaches generation."""
    llm = get_llm()
    candidates = state["candidates"]

    if not candidates:
        state["graded"] = []
        state["trace"] = state["trace"] + ["grade: no candidates to grade"]
        return state

    numbered = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Below are numbered passages. List ONLY the numbers of passages that "
        f"contain information relevant to answering the question, as a comma-"
        f"separated list (e.g. '0,2,3'). If none are relevant, reply 'none'.\n\n"
        f"{numbered}"
    )
    response = llm.invoke(prompt).content.strip().lower()

    if response == "none" or not any(ch.isdigit() for ch in response):
        graded = []
    else:
        indices = [int(tok) for tok in response.replace(" ", "").split(",") if tok.isdigit()]
        graded = [candidates[i] for i in indices if i < len(candidates)]

    state["graded"] = graded
    state["trace"] = state["trace"] + [f"grade: {len(graded)}/{len(candidates)} passages relevant"]
    return state


def rewrite_query(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = (
        "The following search query did not retrieve useful results from a "
        "document knowledge base. Rewrite it to be more likely to match relevant "
        "content — consider synonyms, more specific terms, or a different phrasing. "
        "Reply with ONLY the rewritten query, nothing else.\n\n"
        f"Original query: {state['question']}"
    )
    rewritten = llm.invoke(prompt).content.strip().strip('"')

    state["question"] = rewritten
    state["retries"] = state["retries"] + 1
    state["trace"] = state["trace"] + [f"rewrite: retry {state['retries']} -> \"{rewritten}\""]
    return state


def generate(state: AgentState) -> AgentState:
    llm = get_llm()
    graded = state["graded"]

    if not graded:
        answer = (
            "I couldn't find relevant information in the knowledge base to "
            "answer that confidently. Could you rephrase, or is this outside "
            "the scope of the ingested documents?"
        )
        state["answer"] = answer
        state["sources"] = ""
        state["trace"] = state["trace"] + ["generate: no grounded context available -> fallback answer"]
        return state

    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(graded))
    prompt = (
        "Answer the question using ONLY the numbered context below. Cite sources "
        "inline using their bracket number, e.g. [1]. If the context doesn't "
        "fully answer the question, say what's missing rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n\n"
        "Answer:"
    )
    answer = llm.invoke(prompt).content.strip()

    state["answer"] = answer
    state["sources"] = guardrails.format_sources(graded)
    state["trace"] = state["trace"] + ["generate: answer drafted with citations"]
    return state


def direct_answer(state: AgentState) -> AgentState:
    llm = get_llm()
    answer = llm.invoke(state["original_question"]).content.strip()
    state["answer"] = answer
    state["sources"] = ""
    state["grounded"] = True
    state["trace"] = state["trace"] + ["direct_answer: answered without retrieval"]
    return state


def check_groundedness_node(state: AgentState) -> AgentState:
    result = guardrails.check_groundedness(state["answer"], state["graded"])
    state["grounded"] = result["grounded"]
    state["groundedness_score"] = result["score"]
    state["trace"] = state["trace"] + [f"check_grounding: score={result['score']} grounded={result['grounded']}"]
    return state


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def after_route(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "direct_answer"


def after_grading(state: AgentState) -> str:
    if not state["graded"] and state["retries"] < config.MAX_AGENT_RETRIES:
        return "rewrite"
    return "generate"


def after_groundedness(state: AgentState) -> str:
    if not state["grounded"] and state["retries"] < config.MAX_AGENT_RETRIES:
        return "rewrite"
    return "end"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("check_grounding", check_groundedness_node)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("route")
    graph.add_conditional_edges("route", after_route, {"retrieve": "retrieve", "direct_answer": "direct_answer"})
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges("grade", after_grading, {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check_grounding")
    graph.add_conditional_edges("check_grounding", after_groundedness, {"rewrite": "rewrite", "end": END})
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run(question: str) -> dict:
    """Entry point used by app.py and eval/evaluate.py."""
    initial_state: AgentState = {
        "question": question,
        "original_question": question,
        "needs_retrieval": False,
        "retries": 0,
        "candidates": [],
        "graded": [],
        "answer": "",
        "grounded": False,
        "groundedness_score": 0.0,
        "sources": "",
        "trace": [],
    }
    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
