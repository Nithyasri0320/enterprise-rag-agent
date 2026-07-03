"""
Guardrails: catch ungrounded (hallucinated) answers before they reach the
user, and format citations so every claim is traceable back to a source
chunk. This is the piece that answers "how do you know it's not making
things up" in an interview.
"""
import re

from src import config


def _split_sentences(text: str):
    # Lightweight sentence splitter — good enough for a groundedness check,
    # no need for a full NLP sentence tokenizer here.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) > 3]


def check_groundedness(answer: str, context_chunks: list) -> dict:
    """Heuristic groundedness check: what fraction of the answer's sentences
    have meaningful word-overlap with *some* retrieved chunk?

    This is a cheap, fast, dependency-free first line of defense. For
    production you'd likely swap this for (or combine it with) an LLM-judge
    call or a NLI-based entailment model — see eval/evaluate.py, which uses
    RAGAS's faithfulness metric for a more rigorous version of this same idea.
    """
    if not context_chunks:
        return {"grounded": False, "score": 0.0, "reason": "No context retrieved."}

    context_text = " ".join(c["text"] for c in context_chunks).lower()
    context_words = set(re.findall(r"\b\w+\b", context_text))

    sentences = _split_sentences(answer)
    if not sentences:
        return {"grounded": False, "score": 0.0, "reason": "Empty answer."}

    grounded_count = 0
    for sentence in sentences:
        words = set(re.findall(r"\b\w+\b", sentence.lower()))
        # ignore common stopwords/short tokens so the overlap check reflects
        # actual content words, not "the", "is", "of", etc.
        content_words = {w for w in words if len(w) > 3}
        if not content_words:
            grounded_count += 1  # too short to falsify, don't penalize
            continue
        overlap = len(content_words & context_words) / len(content_words)
        if overlap >= 0.4:
            grounded_count += 1

    score = grounded_count / len(sentences)
    return {
        "grounded": score >= config.GROUNDEDNESS_THRESHOLD,
        "score": round(score, 2),
        "reason": f"{grounded_count}/{len(sentences)} sentences traced to retrieved context.",
    }


def format_sources(context_chunks: list) -> str:
    """Dedup + format a citation footer, e.g. [1] company_policy.txt"""
    seen = {}
    for c in context_chunks:
        source = c["metadata"].get("source", "unknown")
        name = source.split("/")[-1]
        if name not in seen:
            seen[name] = len(seen) + 1

    lines = [f"[{idx}] {name}" for name, idx in seen.items()]
    return "\n".join(lines)
