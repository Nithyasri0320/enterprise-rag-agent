"""
Evaluation harness using RAGAS.

Runs each question in eval_dataset.json through the full agent, then scores
the results on:
  - faithfulness       : is the answer actually supported by the retrieved context?
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : is the retrieved context relevant (not noisy)?

This is the rigorous version of the heuristic check in src/guardrails.py —
in an interview, guardrails.py is your *runtime* safety net, this file is
your *offline* quality measurement.

Run:  python -m eval.evaluate
"""
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import config, agent


def load_eval_set():
    path = Path(__file__).parent / "eval_dataset.json"
    with open(path) as f:
        return json.load(f)


def run_agent_over_eval_set(eval_set):
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in eval_set:
        print(f"Running agent on: {item['question']}")
        result = agent.run(item["question"])

        contexts = [c["text"] for c in result.get("graded", [])] or ["(no context retrieved)"]

        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def main():
    eval_set = load_eval_set()
    dataset = run_agent_over_eval_set(eval_set)

    judge_llm = LangchainLLMWrapper(ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL))

    print("\nScoring with RAGAS...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)

    df = results.to_pandas()
    out_path = Path(__file__).parent / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-question breakdown saved to {out_path}")


if __name__ == "__main__":
    main()
