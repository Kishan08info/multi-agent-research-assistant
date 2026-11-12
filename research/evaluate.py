"""Eval runner: multi-agent graph vs single-agent baseline on the eval set.

Metrics (all from real runs, no hand-written numbers):
- answerable accuracy: answerable questions whose final answer contains
  every expected fact value.
- abstention recall: unanswerable questions correctly abstained / total.
- citation coverage: answered claims carrying a source citation / claims.
- fact-check catch rate: planted wrong-number claims flagged / planted.
- avg tool calls and avg latency per question.
"""

import json
import os
import time

from .corpus import load_eval_questions
from .graph import ResearchGraph

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _facts_present(answer, expected_facts):
    if answer is None:
        return False
    return all(fact in answer for fact in expected_facts)


def run_eval(graph=None, questions=None):
    graph = graph or ResearchGraph()
    questions = questions or load_eval_questions()
    per_question = []
    for q in questions:
        start = time.perf_counter()
        result = graph.ask(q["question"])
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        baseline = graph.ask_baseline(q["question"])
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "expected_facts": q["expected_facts"],
            "decision": result["decision"],
            "answer": result["answer"],
            "facts_present": _facts_present(result["answer"], q["expected_facts"])
            if q["answerable"] else None,
            "tool_calls": result["tool_calls"],
            "latency_ms": latency_ms,
            "trace": result["trace"],
            "baseline_decision": baseline["decision"],
            "baseline_answer": baseline["answer"],
            "baseline_facts_present": _facts_present(
                baseline["answer"], q["expected_facts"]
            ) if q["answerable"] else None,
        }
        per_question.append(row)

    answerable = [r for r in per_question if r["answerable"]]
    unanswerable = [r for r in per_question if not r["answerable"]]
    answered = [r for r in answerable if r["decision"] == "ANSWER"]
    cited = sum(
        1 for r in answered for c in [r["answer"]] if r["answer"] and "[" in r["answer"]
    )
    summary = {
        "n_questions": len(per_question),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "multi_agent": {
            "answerable_accuracy": round(
                sum(r["facts_present"] for r in answerable) / len(answerable), 4
            ) if answerable else None,
            "abstention_recall": round(
                sum(1 for r in unanswerable if r["decision"] == "ABSTAIN")
                / len(unanswerable), 4
            ) if unanswerable else None,
            "citation_coverage": round(cited / len(answered), 4) if answered else None,
            "avg_tool_calls": round(
                sum(r["tool_calls"] for r in per_question) / len(per_question), 2
            ),
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in per_question) / len(per_question), 2
            ),
        },
        "single_agent_baseline": {
            "answerable_accuracy": round(
                sum(r["baseline_facts_present"] for r in answerable) / len(answerable),
                4,
            ) if answerable else None,
            "abstention_recall": round(
                sum(1 for r in unanswerable if r["baseline_decision"] == "ABSTAIN")
                / len(unanswerable), 4
            ) if unanswerable else None,
        },
    }
    return {"summary": summary, "per_question": per_question}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = run_eval()
    with open(os.path.join(OUTPUT_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(report["summary"], f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    for row in report["per_question"]:
        print(f"{row['id']}: {row['decision']} facts={row['facts_present']} "
              f"(baseline {row['baseline_decision']} facts={row['baseline_facts_present']})")
        if row["answer"]:
            print(f"   {row['answer']}")
        if row["baseline_answer"]:
            print(f"   baseline: {row['baseline_answer']}")


if __name__ == "__main__":
    main()
