"""Demo: one research question end-to-end, with the agent trace printed.

Run: python demo.py
"""

from research.graph import ResearchGraph

QUESTION = "What is Northwind Analytics Pro plan price?"


def main():
    graph = ResearchGraph()
    print(f"Question: {QUESTION}\n")
    result = graph.ask(QUESTION)
    print("Answer:", result["answer"] or "(abstained)")
    print(f"Decision: {result['decision']} | tool calls: {result['tool_calls']}\n")
    print("Trace:")
    for step in result["trace"]:
        detail = ", ".join(f"{k}={v}" for k, v in step.items() if k != "node")
        print(f"  {step['node']}: {detail}")
    print("\nBaseline (single agent, no freshness filter, no fact-check):")
    baseline = graph.ask_baseline(QUESTION)
    print("Answer:", baseline["answer"] or "(abstained)")


if __name__ == "__main__":
    main()
