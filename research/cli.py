"""CLI: ask one question, or run the full eval set.

Usage:
    python -m research.cli ask "What is the Pro plan price?"
    python -m research.cli eval
"""

import argparse
import json
import sys

from .evaluate import run_eval
from .graph import ResearchGraph


def cmd_ask(args):
    graph = ResearchGraph()
    result = graph.ask(args.question)
    if result["decision"] == "ABSTAIN":
        print("I don't have enough information in the retrieved sources to answer.")
    else:
        print(result["answer"])
    print(f"\n[decision={result['decision']} tool_calls={result['tool_calls']}]")
    for step in result["trace"]:
        print(f"  {step['node']}: " + ", ".join(
            f"{k}={v}" for k, v in step.items() if k != "node"))
    return 0


def cmd_eval(_args):
    report = run_eval()
    print(json.dumps(report["summary"], indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask one research question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)
    ev = sub.add_parser("eval", help="Run the eval set (multi-agent vs baseline)")
    ev.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
