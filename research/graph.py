"""The supervisor graph: explicit nodes + conditional edges, LangGraph-style.

Each node is a function over a shared state dict; edges decide the next node
from the state. This is a small stdlib implementation of the supervisor
pattern (no LangGraph dependency, so it runs anywhere) — swap the runner
for langgraph.StateGraph to run the same nodes on the real framework.
"""

from .agents import Supervisor, run_single_agent_baseline
from .corpus import Searcher, load_corpus


class ResearchGraph:
    NODES = ["plan", "search", "summarise", "fact_check", "decide"]

    def __init__(self, pages=None):
        self.pages = pages or load_corpus()
        self.pages_by_id = {p["id"]: p for p in self.pages}
        self.searcher = Searcher(self.pages)
        self.supervisor = Supervisor(self.searcher, self.pages_by_id)

    def ask(self, question):
        """Run the multi-agent graph for one question."""
        return self.supervisor.run(question)

    def ask_baseline(self, question):
        """Run the single-agent baseline for the same question."""
        return run_single_agent_baseline(question, self.searcher)
