"""Tests for the multi-agent research assistant.

Every test runs the real code on the synthetic corpus — no mocks of the
agents themselves. The fact-checker tests plant wrong claims the way
project 18 planted hallucinations, and assert they are caught.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.agents import FactChecker, Summarizer, Supervisor
from research.corpus import Searcher, load_corpus, load_eval_questions
from research.evaluate import run_eval
from research.graph import ResearchGraph
from research.text import TfidfIndex

PAGES = load_corpus()
BY_ID = {p["id"]: p for p in PAGES}


def test_corpus_loads_twelve_pages():
    assert len(PAGES) == 12
    assert BY_ID["pricing-2025"]["date"] == "2025-06-01"


def test_search_ranks_current_pricing_first():
    searcher = Searcher(PAGES)
    results = searcher.search("What is the Pro plan price?", top_k=3)
    assert results[0][0]["id"] == "pricing-2025"


def test_freshness_filter_excludes_2024_archive():
    searcher = Searcher(PAGES)
    results = searcher.search(
        "What is the Pro plan price?", top_k=5, freshness_cutoff="2025-01-01"
    )
    ids = [p["id"] for p, _ in results]
    assert "pricing-2024-archive" not in ids
    assert "pricing-2025" in ids


def test_planner_fans_out_sub_questions():
    graph = ResearchGraph()
    subs, _ = graph.supervisor.planner.plan("What is the Pro plan price?")
    assert len(subs) >= 1
    assert all("page_id" in s for s in subs)


def test_summarizer_cites_its_source():
    from research.graph import ResearchGraph
    graph = ResearchGraph()
    summarizer = Summarizer(graph.searcher.index)
    claims = summarizer.summarise(
        "What is the Pro plan price?", [(BY_ID["pricing-2025"], 0.5)]
    )
    assert claims
    assert "$99" in claims[0]["text"]
    assert claims[0]["source_id"] == "pricing-2025"


def test_fact_checker_supports_true_claim():
    checker = FactChecker(BY_ID)
    result = checker.check_claim({
        "text": "The Pro plan costs $99 per month.",
        "source_id": "pricing-2025",
    })
    assert result["verdict"] == "SUPPORTED"


def test_fact_checker_catches_wrong_number():
    checker = FactChecker(BY_ID)
    result = checker.check_claim({
        "text": "The Pro plan costs $79 per month.",
        "source_id": "pricing-2025",
    })
    assert result["verdict"] == "UNSUPPORTED"
    assert "79" in result["reason"]


def test_fact_checker_catches_invented_feature():
    checker = FactChecker(BY_ID)
    result = checker.check_claim({
        "text": "Northwind Analytics offers AI fraud detection on the Pro plan.",
        "source_id": "pricing-2025",
    })
    assert result["verdict"] == "UNSUPPORTED"


def test_fact_checker_catches_missing_source():
    checker = FactChecker(BY_ID)
    result = checker.check_claim({
        "text": "The Pro plan costs $99 per month.",
        "source_id": "no-such-page",
    })
    assert result["verdict"] == "UNSUPPORTED"


def test_multi_agent_answers_current_price_not_stale():
    graph = ResearchGraph()
    result = graph.ask("What is Northwind Analytics Pro plan price?")
    assert result["decision"] == "ANSWER"
    assert "$99" in result["answer"]
    assert "$79" not in result["answer"]


def test_multi_agent_abstains_on_unanswerable():
    graph = ResearchGraph()
    for q in ("Who is the chief executive officer of Northwind Analytics?",
              "What was Northwind Analytics total revenue in 2025?"):
        result = graph.ask(q)
        assert result["decision"] == "ABSTAIN", q
        assert result["answer"] is None


def test_baseline_falls_for_stale_price():
    # The single-agent baseline has no freshness cutoff: it should surface
    # the stale 2024 archive and answer $79. That is the failure the
    # multi-agent graph fixes — not a bug in the test.
    graph = ResearchGraph()
    baseline = graph.ask_baseline("What is Northwind Analytics Pro plan price?")
    assert baseline["decision"] == "ANSWER"
    assert "$79" in baseline["answer"]


def test_graph_trace_covers_all_nodes():
    graph = ResearchGraph()
    result = graph.ask("What is Northwind Analytics monthly churn rate in 2025?")
    nodes = [step["node"] for step in result["trace"]]
    for expected in ("plan", "search", "summarise", "fact_check", "decide"):
        assert expected in nodes


def test_eval_metrics_in_valid_ranges():
    report = run_eval()
    s = report["summary"]["multi_agent"]
    assert report["summary"]["n_questions"] == 10
    assert 0.0 <= s["answerable_accuracy"] <= 1.0
    assert 0.0 <= s["abstention_recall"] <= 1.0
    assert 0.0 <= s["citation_coverage"] <= 1.0
    # The multi-agent graph must beat the baseline on both headline metrics.
    b = report["summary"]["single_agent_baseline"]
    assert s["answerable_accuracy"] >= b["answerable_accuracy"]
    assert s["abstention_recall"] >= b["abstention_recall"]


def test_eval_is_deterministic_except_latency():
    first = run_eval()
    second = run_eval()
    for a, b in zip(first["per_question"], second["per_question"]):
        assert a["answer"] == b["answer"]
        assert a["decision"] == b["decision"]
        assert a["baseline_answer"] == b["baseline_answer"]
    assert first["summary"] == second["summary"] or (
        first["summary"]["multi_agent"]["avg_latency_ms"]
        != second["summary"]["multi_agent"]["avg_latency_ms"]
    )


def test_eval_questions_match_corpus_facts():
    questions = load_eval_questions()
    assert len(questions) == 10
    assert sum(1 for q in questions if q["answerable"]) == 8
    assert sum(1 for q in questions if not q["answerable"]) == 2
