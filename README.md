# Multi-Agent Research Assistant

A supervisor + worker-agent research assistant that answers questions from a
corpus of documents — and abstains when the corpus cannot support an answer.

> **SYNTHETIC DATA — this is a portfolio skill demo, not work experience.**
> All companies, prices, and metrics in `data/corpus.json` are fictional and
> were created for this demo. The system is a local stand-in: the "web search"
> runs over a bundled corpus, the summariser is extractive, and the graph is
> a small stdlib implementation of the LangGraph supervisor pattern — no
> LangGraph, no LLM API key, no production agent platform is claimed.

## The business question

Single-agent RAG pipelines fail in two predictable ways: they quote stale
documents, and they answer questions the sources cannot support. This project
shows the multi-agent fix — a supervisor that plans (fan-out), searches with
a freshness cutoff, summarises extractively, and fact-checks every claim
before answering — measured against the single-agent baseline on the same
10-question eval set.

## How it works

```
question
  └─ Planner    — fan-out: one sub-question per retrieved source topic
  └─ Searcher   — TF-IDF retrieval over 12 synthetic "web pages",
                  with a freshness cutoff (2025-01-01) that excludes stale pages
  └─ Summarizer — extractive: top sentences by IDF-weighted question overlap,
                  quoted with source ids; a sentence only counts if it shares
                  ≥2 distinctive question tokens (numbers/years don't count)
  └─ FactChecker— every draft claim verified against its cited source:
                  all distinctive words must occur in the source, token
                  overlap ≥ 0.35, every number in the claim must appear there
  └─ Supervisor — retries once with a wider search if everything is dropped,
                  then ANSWERS with the verified claims or ABSTAINS
```

The graph is explicit nodes + conditional edges (`research/graph.py`), the
same shape as a `langgraph.StateGraph` supervisor — swap the runner for the
real framework to run these nodes on LangGraph. Stdlib only, so it runs
anywhere: `python3 demo.py`.

## Results (real runs — reproduced byte-identically from scratch, excl. latency)

10-question eval set: 8 answerable, 2 unanswerable (`data/eval_questions.json`).
Full per-question report: `outputs/eval_report.json`.

| Metric | Multi-agent graph | Single-agent baseline |
|---|---|---|
| Answerable accuracy (expected facts in answer) | **1.00 (8/8)** | 0.75 (6/8) |
| Abstention recall (unanswerable → abstain) | **1.00 (2/2)** | 0.00 (0/2) |
| Citation coverage | 1.00 | — |
| Avg tool calls / question | 3.7 | 1 |

What the baseline gets wrong, concretely:

- **Stale price trap.** The corpus contains a 2024 pricing archive (Pro $79)
  and the current 2025 page (Pro $99). The baseline has no freshness cutoff
  and answers **$79**. The multi-agent graph answers **$99**.
- **Unanswerable questions.** Asked for the CEO's name and for 2025 total
  revenue — neither is in the corpus — the baseline answers anyway (from
  unrelated pages). The graph abstains on both.
- **Wrong sentence.** Asked how much Meridian Health cut reporting time, the
  baseline quotes the deployment sentence (no number). The graph's
  IDF-weighted summariser surfaces the sentence with the 40% figure.

The fact-checker also catches planted errors (unit-tested): a wrong price
($79 claimed against the $99 source) and an invented feature ("AI fraud
detection") are both flagged UNSUPPORTED.

## Known limitations (honest)

- The corpus is 12 hand-written synthetic pages; retrieval quality on a real
  web index is untested.
- The summariser is extractive, so answers are source sentences, not prose.
- The fact-checker is bag-of-words: it cannot catch attribution swaps
  ("Meridian did X" vs "Beta did X" with the same words) — that needs NLI,
  as documented in the sibling project `hallucination-guardrails`.
- The grounding rule (≥2 distinctive shared tokens) is tuned to this corpus;
  re-tune the threshold on your own data.

## Run it

```bash
python3 demo.py                                   # one question, full trace
python3 -m research.cli ask "What is the Pro plan price?"
python3 -m research.cli eval                      # full eval vs baseline
pytest tests/ -q                                  # 16 tests
```

## Layout

```
research/
  text.py       tokenisation, TF-IDF index, sentence splitting
  corpus.py     corpus loading + Searcher (freshness cutoff, min score)
  agents.py     Planner, Summarizer, FactChecker, Supervisor, baseline
  graph.py      supervisor graph runner (nodes + conditional edges)
  evaluate.py   eval runner: multi-agent vs single-agent baseline
  cli.py        `ask` / `eval` commands
data/
  corpus.json          12 synthetic pages (incl. one stale 2024 archive)
  eval_questions.json  10 questions with expected facts
tests/          16 pytest checks (incl. planted-error fact-check tests)
outputs/        eval_report.json, summary.json from the real run
```

## Related projects

- Project 3, `agentic-rag-evals` — single agentic RAG with evals (this
  project is the multi-agent orchestration step beyond it).
- Project 17, `rag-eval-harness` — standalone eval CLI (same eval-set schema).
- Project 18, `hallucination-guardrails` — per-answer claim verification
  library; the fact-checker here reuses its number-agreement idea.
