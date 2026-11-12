"""The worker agents and the supervisor.

Planner     — decomposes a question into sub-questions, one per retrieved
              source topic (fan-out), so each worker searches independently.
Searcher     — see corpus.py (TF-IDF retrieval with a freshness cutoff).
Summarizer   — extractive: picks the best source sentences per sub-question
              and composes a cited draft. It never invents a sentence.
FactChecker  — verifies every draft claim against its cited source:
              citation present, token overlap >= threshold, and every number
              in the claim must appear in the cited source. Failed claims
              are dropped, never repaired by guessing.
Supervisor   — runs plan -> search -> summarise -> fact-check, retries once
              with a wider search if everything was dropped, then decides:
              answer with the verified claims, or abstain.
"""

import re

from .text import content_tokens, numbers_in, split_sentences

_NUMBER_LIKE = re.compile(r"^\d[\d,\.]*$")

FRESHNESS_CUTOFF = "2025-01-01"  # stale pages (e.g. the 2024 price archive) excluded
MIN_RETRIEVAL_SCORE = 0.10       # below this, a page is not evidence for the question
OVERLAP_THRESHOLD = 0.35         # claim/source content-token overlap to count as supported
MAX_CLAIMS = 3


class Planner:
    """Fan-out: retrieve candidate pages for the question, then make one
    sub-question per distinct source page so workers search in parallel."""

    def __init__(self, searcher):
        self.searcher = searcher

    def plan(self, question, top_k=3):
        candidates = self.searcher.search(
            question, top_k=top_k, freshness_cutoff=FRESHNESS_CUTOFF,
            min_score=MIN_RETRIEVAL_SCORE,
        )
        sub_questions = []
        for page, score in candidates:
            sub_questions.append({
                "sub_question": f"What does '{page['title']}' say about {question}",
                "page_id": page["id"],
                "retrieval_score": score,
            })
        return sub_questions, candidates


class Summarizer:
    """Extractive summariser: the best source sentences per sub-question,
    quoted with their source id. No generation, so no invented facts.

    Sentences are scored by IDF-weighted overlap with the question, so a
    sentence sharing a rare question word (e.g. 'throughput') beats one
    sharing only the company name. Up to two sentences per page are kept,
    pooled across pages, and the top three overall become the draft.
    A sentence only counts as grounding the question if it shares at least
    two non-ubiquitous question tokens — otherwise it is tangential, and
    a question no sentence grounds is unanswerable from this corpus.
    """

    def __init__(self, index):
        self.index = index

    def sentence_score(self, question, sentence):
        q_tokens = set(content_tokens(question))
        s_tokens = set(content_tokens(sentence))
        shared = q_tokens & s_tokens
        score = sum(self.index.idf.get(t, 0.0) for t in shared)
        score += 0.3 * len(numbers_in(sentence))
        return score, shared

    def grounding_tokens(self, question, sentence):
        """Shared question tokens that are not ubiquitous in the corpus and
        not pure numbers/years. A year (2025) or a bare number matches too
        easily to count as evidence — 'total revenue in 2025' must not be
        grounded by a sentence that merely also says 2025."""
        _score, shared = self.sentence_score(question, sentence)
        return {
            t for t in shared
            if not self.index.is_ubiquitous(t) and not _NUMBER_LIKE.match(t)
        }

    def best_sentence(self, question, page_text):
        """Single best sentence in a page (used by the baseline)."""
        best, best_score = None, -1.0
        for sentence in split_sentences(page_text):
            score, _ = self.sentence_score(question, sentence)
            if score > best_score:
                best, best_score = sentence, score
        return best, best_score

    def summarise(self, question, pages_with_scores):
        scored = []
        for rank, (page, _score) in enumerate(pages_with_scores):
            page_sentences = []
            for pos, sentence in enumerate(split_sentences(page["text"])):
                score, _ = self.sentence_score(question, sentence)
                grounding = self.grounding_tokens(question, sentence)
                if len(grounding) < 2:
                    continue
                page_sentences.append((score, sentence))
            page_sentences.sort(key=lambda x: -x[0])
            for score, sentence in page_sentences[:2]:
                scored.append((score, rank, sentence, page))
        scored.sort(key=lambda x: (-x[0], x[1]))
        claims = []
        for _score, _rank, sentence, page in scored[:MAX_CLAIMS]:
            claims.append({
                "text": sentence,
                "source_id": page["id"],
                "source_title": page["title"],
            })
        return claims


class FactChecker:
    """Per-claim verification against the cited source page.

    A claim passes only if (a) its distinctive words — content tokens that
    are neither ubiquitous in the corpus nor numbers — all occur in the
    cited source (an invented feature introduces a word the source never
    uses, e.g. 'fraud detection'), (b) token overlap with the source is
    above threshold, and (c) every number in the claim appears in the
    source. Extractive claims pass by construction; invented ones fail.
    """

    def __init__(self, pages_by_id, index=None):
        self.pages_by_id = pages_by_id
        self.index = index

    def _distinctive_tokens(self, text):
        tokens = set(content_tokens(text))
        if self.index is None:
            return tokens
        return {
            t for t in tokens
            if not self.index.is_ubiquitous(t) and not _NUMBER_LIKE.match(t)
        }

    def check_claim(self, claim):
        source = self.pages_by_id.get(claim["source_id"])
        if source is None:
            return {"verdict": "UNSUPPORTED", "reason": "cited source not found"}
        claim_tokens = set(content_tokens(claim["text"]))
        source_tokens = set(content_tokens(source["text"]))
        if not claim_tokens:
            return {"verdict": "UNSUPPORTED", "reason": "empty claim"}
        unseen = self._distinctive_tokens(claim["text"]) - source_tokens
        if unseen:
            return {
                "verdict": "UNSUPPORTED",
                "reason": f"words not in cited source: {sorted(unseen)}",
            }
        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        if overlap < OVERLAP_THRESHOLD:
            return {
                "verdict": "UNSUPPORTED",
                "reason": f"token overlap {overlap:.2f} < {OVERLAP_THRESHOLD}",
            }
        claim_numbers = numbers_in(claim["text"])
        source_numbers = numbers_in(source["text"])
        missing = claim_numbers - source_numbers
        if missing:
            return {
                "verdict": "UNSUPPORTED",
                "reason": f"numbers not in source: {sorted(missing)}",
            }
        return {"verdict": "SUPPORTED", "reason": f"overlap {overlap:.2f}, numbers match"}

    def check_all(self, claims):
        checked = []
        for claim in claims:
            result = self.check_claim(claim)
            checked.append({**claim, **result})
        return checked


class Supervisor:
    """Owns the graph: plan -> search -> summarise -> fact-check -> decide.
    Retries once with a wider search if the fact-checker drops everything."""

    def __init__(self, searcher, pages_by_id):
        self.searcher = searcher
        self.planner = Planner(searcher)
        self.summarizer = Summarizer(searcher.index)
        self.fact_checker = FactChecker(pages_by_id, searcher.index)
        self.pages_by_id = pages_by_id

    def run(self, question):
        trace = []
        sub_questions, candidates = self.planner.plan(question)
        trace.append({"node": "plan", "sub_questions": len(sub_questions)})
        if not candidates:
            trace.append({"node": "decide", "decision": "ABSTAIN",
                          "reason": "no source passed the retrieval threshold"})
            return {"answer": None, "claims": [], "trace": trace,
                    "decision": "ABSTAIN", "tool_calls": 1}

        # Fan-out: one search per sub-question (each worker searches its page).
        tool_calls = 1 + len(sub_questions)
        pages_with_scores = []
        for sq in sub_questions:
            page = self.pages_by_id[sq["page_id"]]
            pages_with_scores.append((page, sq["retrieval_score"]))
        trace.append({"node": "search", "pages": [p["id"] for p, _ in pages_with_scores]})

        claims = self.summarizer.summarise(question, pages_with_scores)
        trace.append({"node": "summarise", "draft_claims": len(claims)})
        checked = self.fact_checker.check_all(claims)
        supported = [c for c in checked if c["verdict"] == "SUPPORTED"]
        dropped = [c for c in checked if c["verdict"] != "SUPPORTED"]
        trace.append({"node": "fact_check", "supported": len(supported),
                      "dropped": len(dropped)})

        if not supported:
            # One retry: wider search (more pages), same freshness rule.
            retry = self.searcher.search(
                question, top_k=5, freshness_cutoff=FRESHNESS_CUTOFF,
                min_score=MIN_RETRIEVAL_SCORE,
            )
            tool_calls += 1
            retry_claims = self.summarizer.summarise(question, retry)
            retry_checked = self.fact_checker.check_all(retry_claims)
            supported = [c for c in retry_checked if c["verdict"] == "SUPPORTED"]
            dropped += [c for c in retry_checked if c["verdict"] != "SUPPORTED"]
            trace.append({"node": "retry_search", "supported_after_retry": len(supported)})

        if not supported:
            trace.append({"node": "decide", "decision": "ABSTAIN",
                          "reason": "no claim survived fact-checking"})
            return {"answer": None, "claims": checked, "trace": trace,
                    "decision": "ABSTAIN", "tool_calls": tool_calls}

        answer = " ".join(
            f"{c['text']} [{c['source_id']}]" for c in supported
        )
        trace.append({"node": "decide", "decision": "ANSWER",
                      "claims_used": len(supported), "claims_dropped": len(dropped)})
        return {"answer": answer, "claims": checked, "trace": trace,
                "decision": "ANSWER", "tool_calls": tool_calls}


def run_single_agent_baseline(question, searcher):
    """Baseline for comparison: one search (no freshness cutoff, no
    fact-check), extractive answer from the best sentence. This is what a
    single-agent RAG pipeline does — and where the stale-page trap bites."""
    results = searcher.search(question, top_k=3, min_score=0.0)
    if not results or results[0][1] < 0.05:
        return {"answer": None, "decision": "ABSTAIN", "tool_calls": 1}
    summarizer = Summarizer(searcher.index)
    sentence, _ = summarizer.best_sentence(question, results[0][0]["text"])
    if sentence is None:
        return {"answer": None, "decision": "ABSTAIN", "tool_calls": 1}
    return {"answer": f"{sentence} [{results[0][0]['id']}]",
            "decision": "ANSWER", "tool_calls": 1}
