"""Corpus loading and the Searcher agent's retrieval.

The corpus is a small set of synthetic "web pages" (fictional companies).
The Searcher ranks pages by TF-IDF cosine similarity and can apply a
freshness cutoff so stale pages (e.g. a 2024 pricing archive) are excluded
from the multi-agent path. The single-agent baseline skips the cutoff —
that is exactly the failure mode the eval measures.
"""

import json
import os

from .text import TfidfIndex

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_corpus(path=None):
    path = path or os.path.join(DATA_DIR, "corpus.json")
    with open(path) as f:
        data = json.load(f)
    return data["pages"]


def load_eval_questions(path=None):
    path = path or os.path.join(DATA_DIR, "eval_questions.json")
    with open(path) as f:
        return json.load(f)["questions"]


class Searcher:
    def __init__(self, pages):
        self.pages = {p["id"]: p for p in pages}
        self.index = TfidfIndex([(p["id"], p["title"] + " " + p["text"]) for p in pages])

    def search(self, query, top_k=3, freshness_cutoff=None, min_score=0.0):
        """Return [(page, score)] ranked best-first.

        freshness_cutoff: ISO date string; pages older than it are excluded.
        min_score: drop results below this TF-IDF cosine score.
        """
        results = []
        for doc_id, score in self.index.search(query, top_k=len(self.pages)):
            page = self.pages[doc_id]
            if freshness_cutoff and page["date"] < freshness_cutoff:
                continue
            if score < min_score:
                continue
            results.append((page, score))
            if len(results) >= top_k:
                break
        return results
