"""Shared text helpers: tokenisation, TF-IDF cosine similarity, sentence splitting.

Stdlib only. The same helpers power retrieval, extractive summarisation and
claim verification, so the three agents share one definition of "match".
"""

import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "and", "or", "at", "by", "from",
    "as", "it", "its", "this", "that", "what", "who", "how", "much", "many",
    "does", "do", "did", "per", "after", "before", "up", "down", "now",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def tokenize(text):
    """Lowercase word tokens, keeping numbers like 50,000 and 4.2 intact."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text):
    """Tokens minus stopwords — the words that carry meaning for matching."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text):
    """Split on sentence-ending punctuation. Keeps the trailing period."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts


def numbers_in(text):
    """Normalised numbers mentioned in a string ('$99', '4.2 percent', ...)."""
    found = set()
    for raw in _NUMBER_RE.findall(text):
        norm = raw.replace("$", "").replace(",", "").rstrip("%")
        found.add(norm)
    return found


class TfidfIndex:
    """Tiny TF-IDF index over documents, cosine similarity ranking."""

    def __init__(self, docs):
        """docs: list of (doc_id, text)."""
        self.doc_ids = [d for d, _ in docs]
        self.doc_tokens = {d: content_tokens(t) for d, t in docs}
        n = len(docs)
        df = Counter()
        for tokens in self.doc_tokens.values():
            for tok in set(tokens):
                df[tok] += 1
        self.df = dict(df)
        self.n_docs = n
        self.idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}

    def doc_freq(self, token):
        return self.df.get(token, 0)

    def is_ubiquitous(self, token):
        """True if the token appears in more than half the documents —
        e.g. the company name. Ubiquitous tokens carry no evidence that a
        sentence answers a specific question."""
        return self.df.get(token, 0) > self.n_docs / 2

    def _vector(self, tokens):
        counts = Counter(tokens)
        vec = {}
        for tok, c in counts.items():
            if tok in self.idf:
                vec[tok] = (1 + math.log(c)) * self.idf[tok]
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {tok: v / norm for tok, v in vec.items()}

    def score(self, query, doc_id):
        q = self._vector(content_tokens(query))
        d = self._vector(self.doc_tokens[doc_id])
        return sum(q.get(tok, 0.0) * d.get(tok, 0.0) for tok in q)

    def search(self, query, top_k=3):
        ranked = sorted(
            ((self.score(query, d), d) for d in self.doc_ids), reverse=True
        )
        return [(doc_id, round(score, 4)) for score, doc_id in ranked[:top_k]]
