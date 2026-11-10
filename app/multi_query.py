"""Multi-query retrieval without an LLM: search with several versions of the question, then fuse the results.

Normally an LLM rewrites the question. Here the versions come from fixed rules:
1. the question as asked
2. the question in KJV wording (a word list of common modern -> KJV differences)
3. the question's keywords only (common English words removed)
"""

import re

from app.hybrid_retriever import reciprocal_rank_fusion

#common modern words and the words the KJV uses instead
KJV_WORDS = {
    "holy spirit": "Holy Ghost",
    "you": "ye",
    "your": "thy",
    "yours": "thine",
    "has": "hath",
    "does": "doth",
    "says": "saith",
    "big": "great",
    "rooster": "cock",
    "languages": "tongues",
    "tomb": "sepulchre",
    "donkey": "ass",
    "boat": "ship",
    "thrown": "cast",
    "throws": "casteth",
}
_KJV_PATTERN = re.compile(r"\b(" + "|".join(sorted(KJV_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def kjv_wording(question):
    return _KJV_PATTERN.sub(lambda m: KJV_WORDS[m.group(1).lower()], question)


def keywords(question):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = re.findall(r"[A-Za-z']+", question)
    return " ".join(w for w in words if w.lower() not in ENGLISH_STOP_WORDS)


def query_versions(question):
    """The distinct versions of the question to search with."""
    return list(dict.fromkeys(v for v in [question, kjv_wording(question), keywords(question)] if v.strip()))


class MultiQueryRetriever:
    def __init__(self, base, depth=50):
        self.base = base
        self.depth = depth

    def search_with_scores(self, query, k=10):
        return reciprocal_rank_fusion([self.base.search(v, self.depth) for v in query_versions(query)], k=k)

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]
