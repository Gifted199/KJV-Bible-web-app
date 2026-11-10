"""Metadata filtering: when a question names a book or testament, only search inside it.

The filter is read from the question with simple rules (no LLM), e.g. "In Psalms, ...",
"in the book of Ruth", "in the Gospel of Luke", "in the New Testament".
Questions without such a phrase are searched normally.
"""

import re

import numpy as np

from app.books import BOOK_GROUPS, BOOKS, NEW_TESTAMENT

ALIASES = {"Psalm": "Psalms", "Songs": "Song of Solomon", "Song of Songs": "Song of Solomon", "Revelations": "Revelation"}
_NAMES = sorted([*BOOKS, *ALIASES], key=len, reverse=True)  # longest first, so "1 John" wins over "John"
BOOK_PHRASE = re.compile(
    r"\b(?:in|from)\s+(?:the\s+)?(?:book\s+of\s+|gospel\s+of\s+|epistle\s+to\s+the\s+)?(" + "|".join(map(re.escape, _NAMES)) + r")\b",
    re.IGNORECASE,
)
TESTAMENT_PHRASE = re.compile(r"\b(?:in|from)\s+the\s+(old|new)\s+testament\b", re.IGNORECASE)
GOSPELS_PHRASE = re.compile(r"\b(?:in|from)\s+the\s+gospels\b", re.IGNORECASE)


def books_named_in(question):
    """The set of books the question limits itself to, or None if it names none."""
    if m := TESTAMENT_PHRASE.search(question):
        new = m.group(1).lower() == "new"
        return {b for b in BOOKS if (b in NEW_TESTAMENT) == new}
    if GOSPELS_PHRASE.search(question):
        return set(BOOK_GROUPS["Gospels"])
    names = {name.lower(): name for name in _NAMES}
    books = {ALIASES.get(names[m.group(1).lower()], names[m.group(1).lower()]) for m in BOOK_PHRASE.finditer(question)}
    return books or None


class MetadataFilterRetriever:
    """Parent-child dense search restricted to the books the question names."""

    def __init__(self, child_vectors, children, parents_by_id, encode, child_k=50):
        self.child_vectors = child_vectors
        self.children = children
        self.child_books = np.array([c["book"] for c in children])
        self.parents_by_id = parents_by_id
        self.encode = encode
        self.child_k = child_k

    def search_with_scores(self, query, k=10):
        scores = self.child_vectors @ self.encode(query)
        books = books_named_in(query)
        if books:
            scores = np.where(np.isin(self.child_books, list(books)), scores, -np.inf)
        best = {}
        for i in np.argsort(-scores)[:self.child_k]:
            if np.isfinite(scores[i]):
                best.setdefault(self.children[i]["parent_id"], float(scores[i]))
        return [(self.parents_by_id[pid], s) for pid, s in list(best.items())[:k]]

    def search(self, query, k=10):
        return [parent for parent, _ in self.search_with_scores(query, k)]
