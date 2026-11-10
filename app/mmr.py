"""Maximum Marginal Relevance: re-rank results so they are relevant but not repetitive.

Each step picks the candidate with the best  lambda * relevance - (1 - lambda) * similarity to anything already
picked. With lambda = 1 it is the plain ranking; lower values push near-duplicates (e.g. the next part of the
same section) further down, making room for different passages.
"""

import numpy as np

MMR_LAMBDA = 0.7
CANDIDATES = 30


def mmr(relevance, similarity, k, lambda_=MMR_LAMBDA):
    """Indices of the chosen candidates, in order.

    relevance: one score per candidate; similarity: candidate-by-candidate matrix.
    """
    chosen, remaining = [], list(range(len(relevance)))
    while remaining and len(chosen) < k:
        def value(i):
            redundancy = max(similarity[i][j] for j in chosen) if chosen else 0.0
            return lambda_ * relevance[i] - (1 - lambda_) * redundancy

        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen


class MMRRetriever:
    """Take the base retriever's top candidates and re-rank them with MMR, using passage vectors for similarity."""

    def __init__(self, base, vectors_by_id, lambda_=MMR_LAMBDA, candidates=CANDIDATES):
        self.base = base  # must offer search_with_scores
        self.vectors_by_id = vectors_by_id  # passage id -> normalized vector
        self.lambda_ = lambda_
        self.candidates = candidates

    def search_with_scores(self, query, k=10):
        results = self.base.search_with_scores(query, self.candidates)
        if not results:
            return []
        items = [item for item, _ in results]
        relevance = [s for _, s in results]
        vectors = np.stack([self.vectors_by_id[item["id"]] for item in items])
        chosen = mmr(relevance, vectors @ vectors.T, k, self.lambda_)
        return [(items[i], relevance[i]) for i in chosen]

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]
