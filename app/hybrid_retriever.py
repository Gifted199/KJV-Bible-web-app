"""Combine several ranked result lists into one with reciprocal rank fusion (RRF).

Each list gives a passage 1 / (RRF_K + rank) points; passages are re-ranked by their total. Only ranks are used,
so retrievers whose scores are on different scales (cosine similarity, BM25) can be mixed fairly.
"""

RRF_K = 60  # the usual constant; larger values flatten the difference between ranks


def reciprocal_rank_fusion(ranked_lists, k=10, rrf_k=RRF_K):
    points, items = {}, {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            points[item["id"]] = points.get(item["id"], 0) + 1 / (rrf_k + rank)
            items[item["id"]] = item
    best = sorted(points, key=points.get, reverse=True)[:k]
    return [(items[i], points[i]) for i in best]


class HybridRetriever:
    """Dense (meaning) plus sparse (exact words): run every retriever, then fuse their rankings."""

    def __init__(self, retrievers, depth=50):
        self.retrievers = retrievers
        self.depth = depth  # how many results to take from each retriever before fusing

    def search_with_scores(self, query, k=10):
        return reciprocal_rank_fusion([r.search(query, self.depth) for r in self.retrievers], k=k)

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]
