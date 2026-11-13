"""Re-rank a retriever's top candidates with a second, more careful model.

The first retriever (Hybrid) is fast and finds good candidates; a reranker re-scores only those few, so it can afford
to be slower:

    bi-encoder     embeds the question and each passage separately and compares the vectors
    cross-encoder  reads the question and a passage together and outputs one relevance score

A scorer is any function (question, passage texts) -> one score per passage, so tests can pass a fake one.
"""

from functools import cache

BI_ENCODER = "sentence-transformers/multi-qa-mpnet-base-cos-v1"  # trained on question -> answer pairs
CROSS_ENCODERS = {
    "MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # small and fast
    "bge": "BAAI/bge-reranker-base",  # larger, usually more accurate
}
#how many of the first retriever's results get re-scored: 10 scored exactly as well as 20 on the test questions,
#in half the time
CANDIDATES = 10


@cache
def load_bi_encoder(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@cache
def load_cross_encoder(model_name):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def bi_encoder_scorer(model_name=BI_ENCODER):
    def scores(question, texts):
        model = load_bi_encoder(model_name)
        question_vector = model.encode([question], normalize_embeddings=True)[0]
        passage_vectors = model.encode(texts, normalize_embeddings=True)
        return (passage_vectors @ question_vector).tolist()  # cosine similarity

    return scores


def cross_encoder_scorer(model_name):
    def scores(question, texts):
        #one pair at a time: a batch is padded to its longest passage, which wastes about 30% of the time on a CPU
        pairs = [(question, text) for text in texts]
        return load_cross_encoder(model_name).predict(pairs, batch_size=1).tolist()

    return scores


class RerankRetriever:
    """Take the base retriever's top candidates and put them in the order the scorer prefers."""

    def __init__(self, base, scorer, candidates=CANDIDATES):
        self.base = base
        self.scorer = scorer
        self.candidates = candidates

    def search_with_scores(self, query, k=10):
        items = self.base.search(query, self.candidates)
        if not items:
            return []
        scores = self.scorer(query, [item["text"] for item in items])
        ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]


def build_rerankers(base, candidates=CANDIDATES):
    """name -> retriever: the base on its own, then the base re-ranked by each model."""
    return {
        "Hybrid (no reranker)": base,
        "Hybrid + bi-encoder": RerankRetriever(base, bi_encoder_scorer(), candidates),
        **{
            f"Hybrid + cross-encoder ({name})": RerankRetriever(base, cross_encoder_scorer(model), candidates)
            for name, model in CROSS_ENCODERS.items()
        },
    }
