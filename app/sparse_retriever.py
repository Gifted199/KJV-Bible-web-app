"""Sparse retrieval: rank passages by the words they share with the question.

- TF-IDF: words weighted by how rare they are across the Bible, compared by cosine similarity
- BM25: like TF-IDF, but repeated words give diminishing returns and long passages are not favoured
- SPLADE: a neural model that predicts a weight for every vocabulary word, including related words not in the text
"""

import re

import numpy as np

TOKEN_PATTERN = r"[a-z]+"  # lowercase words; "brother’s" -> "brother", "s"


def tokenize(text):
    return re.findall(TOKEN_PATTERN, text.lower())


def _top_k(items, scores, k):
    best = np.argsort(-scores)[:k]
    return [(items[i], float(scores[i])) for i in best if scores[i] > 0]


class TfidfRetriever:
    def __init__(self, items, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.items = items
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)  # rows are L2-normalized, so a dot product is cosine

    def search_with_scores(self, query, k=10):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return _top_k(self.items, scores, k)

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]


class BM25Retriever:
    def __init__(self, items, texts, k1=1.5, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.items = items
        self.vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        counts = self.vectorizer.fit_transform(texts).tocsc().astype(np.float32)
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.diff(counts.indptr)  # number of passages containing each word
        self.idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        #precompute each word's BM25 weight in each passage; a search then only adds up columns
        length_norm = k1 * (1 - b + b * doc_lengths / doc_lengths.mean())
        weights = counts.copy()
        rows = weights.indices
        weights.data = weights.data * (k1 + 1) / (weights.data + length_norm[rows])
        self.weights = weights

    def search_with_scores(self, query, k=10):
        vocabulary = self.vectorizer.vocabulary_
        word_ids = [vocabulary[w] for w in set(tokenize(query)) if w in vocabulary]
        if not word_ids:
            return []
        scores = np.asarray((self.weights[:, word_ids] @ self.idf[word_ids])).ravel()
        return _top_k(self.items, scores, k)

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]


class SpladeRetriever:
    """Learned sparse retrieval. The document vectors come from embeddings.load_or_sparse_embed."""

    def __init__(self, items, document_matrix, encode_query):
        self.items = items
        self.matrix = document_matrix.tocsr()
        self.encode_query = encode_query  # query -> sparse row (scipy)

    def search_with_scores(self, query, k=10):
        scores = (self.matrix @ self.encode_query(query).T).toarray().ravel()
        return _top_k(self.items, scores, k)

    def search(self, query, k=10):
        return [item for item, _ in self.search_with_scores(query, k)]
