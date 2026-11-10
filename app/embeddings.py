"""Embed text with sentence-transformers models and cache the vectors on disk."""

from functools import cache
from pathlib import Path

import numpy as np

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


@cache
def load_model(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELS[name])


def embed(name, texts, show_progress_bar=False):
    """Normalized vectors, so inner product = cosine similarity in every store."""
    return load_model(name).encode(texts, normalize_embeddings=True, show_progress_bar=show_progress_bar)


def encoder(name):
    """A function that embeds one query with the given model, as retrievers expect."""
    return lambda query: embed(name, [query])[0]


def count_truncated(name, texts):
    """How many texts are longer than the model's limit and get cut off (2 tokens go to start/end markers)."""
    model = load_model(name)
    return sum(len(model.tokenizer.tokenize(t)) > model.max_seq_length - 2 for t in texts)


def embeddings_path(out_dir, cache_name, suffix=".npy"):
    return Path(out_dir) / "embeddings" / f"{cache_name}{suffix}"


def load_or_embed(name, texts, out_dir, cache_name=None):
    """Load the saved vectors, or embed the texts with this model and save them. Row i is the vector for texts[i].

    cache_name tells apart different texts embedded with the same model (default: the model name, for the children).
    """
    path = embeddings_path(out_dir, cache_name or name)
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(texts):
            return vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = embed(name, texts, show_progress_bar=True)
    np.save(path, vectors)
    return vectors


#learned sparse (SPLADE): one weight per vocabulary word, almost all of them zero
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


@cache
def load_sparse_model():
    from sentence_transformers import SparseEncoder

    return SparseEncoder(SPLADE_MODEL)


def _to_scipy(tensor):
    import scipy.sparse
    import torch

    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    tensor = tensor.coalesce().cpu()
    rows, cols = tensor.indices().numpy()
    return scipy.sparse.csr_matrix((tensor.values().numpy(), (rows, cols)), shape=tuple(tensor.shape))


def sparse_embed_documents(texts, show_progress_bar=False):
    """A scipy sparse matrix with one row per text."""
    return _to_scipy(load_sparse_model().encode_document(texts, show_progress_bar=show_progress_bar))


def sparse_embed_query(query):
    """A scipy sparse matrix with one row for the query."""
    return _to_scipy(load_sparse_model().encode_query([query]))


def load_or_sparse_embed(texts, out_dir, cache_name="splade_parents"):
    """Load the saved SPLADE vectors, or embed the texts and save them. Row i is the vector for texts[i]."""
    import scipy.sparse

    path = embeddings_path(out_dir, cache_name, suffix=".npz")
    if path.exists():
        vectors = scipy.sparse.load_npz(path)
        if vectors.shape[0] == len(texts):
            return vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = sparse_embed_documents(texts, show_progress_bar=True)
    scipy.sparse.save_npz(path, vectors)
    return vectors
