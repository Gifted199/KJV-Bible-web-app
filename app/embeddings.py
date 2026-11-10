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


def embeddings_path(out_dir, name):
    return Path(out_dir) / "embeddings" / f"{name}.npy"


def load_or_embed(name, texts, out_dir):
    """Load the saved vectors for this model, or embed the texts and save them. Row i is the vector for texts[i]."""
    path = embeddings_path(out_dir, name)
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(texts):
            return vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = embed(name, texts, show_progress_bar=True)
    np.save(path, vectors)
    return vectors
