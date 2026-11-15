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


#Azure OpenAI text-embedding-3-small: 1,536 dimensions, already normalized to length 1 (so dot product = cosine)
AZURE_BATCH = 100  # texts per request (~25,000 tokens for the parents), far below the API's per-request limits
AZURE_TOKENIZER = "cl100k_base"


def count_tokens(texts):
    """Tokens text-embedding-3-small will be billed for."""
    import tiktoken

    encoding = tiktoken.get_encoding(AZURE_TOKENIZER)
    return sum(len(encoding.encode(t)) for t in texts)


def azure_embed(texts, client=None, deployment=None, batch_size=AZURE_BATCH, show_progress=False):
    """Embed with the Azure deployment, in batches. Row i is the vector for texts[i]."""
    from app import config

    client = client or config.openai_client().with_options(max_retries=8)  # retry rate limits with backoff
    deployment = deployment or config.embed_deployment()
    vectors = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=deployment, input=texts[start:start + batch_size])
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        if show_progress:
            print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=np.float32)


def azure_encoder(client=None, deployment=None):
    """A function that embeds one query with the Azure deployment, as retrievers expect."""
    return lambda query: azure_embed([query], client, deployment)[0]


def load_or_azure_embed(texts, out_dir, cache_name="azure_parents", client=None):
    """Load the saved Azure vectors, or embed the texts and save them, so the corpus is only paid for once."""
    path = embeddings_path(out_dir, cache_name)
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(texts):
            return vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = azure_embed(texts, client, show_progress=True)
    np.save(path, vectors)
    return vectors
