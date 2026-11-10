"""Build the ten retrieval techniques over the same data, so they can be compared on equal terms.

Every technique returns ranked parents (passages of up to 10 verses), and every dense part uses the same model.
Hybrid, metadata filtering, MMR, multi-query and contextual retrieval are built on top of parent-child
retrieval, so comparing them with parent-child shows what each one adds.
"""

from app import embeddings
from app.chunking import read_jsonl
from app.contextual import contextual_texts
from app.dense_retriever import FaissRetriever, ParentChildRetriever, faiss_index_path
from app.hybrid_retriever import HybridRetriever
from app.metadata_filter import MetadataFilterRetriever
from app.mmr import MMRRetriever
from app.multi_query import MultiQueryRetriever
from app.sparse_retriever import BM25Retriever, SpladeRetriever, TfidfRetriever

MODEL = "mpnet"  # the best embedding model in the MiniLM vs mpnet comparison

DESCRIPTIONS = {
    "Dense vector": "mpnet vectors of the parents themselves; find the closest passages by meaning",
    "Sparse (SPLADE)": "a neural model gives each passage weighted keywords, including related words",
    "TF-IDF": "shared words, weighted by how rare they are",
    "BM25": "shared words, like TF-IDF but with diminishing returns and a length correction",
    "Hybrid": "parent-child dense + BM25, fused with reciprocal rank fusion",
    "Metadata filtering": "parent-child dense, only inside the book or testament the question names",
    "MMR": "parent-child dense, re-ranked to avoid near-duplicate passages",
    "Parent-child": "search the 3-verse children with mpnet, return their parents",
    "Multi-query": "parent-child dense with 3 versions of the question (as asked, KJV wording, keywords), fused",
    "Contextual": "parent-child dense, but each child is embedded with a line saying where it sits in the Bible",
}


def best_retriever(data_dir="data"):
    """The winner of the comparison: Hybrid (parent-child dense + BM25), loading the saved mpnet FAISS index."""
    children = read_jsonl(f"{data_dir}/children.jsonl")
    parents = read_jsonl(f"{data_dir}/parents.jsonl")
    parents_by_id = {p["id"]: p for p in parents}
    child_retriever = FaissRetriever.load(faiss_index_path(data_dir, MODEL), children, embeddings.encoder(MODEL))
    parent_child = ParentChildRetriever(child_retriever, parents_by_id)
    return HybridRetriever([parent_child, BM25Retriever(parents, [p["text"] for p in parents])])


def build_techniques(data_dir="data"):
    """name -> retriever with search(query, k) returning ranked parents."""
    children = read_jsonl(f"{data_dir}/children.jsonl")
    parents = read_jsonl(f"{data_dir}/parents.jsonl")
    parents_by_id = {p["id"]: p for p in parents}
    parent_texts = [p["text"] for p in parents]
    encode = embeddings.encoder(MODEL)

    #all cached by `python -m app.main embed`; these only load files
    child_vectors = embeddings.load_or_embed(MODEL, [c["text"] for c in children], data_dir)
    parent_vectors = embeddings.load_or_embed(MODEL, parent_texts, data_dir, "mpnet_parents")
    contextual_vectors = embeddings.load_or_embed(MODEL, contextual_texts(children), data_dir, "mpnet_contextual")
    splade_vectors = embeddings.load_or_sparse_embed(parent_texts, data_dir)

    parent_child = ParentChildRetriever(FaissRetriever.build(child_vectors, children, encode), parents_by_id)
    bm25 = BM25Retriever(parents, parent_texts)

    return {
        "Dense vector": FaissRetriever.build(parent_vectors, parents, encode),
        "Sparse (SPLADE)": SpladeRetriever(parents, splade_vectors, embeddings.sparse_embed_query),
        "TF-IDF": TfidfRetriever(parents, parent_texts),
        "BM25": bm25,
        "Hybrid": HybridRetriever([parent_child, bm25]),
        "Metadata filtering": MetadataFilterRetriever(child_vectors, children, parents_by_id, encode),
        "MMR": MMRRetriever(parent_child, {p["id"]: v for p, v in zip(parents, parent_vectors)}),
        "Parent-child": parent_child,
        "Multi-query": MultiQueryRetriever(parent_child),
        "Contextual": ParentChildRetriever(FaissRetriever.build(contextual_vectors, children, encode), parents_by_id),
    }
