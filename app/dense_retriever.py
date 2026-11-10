"""Dense (vector) retrieval over the children, in FAISS or Chroma, returning their parents.

Retrievers take an `encode` function (query -> vector), so tests can pass fake vectors instead of a real model.
"""

from pathlib import Path

import numpy as np

CHROMA_BATCH_SIZE = 5000  # Chroma limits how many rows one add() call can take


def parents_for(child_hits, parents_by_id, k=5):
    """Map ranked child hits to their parents, dropping duplicates and keeping rank order."""
    parent_ids = []
    for child in child_hits:
        if child["parent_id"] not in parent_ids:
            parent_ids.append(child["parent_id"])
    return [parents_by_id[pid] for pid in parent_ids[:k]]


class ParentChildRetriever:
    """Search the small children, return their parents. A parent's score is its best child's score."""

    def __init__(self, child_retriever, parents_by_id, child_k=50):
        self.child_retriever = child_retriever
        self.parents_by_id = parents_by_id
        self.child_k = child_k  # children fetched per search, so that k distinct parents are usually found

    def search_with_scores(self, query, k=10):
        best = {}
        for child, child_score in self.child_retriever.search_with_scores(query, self.child_k):
            best.setdefault(child["parent_id"], child_score)  # children arrive best first
        return [(self.parents_by_id[pid], s) for pid, s in list(best.items())[:k]]

    def search(self, query, k=10):
        return [parent for parent, _ in self.search_with_scores(query, k)]


class FaissRetriever:
    """An exact search index: compares the query with every item (children, or parents for plain dense search)."""

    def __init__(self, index, children, encode):
        self.index = index
        self.children = children
        self.encode = encode

    @classmethod
    def build(cls, vectors, children, encode):
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors = cosine similarity
        index.add(np.asarray(vectors, dtype=np.float32))
        return cls(index, children, encode)

    @classmethod
    def load(cls, path, children, encode):
        import faiss

        return cls(faiss.read_index(str(path)), children, encode)

    def save(self, path):
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def search_with_scores(self, query, k=10):
        """The k items closest to the query, best first, with their cosine similarities."""
        query_vector = np.asarray([self.encode(query)], dtype=np.float32)
        scores, positions = self.index.search(query_vector, k)
        return [(self.children[i], float(s)) for i, s in zip(positions[0], scores[0]) if i != -1]

    def search(self, query, k=10):
        """The k items closest to the query, best first."""
        return [item for item, _ in self.search_with_scores(query, k)]


class ChromaRetriever:
    """A database that stores each child's vector, text and metadata (approximate HNSW search)."""

    def __init__(self, collection, children, encode):
        self.collection = collection
        self.children_by_id = {c["id"]: c for c in children}
        self.encode = encode

    @classmethod
    def build(cls, client, name, vectors, children, encode):
        #start fresh so rebuilding doesn't add every child twice
        client.get_or_create_collection(name)
        client.delete_collection(name)
        #embedding_function=None: Chroma stores our vectors and never embeds anything itself
        collection = client.create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        for start in range(0, len(children), CHROMA_BATCH_SIZE):
            batch = children[start:start + CHROMA_BATCH_SIZE]
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=np.asarray(vectors[start:start + CHROMA_BATCH_SIZE], dtype=np.float32),
                documents=[c["text"] for c in batch],
                metadatas=[{
                    "parent_id": c["parent_id"],
                    "book": c["book"],
                    "chapter": c["chapter"],
                    "verse_start": c["verse_start"],
                    "verse_end": c["verse_end"],
                    "heading": c["heading"] or "",  # Chroma doesn't accept None
                } for c in batch],
            )
        return cls(collection, children, encode)

    @classmethod
    def load(cls, client, name, children, encode):
        return cls(client.get_collection(name, embedding_function=None), children, encode)

    def search(self, query, k=10):
        """The k children closest to the query, best first."""
        result = self.collection.query(query_embeddings=[self.encode(query)], n_results=k)
        return [self.children_by_id[i] for i in result["ids"][0]]


def chroma_collection_name(model_name):
    return f"kjv_children_{model_name}"


def faiss_index_path(out_dir, model_name):
    return Path(out_dir) / "faiss" / f"{model_name}.index"
