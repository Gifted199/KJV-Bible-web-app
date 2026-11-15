"""Azure AI Search: the `kjv-parents` index, uploading the parents, and searching it.

One document per parent passage (4,723), each with:

    keyword fields   text, reference and heading, scored with BM25, the same idea as the local BM25 retriever
    vector field     the text-embedding-3-small vector (1,536 dimensions, HNSW, cosine similarity). It is searchable
                     but not stored for retrieval, which saves space on the free tier
    filter fields    book, chapter and verse range, for filtering by book later

A hybrid query runs the keyword search and the vector search, then fuses the two rankings with reciprocal rank fusion
(RRF), as the local Hybrid retriever does. Results are the local parent dicts, so evaluation.py scores them unchanged.
"""

import re
import time

INDEX_NAME = "kjv-parents"
VECTOR_FIELD = "vector"
VECTOR_PROFILE = "kjv-hnsw"
UPLOAD_BATCH = 200  # documents per request (~3 MB of JSON), well under the 16 MB request limit
DEPTH = 50  # results taken from each of keyword and vector search before fusing, as in the local Hybrid
VALID_KEY = re.compile(r"^[A-Za-z0-9_\-=]+$")  # the characters Azure allows in a document key
QUERY_OPERATORS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')  # characters with a meaning in the query syntax


def index_definition(name=INDEX_NAME, dimensions=1536):
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchField,
        SearchIndex,
        VectorSearch,
        VectorSearchProfile,
    )

    text = dict(type="Edm.String", searchable=True)
    number = dict(type="Edm.Int32", filterable=True)
    fields = [
        SearchField(name="id", type="Edm.String", key=True, filterable=True),
        SearchField(name="reference", **text),
        SearchField(name="heading", **text),
        SearchField(name="text", **text),
        SearchField(name="book", type="Edm.String", filterable=True, facetable=True),
        SearchField(name="chapter", **number),
        SearchField(name="verse_start", **number),
        SearchField(name="verse_end", **number),
        SearchField(name=VECTOR_FIELD, type="Collection(Edm.Single)", searchable=True, retrievable=False,
                    stored=False, vector_search_dimensions=dimensions, vector_search_profile_name=VECTOR_PROFILE),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name="hnsw")],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


def to_documents(parents, vectors):
    """One search document per parent; row i of vectors belongs to parents[i]."""
    if len(parents) != len(vectors):
        raise ValueError(f"{len(parents)} parents but {len(vectors)} vectors")
    documents = []
    for parent, vector in zip(parents, vectors):
        if not VALID_KEY.match(parent["id"]):
            raise ValueError(f"{parent['id']!r} is not a valid Azure AI Search key")
        documents.append({
            "id": parent["id"],
            "reference": parent["reference"],
            "heading": parent.get("heading") or "",
            "text": parent["text"],
            "book": parent["book"],
            "chapter": parent["chapter"],
            "verse_start": parent["verse_start"],
            "verse_end": parent["verse_end"],
            VECTOR_FIELD: [float(x) for x in vector],
        })
    return documents


def create_index(index_client, name=INDEX_NAME, dimensions=1536):
    """Delete the index if it exists and create it again, so re-running never leaves stale documents."""
    if name in set(index_client.list_index_names()):
        index_client.delete_index(name)
    return index_client.create_index(index_definition(name, dimensions))


def upload(search_client, documents, batch_size=UPLOAD_BATCH):
    """Upload in batches; raises if Azure rejects any document."""
    for start in range(0, len(documents), batch_size):
        results = search_client.upload_documents(documents=documents[start:start + batch_size])
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"{len(failed)} documents were rejected, e.g. {failed[:3]}")


def wait_for_documents(search_client, expected, timeout=120):
    """Uploads become searchable after a short delay; wait until all of them are counted."""
    deadline = time.monotonic() + timeout
    while (count := search_client.get_document_count()) < expected and time.monotonic() < deadline:
        time.sleep(3)
    return count


def storage_report(index_client, name=INDEX_NAME):
    """The index's size and the service's usage against its quota (the free tier's limits), in bytes."""
    stats = index_client.get_index_statistics(name)
    service = index_client.get_service_statistics()
    counters = service["counters"] if isinstance(service, dict) else service.counters

    def counter(key):
        c = counters[key] if isinstance(counters, dict) else getattr(counters, key)
        return (c["usage"], c["quota"]) if isinstance(c, dict) else (c.usage, c.quota)

    get = (lambda k: stats[k]) if isinstance(stats, dict) else (lambda k: getattr(stats, k))
    return {
        "documents": get("document_count"),
        "index storage": get("storage_size"),
        "index vector size": get("vector_index_size"),
        "service storage": counter("storage_size_counter"),
        "service vector index": counter("vector_index_size_counter"),
    }


def clean_query(query):
    """Remove query-syntax characters, so a question is always read as plain words (a hyphen would mean NOT)."""
    return " ".join(QUERY_OPERATORS.sub(" ", query).split())


class AzureSearchRetriever:
    """Search the index by keywords, by vector, or both (hybrid); returns the local parent dicts, best first."""

    def __init__(self, search_client, parents_by_id, encode=None, mode="hybrid", depth=DEPTH):
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError("mode must be hybrid, vector or keyword")
        if mode != "keyword" and encode is None:
            raise ValueError(f"{mode} search needs an encode function for the question")
        self.client = search_client
        self.parents_by_id = parents_by_id
        self.encode = encode
        self.mode = mode
        self.depth = depth

    def search_with_scores(self, query, k=10):
        from azure.search.documents.models import VectorizedQuery

        options = {"top": k, "select": ["id"]}
        if self.mode != "keyword":
            vector = [float(x) for x in self.encode(query)]
            neighbours = max(k, self.depth) if self.mode == "hybrid" else k
            options["vector_queries"] = [VectorizedQuery(vector=vector, k_nearest_neighbors=neighbours,
                                                         fields=VECTOR_FIELD)]
        search_text = clean_query(query) if self.mode != "vector" else None
        results = self.client.search(search_text=search_text, **options)
        return [(self.parents_by_id[r["id"]], r["@search.score"]) for r in results]

    def search(self, query, k=10):
        return [parent for parent, _ in self.search_with_scores(query, k)]
