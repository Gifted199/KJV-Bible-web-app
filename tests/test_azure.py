"""Azure embedding and AI Search code, tested offline with fake clients (no network, no keys)."""

from types import SimpleNamespace

import numpy as np
import pytest

from app import azure_search, embeddings

PARENTS = [
    {"id": "Genesis-4-8-10", "book": "Genesis", "chapter": 4, "verse_start": 8, "verse_end": 10, "heading": "Cain",
     "reference": "Genesis 4:8-10", "text": "Genesis 4:8-10 (Cain)\n9 Am I my brother's keeper?"},
    {"id": "John-11-35-36", "book": "John", "chapter": 11, "verse_start": 35, "verse_end": 36, "heading": None,
     "reference": "John 11:35-36", "text": "John 11:35-36\n35 Jesus wept."},
]
PARENTS_BY_ID = {p["id"]: p for p in PARENTS}


class FakeEmbeddingsClient:
    """Returns a vector per input; returns each batch in reverse order to check results are put back in order."""

    def __init__(self):
        self.batches = []
        self.embeddings = self

    def create(self, model, input):
        self.batches.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(text)), 1.0]) for i, text in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def test_azure_embed_batches_and_keeps_the_order():
    client = FakeEmbeddingsClient()
    vectors = embeddings.azure_embed(["a", "bb", "ccc"], client=client, deployment="emb", batch_size=2)
    assert [len(b) for b in client.batches] == [2, 1]
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert vectors.dtype == np.float32


def test_load_or_azure_embed_pays_only_once(tmp_path):
    client = FakeEmbeddingsClient()
    first = embeddings.load_or_azure_embed(["a", "bb"], tmp_path, client=client)
    second = embeddings.load_or_azure_embed(["a", "bb"], tmp_path, client=client)
    assert len(client.batches) == 1
    assert np.array_equal(first, second)


def test_count_tokens():
    assert embeddings.count_tokens(["Jesus wept."]) == 4


def test_documents_carry_the_fields_and_vector():
    documents = azure_search.to_documents(PARENTS, np.ones((2, 3), dtype=np.float32))
    assert documents[1]["heading"] == ""  # Azure strings can't be None
    assert documents[0]["vector"] == [1.0, 1.0, 1.0]
    assert documents[0]["reference"] == "Genesis 4:8-10"


def test_documents_reject_a_bad_key_or_a_missing_vector():
    with pytest.raises(ValueError):
        azure_search.to_documents([{**PARENTS[0], "id": "Song of Solomon-1-1-2"}], np.ones((1, 3)))
    with pytest.raises(ValueError):
        azure_search.to_documents(PARENTS, np.ones((1, 3)))


def test_index_has_keyword_and_vector_fields():
    index = azure_search.index_definition(dimensions=1536)
    fields = {f.name: f for f in index.fields}
    assert fields["id"].key
    assert fields["text"].searchable
    assert fields["vector"].vector_search_dimensions == 1536
    assert fields["vector"].stored is False  # saves space on the free tier


def test_clean_query_removes_query_syntax():
    assert azure_search.clean_query('Am I my brother\'s keeper? -"Cain"') == "Am I my brother's keeper Cain"


class FakeSearchClient:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def search(self, search_text=None, **options):
        self.calls.append({"search_text": search_text, **options})
        return [{"id": i, "@search.score": 1 / (rank + 1)} for rank, i in enumerate(self.ids[:options["top"]])]


def test_hybrid_sends_words_and_vector_and_returns_local_parents():
    client = FakeSearchClient(["John-11-35-36", "Genesis-4-8-10"])
    retriever = azure_search.AzureSearchRetriever(client, PARENTS_BY_ID, encode=lambda q: np.array([0.5, 0.5]))
    found = retriever.search("Jesus wept?", k=2)
    assert [p["reference"] for p in found] == ["John 11:35-36", "Genesis 4:8-10"]
    call = client.calls[0]
    assert call["search_text"] == "Jesus wept"
    assert call["vector_queries"][0].k_nearest_neighbors == azure_search.DEPTH


def test_keyword_and_vector_modes_send_only_their_part():
    client = FakeSearchClient(["John-11-35-36"])
    azure_search.AzureSearchRetriever(client, PARENTS_BY_ID, mode="keyword").search("wept", 1)
    azure_search.AzureSearchRetriever(client, PARENTS_BY_ID, encode=lambda q: [1.0], mode="vector").search("wept", 1)
    assert "vector_queries" not in client.calls[0]
    assert client.calls[1]["search_text"] is None
    with pytest.raises(ValueError):
        azure_search.AzureSearchRetriever(client, PARENTS_BY_ID, mode="hybrid")  # needs an encoder
