"""Tests for chunking, retrieval and evaluation. They use small made-up inputs, so no model is downloaded."""

import numpy as np

from app import chunking
from app.dense_retriever import ChromaRetriever, FaissRetriever, parents_for
from app.evaluation import evaluate, rank_of

#a tiny imitation of the PDF's extracted text, including a chapter header split by a page break
SAMPLE_TEXT = (
    "Genesis 1 \nKJV  [Online] \nThe Creation \n(John 1:1–5; Hebrews 11:1–3) \n"
    "1\n\u202fIn the beginning God created the heaven and the earth. "
    "2\n\u202fAnd the earth was without form, and void; \nand darkness [was] upon the face of the deep.  \n"
    "The First Day \n"
    "3\n\u202fAnd God said, Let there be light: and there was light.  \n \n"
    "Psalm 23 \n\nKJV  [Online] \n"
    "1\n\u202fThe LORD [is] my shepherd; I shall not want.  \n"
)


def make_verses(count, heading="Section", book="Genesis", chapter=1):
    return [
        {"book": book, "chapter": chapter, "verse": n, "heading": heading, "text": f"verse {n}."}
        for n in range(1, count + 1)
    ]


def test_parse_verses_reads_structure_and_removes_headings_and_crossrefs():
    verses = chunking.parse_verses(SAMPLE_TEXT)

    assert [(v["book"], v["chapter"], v["verse"]) for v in verses] == [
        ("Genesis", 1, 1), ("Genesis", 1, 2), ("Genesis", 1, 3), ("Psalms", 23, 1),
    ]
    assert verses[1]["text"] == "And the earth was without form, and void; and darkness [was] upon the face of the deep."
    assert [v["heading"] for v in verses] == ["The Creation", "The Creation", "The First Day", None]
    assert not any("John 1:1" in v["text"] for v in verses)


def test_long_sections_are_split_into_even_parents():
    parents, _ = chunking.build_parents(make_verses(23), max_verses=10)

    assert [(p["verse_start"], p["verse_end"]) for p in parents] == [(1, 8), (9, 16), (17, 23)]


def test_parents_never_cross_a_heading():
    verses = make_verses(4, heading="A") + [dict(v, verse=v["verse"] + 4) for v in make_verses(2, heading="B")]
    parents, _ = chunking.build_parents(verses)

    assert [(p["heading"], p["verse_start"], p["verse_end"]) for p in parents] == [("A", 1, 4), ("B", 5, 6)]


def test_children_overlap_and_cover_every_verse_of_their_parent():
    parents, parent_verses = chunking.build_parents(make_verses(10))
    children = chunking.build_children(parents, parent_verses, size=3, overlap=1)

    assert [(c["verse_start"], c["verse_end"]) for c in children] == [(1, 3), (3, 5), (5, 7), (7, 9), (9, 10)]
    assert {c["parent_id"] for c in children} == {parents[0]["id"]}


def test_short_parent_gets_one_child():
    parents, parent_verses = chunking.build_parents(make_verses(2))
    children = chunking.build_children(parents, parent_verses, size=3, overlap=1)

    assert [(c["verse_start"], c["verse_end"]) for c in children] == [(1, 2)]


def test_parents_for_drops_duplicates_and_keeps_order():
    parents_by_id = {pid: {"id": pid} for pid in ["a", "b", "c"]}
    hits = [{"parent_id": pid} for pid in ["b", "b", "a", "c", "a"]]

    assert [p["id"] for p in parents_for(hits, parents_by_id, k=2)] == ["b", "a"]


def fake_store():
    """Three children with one-hot vectors, and an encoder that maps a word to one of them."""
    parents, parent_verses = chunking.build_parents(
        make_verses(1, heading="A") + [dict(v, verse=2) for v in make_verses(1, heading="B")]
        + [dict(v, verse=3) for v in make_verses(1, heading="C")]
    )
    children = chunking.build_children(parents, parent_verses)
    vectors = np.eye(3, dtype=np.float32)
    words = {"alpha": 0, "beta": 1, "gamma": 2}
    encode = lambda query: vectors[words[query]]
    return children, vectors, encode


def test_faiss_retriever_returns_closest_child_first():
    children, vectors, encode = fake_store()
    retriever = FaissRetriever.build(vectors, children, encode)

    assert retriever.search("beta", k=1)[0]["verse_start"] == 2
    assert len(retriever.search("gamma", k=3)) == 3


def test_chroma_retriever_matches_faiss():
    import chromadb

    children, vectors, encode = fake_store()
    retriever = ChromaRetriever.build(chromadb.EphemeralClient(), "test_children", vectors, children, encode)

    for word in ["alpha", "beta", "gamma"]:
        assert retriever.search(word, k=1)[0]["id"] == FaissRetriever.build(vectors, children, encode).search(word, k=1)[0]["id"]


def test_rank_of_finds_the_parent_containing_the_verse():
    found = [
        {"book": "John", "chapter": 3, "verse_start": 1, "verse_end": 8},
        {"book": "John", "chapter": 3, "verse_start": 9, "verse_end": 21},
    ]

    assert rank_of(found, "John", 3, 16) == 2
    assert rank_of(found, "John", 4, 16) is None


def test_evaluate_scores():
    passage = {"book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16}
    other = {"book": "Genesis", "chapter": 1, "verse_start": 1, "verse_end": 1}
    results = {"first": [passage], "second": [other, passage], "missed": [other]}
    questions = [{"question": q, "book": "John", "chapter": 3, "verse": 16} for q in results]

    scores, ranks = evaluate(lambda q: results[q], questions)

    assert ranks == [1, 2, None]
    assert scores == {"correct at #1": 1, "correct in top 5": 2, "MRR": round((1 + 0.5) / 3, 3)}
