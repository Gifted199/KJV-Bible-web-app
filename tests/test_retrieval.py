"""Tests for chunking, retrieval and evaluation. They use small made-up inputs, so no model is downloaded."""

import math

import numpy as np
import pytest

from app import chunking
from app.contextual import contextual_texts
from app.dense_retriever import ChromaRetriever, FaissRetriever, parents_for
from app.evaluation import (
    evaluate,
    load_test_questions,
    ndcg_at,
    parse_reference,
    precision_at,
    rank_of,
    recall_at,
)
from app.hybrid_retriever import reciprocal_rank_fusion
from app.metadata_filter import books_named_in
from app.mmr import mmr
from app.multi_query import query_versions
from app.reranker import RerankRetriever
from app.sparse_retriever import BM25Retriever, TfidfRetriever

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


def test_rank_of_finds_the_first_passage_containing_any_answer():
    found = [
        {"book": "John", "chapter": 3, "verse_start": 1, "verse_end": 8},
        {"book": "John", "chapter": 3, "verse_start": 9, "verse_end": 21},
    ]

    assert rank_of(found, [("John", 3, 16)]) == 2
    assert rank_of(found, [("Mark", 1, 1), ("John", 3, 2)]) == 1
    assert rank_of(found, [("John", 4, 16)]) is None


JOHN_3 = {"book": "John", "chapter": 3, "verse_start": 16, "verse_end": 18}
MARK_1 = {"book": "Mark", "chapter": 1, "verse_start": 1, "verse_end": 8}
GENESIS_1 = {"book": "Genesis", "chapter": 1, "verse_start": 1, "verse_end": 5}
ANSWERS = [parse_reference("John 3:16"), parse_reference("John 3:17"), parse_reference("Mark 1:1")]


def test_load_test_questions_counts_the_passages_that_hold_an_answer(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text('[{"type": "quote", "question": "q", "answers": ["John 3:16", "John 3:17", "Mark 1:1"]}]')

    [question] = load_test_questions(path, [JOHN_3, MARK_1, GENESIS_1])

    assert question["answer_verses"] == ANSWERS
    assert question["relevant_passages"] == 2  # John 3:16 and 3:17 share one passage


def test_precision_recall_and_ndcg_at_k():
    found = [GENESIS_1, JOHN_3, MARK_1]

    assert precision_at(found, ANSWERS, k=5) == 2 / 5
    assert recall_at(found, ANSWERS, k=2) == 2 / 3  # John 3:16 and 3:17, not Mark 1:1
    best = 1 + 1 / math.log2(3)
    assert ndcg_at(found, ANSWERS, relevant_passages=2) == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / best)
    assert ndcg_at([JOHN_3, MARK_1], ANSWERS, relevant_passages=2) == 1.0


def test_evaluate_scores():
    results = {"first": [JOHN_3], "second": [GENESIS_1, MARK_1], "missed": [GENESIS_1]}
    questions = [
        {"type": "quote", "question": q, "answer_verses": ANSWERS, "relevant_passages": 2} for q in results
    ]

    scores, details = evaluate(lambda q: results[q], questions)

    assert [d["rank"] for d in details] == [1, 2, None]
    assert scores["MRR"] == round((1 + 0.5) / 3, 3)
    assert scores["precision@5"] == round((1 / 5 + 1 / 5 + 0) / 3, 3)
    assert scores["recall@5"] == round((2 / 3 + 1 / 3 + 0) / 3, 3)
    assert (scores["correct at #1"], scores["correct in top 5"]) == (1, 2)


def passages(*names):
    return [{"id": n, "text": n} for n in names]


def test_bm25_and_tfidf_prefer_passages_with_rare_shared_words():
    items = passages("the lord is my shepherd", "the lord is good", "the fish swallowed jonah")
    texts = [p["text"] for p in items]

    for retriever in [BM25Retriever(items, texts), TfidfRetriever(items, texts)]:
        assert retriever.search("my shepherd")[0]["id"] == "the lord is my shepherd"
        assert retriever.search("jonah and the fish")[0]["id"] == "the fish swallowed jonah"
        assert retriever.search("unknown words") == []


def test_reciprocal_rank_fusion_rewards_items_ranked_well_in_both_lists():
    a, b, c = passages("a", "b", "c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]], k=3)

    assert [item["id"] for item, _ in fused] == ["b", "a", "c"]


def test_mmr_skips_a_near_duplicate():
    relevance = [0.9, 0.89, 0.5]
    similarity = [[1, 0.99, 0], [0.99, 1, 0], [0, 0, 1]]

    assert mmr(relevance, similarity, k=2, lambda_=1.0) == [0, 1]  # plain ranking
    assert mmr(relevance, similarity, k=2, lambda_=0.5) == [0, 2]  # the duplicate makes way


def test_books_named_in_reads_book_and_testament_phrases():
    assert books_named_in("In Psalms, God is our refuge") == {"Psalms"}
    assert books_named_in("in the Gospel of Luke, the shepherds") == {"Luke"}
    assert books_named_in("In 1 John, God is love") == {"1 John"}
    assert books_named_in("in the book of Ruth") == {"Ruth"}
    assert "Matthew" in books_named_in("In the New Testament, the greatest commandment")
    assert "Genesis" not in books_named_in("In the New Testament, the greatest commandment")
    assert books_named_in("Jonah is swallowed by a big fish") is None  # a name alone is not a filter


def test_query_versions_add_kjv_wording_and_keywords():
    versions = query_versions("The Holy Spirit comes and they speak in other languages")

    assert versions[0] == "The Holy Spirit comes and they speak in other languages"
    assert "Holy Ghost" in versions[1] and "tongues" in versions[1]
    assert versions[2] == "Holy Spirit comes speak languages"


def test_contextual_text_says_where_the_child_sits():
    parents, parent_verses = chunking.build_parents(
        make_verses(2, heading="A") + [dict(v, verse=v["verse"] + 2) for v in make_verses(2, heading="B")]
    )
    children = chunking.build_children(parents, parent_verses)
    first, second = contextual_texts(children)

    assert first.startswith("From the book of Genesis (Old Testament, Law), chapter 1 of 1. Section: A. Next section: B.")
    assert "Previous section: A." in second
    assert first.endswith(children[0]["text"])


def test_reranker_reorders_the_base_results_by_the_scorers_score():
    class Base:
        def search(self, query, k):
            return passages("lions den", "great fish", "jonah fled")[:k]

    words_shared = lambda question, texts: [len(set(question.split()) & set(t.split())) for t in texts]
    reranked = RerankRetriever(Base(), words_shared, candidates=3)

    assert [p["id"] for p in reranked.search("jonah and the great fish", k=2)] == ["great fish", "jonah fled"]
    assert [p["id"] for p in RerankRetriever(Base(), words_shared, candidates=1).search("great fish")] == ["lions den"]
