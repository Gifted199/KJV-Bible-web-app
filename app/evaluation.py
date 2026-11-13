"""Score a search function against test questions whose answer verses are known.

A returned passage is *relevant* if it contains at least one of the question's answer verses. With K = 5 (the
passages an LLM would read):

    precision@K   relevant passages in the top K / K
    recall@K      answer verses found in the top K / all answer verses
    MRR           1 / rank of the first relevant passage (0 if none), averaged over the questions
    nDCG@K        relevant passages in the top K, each worth less the lower it ranks (1 / log2(rank + 1)),
                  divided by the best score possible, so 1.0 = all relevant passages at the very top
"""

import json
import math

K = 5


def parse_reference(reference):
    """'1 Samuel 17:49' -> ('1 Samuel', 17, 49)"""
    book, chapter_verse = reference.rsplit(" ", 1)
    chapter, verse = chapter_verse.split(":")
    return book, int(chapter), int(verse)


def contains(passage, book, chapter, verse):
    return passage["book"] == book and passage["chapter"] == chapter and passage["verse_start"] <= verse <= passage["verse_end"]


def is_relevant(passage, answer_verses):
    return any(contains(passage, *answer) for answer in answer_verses)


def load_test_questions(path, parents):
    """Each question: {"type", "question", "answers": ["Book chapter:verse", ...]}. Any listed answer counts as correct.

    Adds the parsed answer verses, and how many parents contain one (the most relevant passages a search can return).
    """
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        q["answer_verses"] = [parse_reference(a) for a in q["answers"]]
        q["relevant_passages"] = sum(is_relevant(p, q["answer_verses"]) for p in parents)
    return questions


def rank_of(found_passages, answer_verses):
    """1-based position of the first passage that contains any answer verse, or None if none was found."""
    for rank, passage in enumerate(found_passages, start=1):
        if is_relevant(passage, answer_verses):
            return rank
    return None


def precision_at(found_passages, answer_verses, k=K):
    return sum(is_relevant(p, answer_verses) for p in found_passages[:k]) / k


def recall_at(found_passages, answer_verses, k=K):
    """Share of the answer verses that appear somewhere in the first k passages."""
    top = found_passages[:k]
    return sum(any(contains(p, *answer) for p in top) for answer in answer_verses) / len(answer_verses)


def ndcg_at(found_passages, answer_verses, relevant_passages, k=K):
    dcg = sum(1 / math.log2(rank + 1) for rank, p in enumerate(found_passages[:k], start=1) if is_relevant(p, answer_verses))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, relevant_passages) + 1))
    return dcg / ideal


def score(details):
    """Summary scores for a list of per-question results."""
    ranks = [d["rank"] for d in details]
    found = [r for r in ranks if r is not None]
    average = lambda key: round(sum(d[key] for d in details) / len(details), 3)
    return {
        f"precision@{K}": average("precision"),
        f"recall@{K}": average("recall"),
        "MRR": round(sum(1 / r for r in found) / len(ranks), 3),
        f"nDCG@{K}": average("ndcg"),
        "correct at #1": sum(r == 1 for r in found),
        f"correct in top {K}": sum(r <= K for r in found),
    }


def evaluate(search, questions):
    """Run search(question) -> ranked passages for every question.

    Returns the summary scores and one result per question: its type, rank (None = missed), precision, recall and nDCG.
    """
    details = []
    for q in questions:
        found = search(q["question"])
        answers = q["answer_verses"]
        details.append({
            "type": q["type"],
            "rank": rank_of(found, answers),
            "precision": precision_at(found, answers),
            "recall": recall_at(found, answers),
            "ndcg": ndcg_at(found, answers, q["relevant_passages"]),
        })
    return score(details), details
