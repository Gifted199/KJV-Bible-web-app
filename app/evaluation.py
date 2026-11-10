"""Score a search function against test questions whose answer verses are known."""

import json


def parse_reference(reference):
    """'1 Samuel 17:49' -> ('1 Samuel', 17, 49)"""
    book, chapter_verse = reference.rsplit(" ", 1)
    chapter, verse = chapter_verse.split(":")
    return book, int(chapter), int(verse)


def load_test_questions(path):
    """Each question: {"type", "question", "answers": ["Book chapter:verse", ...]}. Any listed answer counts as correct."""
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        q["answer_verses"] = [parse_reference(a) for a in q["answers"]]
    return questions


def contains(passage, book, chapter, verse):
    return passage["book"] == book and passage["chapter"] == chapter and passage["verse_start"] <= verse <= passage["verse_end"]


def rank_of(found_passages, answer_verses):
    """1-based position of the first passage that contains any answer verse, or None if none was found."""
    for rank, passage in enumerate(found_passages, start=1):
        if any(contains(passage, *answer) for answer in answer_verses):
            return rank
    return None


def coverage(found_passages, answer_verses, k=5):
    """Share of the answer verses that appear somewhere in the first k passages."""
    top = found_passages[:k]
    return sum(any(contains(p, *answer) for p in top) for answer in answer_verses) / len(answer_verses)


def score(details):
    """Summary scores for a list of per-question results."""
    ranks = [d["rank"] for d in details]
    found = [r for r in ranks if r is not None]
    return {
        "correct at #1": sum(r == 1 for r in found),
        "correct in top 5": sum(r <= 5 for r in found),
        "MRR": round(sum(1 / r for r in found) / len(ranks), 3),
        "coverage@5": round(sum(d["coverage"] for d in details) / len(details), 3),
    }


def evaluate(search, questions):
    """Run search(question) -> ranked passages for every question.

    Returns the summary scores and one result per question: its type, rank (None = missed) and coverage@5.
    """
    details = []
    for q in questions:
        found = search(q["question"])
        details.append({
            "type": q["type"],
            "rank": rank_of(found, q["answer_verses"]),
            "coverage": coverage(found, q["answer_verses"]),
        })
    return score(details), details
