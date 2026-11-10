"""Score a search function against test questions whose answer verse is known."""

import json


def load_test_questions(path):
    """Each question: {"question", "book", "chapter", "verse"}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rank_of(found_parents, book, chapter, verse):
    """1-based position of the first parent that contains the expected verse, or None if it wasn't found."""
    for rank, p in enumerate(found_parents, start=1):
        if p["book"] == book and p["chapter"] == chapter and p["verse_start"] <= verse <= p["verse_end"]:
            return rank
    return None


def evaluate(search, questions):
    """Run search(question) -> ranked parents for every question.

    Returns the scores and the rank for each question (None = missed).
    """
    ranks = [rank_of(search(q["question"]), q["book"], q["chapter"], q["verse"]) for q in questions]
    found = [r for r in ranks if r is not None]
    scores = {
        "correct at #1": sum(r == 1 for r in found),
        "correct in top 5": sum(r <= 5 for r in found),
        "MRR": round(sum(1 / r for r in found) / len(questions), 3),
    }
    return scores, ranks
