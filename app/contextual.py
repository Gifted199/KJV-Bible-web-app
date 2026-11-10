"""Contextual retrieval without an LLM: prefix each child with where it sits in the Bible before embedding.

Anthropic's contextual retrieval asks an LLM to write a short context for every chunk. Here the context is
built only from the data's own structure (testament, book group, chapter, neighbouring section headings),
so it costs nothing and cannot leak knowledge of the test questions.
"""

from collections import defaultdict

from app.books import BOOK_GROUP, testament


def build_contexts(children):
    """child id -> a one-line description of where the child sits."""
    chapters_in_book = defaultdict(int)
    headings_in_chapter = defaultdict(list)
    for c in children:
        chapters_in_book[c["book"]] = max(chapters_in_book[c["book"]], c["chapter"])
        headings = headings_in_chapter[(c["book"], c["chapter"])]
        if c["heading"] and c["heading"] not in headings:
            headings.append(c["heading"])

    contexts = {}
    for c in children:
        book, chapter = c["book"], c["chapter"]
        parts = [
            f"From the book of {book} ({testament(book)}, {BOOK_GROUP[book]}), "
            f"chapter {chapter} of {chapters_in_book[book]}."
        ]
        headings = headings_in_chapter[(book, chapter)]
        if c["heading"] in headings:
            i = headings.index(c["heading"])
            parts.append(f"Section: {c['heading']}.")
            if i > 0:
                parts.append(f"Previous section: {headings[i - 1]}.")
            if i + 1 < len(headings):
                parts.append(f"Next section: {headings[i + 1]}.")
        contexts[c["id"]] = " ".join(parts)
    return contexts


def contextual_texts(children):
    """Each child's text with its context in front, ready to embed."""
    contexts = build_contexts(children)
    return [f"{contexts[c['id']]}\n{c['text']}" for c in children]
