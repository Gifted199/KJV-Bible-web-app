"""Turn the KJV PDF into verses, then into parent and child chunks.

Parents follow the PDF's section headings (document-structure-based chunking) and are what the LLM reads.
Children are small overlapping windows inside each parent (fixed-size with overlap) and are what gets embedded.
"""

import json
import math
import re
from itertools import groupby
from pathlib import Path

CHAPTER = re.compile(r"^(?P<book>(?:[1-3] )?[A-Z][A-Za-z ]+?) (?P<ch>\d+) *\n+KJV +\[Online\] *$", re.M)  # "Genesis 1 \nKJV  [Online]"
CROSSREF = re.compile(r"^\([^)\n]*\d+:\d+[^)\n]*\) *$", re.M)  # "(John 1:1–5; Hebrews 11:1–3)"
VERSE = re.compile(r"(\d+)\n\u202f")  # verse number, newline, narrow no-break space
BOOK_NAMES = {"Psalm": "Psalms"}  # chapter headers say "Psalm", the contents page says "Psalms"

MAX_VERSES = 10
CHILD_SIZE = 3
CHILD_OVERLAP = 1


def read_pdf_text(path):
    """Extract the PDF's text, skipping the contents page and joining pages so verses split across pages are rejoined."""
    import pymupdf

    with pymupdf.open(path) as document:
        return "\n".join(page.get_text() for page in document[1:])


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_verses(full_text):
    """Parse the extracted text into verses: book, chapter, verse, section heading, text."""
    verses = []
    chapters = list(CHAPTER.finditer(full_text))
    for i, m in enumerate(chapters):
        book = BOOK_NAMES.get(m["book"], m["book"])
        end = chapters[i + 1].start() if i + 1 < len(chapters) else len(full_text)
        body = CROSSREF.sub("", full_text[m.end():end])
        parts = VERSE.split(body)  # [before verse 1, num, text, num, text, ...]
        heading = normalize(parts[0]) or None
        for num, text in zip(parts[1::2], parts[2::2]):
            lines = [line for line in text.split("\n") if line.strip()]
            #lines after the last paragraph end ("  " before the newline) are the next section's heading
            breaks = [j for j, line in enumerate(lines[:-1]) if line.endswith("  ")]
            next_heading = None
            if breaks:
                next_heading = normalize(" ".join(lines[breaks[-1] + 1:]))
                lines = lines[:breaks[-1] + 1]
            verses.append({
                "book": book,
                "chapter": int(m["ch"]),
                "verse": int(num),
                "heading": heading,
                "text": normalize(" ".join(lines)),
            })
            if next_heading:
                heading = next_heading
    return verses


def make_chunk(window, heading):
    """Build a chunk from consecutive verses of one chapter."""
    book, chapter = window[0]["book"], window[0]["chapter"]
    first, last = window[0]["verse"], window[-1]["verse"]
    reference = f"{book} {chapter}:{first}" + (f"-{last}" if last != first else "")
    header = f"{reference} ({heading})" if heading else reference
    body = " ".join(f"{v['verse']} {v['text']}" for v in window)
    return {
        "id": f"{book}-{chapter}-{first}-{last}".replace(" ", "_"),
        "book": book,
        "chapter": chapter,
        "verse_start": first,
        "verse_end": last,
        "heading": heading,
        "reference": reference,
        "text": f"{header}\n{body}",
    }


def build_parents(verses, max_verses=MAX_VERSES):
    """Group consecutive verses that share a section heading, splitting long sections evenly.

    Returns the parents and a dict of parent id -> its verses, used to build the children.
    """
    parents = []
    parent_verses = {}
    for (_, _, heading), group in groupby(verses, key=lambda v: (v["book"], v["chapter"], v["heading"])):
        group = list(group)
        size = math.ceil(len(group) / math.ceil(len(group) / max_verses))
        for start in range(0, len(group), size):
            window = group[start:start + size]
            parent = make_chunk(window, heading)
            parents.append(parent)
            parent_verses[parent["id"]] = window
    return parents, parent_verses


def build_children(parents, parent_verses, size=CHILD_SIZE, overlap=CHILD_OVERLAP):
    """Small overlapping windows inside each parent. Each child belongs to exactly one parent."""
    children = []
    for parent in parents:
        window_verses = parent_verses[parent["id"]]
        #stop before a start whose window would sit entirely inside the previous one
        starts = range(0, max(len(window_verses) - overlap, 1), size - overlap)
        for n, start in enumerate(starts):
            child = make_chunk(window_verses[start:start + size], parent["heading"])
            child["id"] = f"{parent['id']}-c{n}"
            child["parent_id"] = parent["id"]
            children.append(child)
    return children


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ingest(pdf_path, out_dir):
    """Run the whole ingestion: PDF -> verses, parents and children, saved as JSONL in out_dir."""
    out_dir = Path(out_dir)
    verses = parse_verses(read_pdf_text(pdf_path))
    parents, parent_verses = build_parents(verses)
    children = build_children(parents, parent_verses)
    write_jsonl(out_dir / "verses.jsonl", verses)
    write_jsonl(out_dir / "parents.jsonl", parents)
    write_jsonl(out_dir / "children.jsonl", children)
    return verses, parents, children
