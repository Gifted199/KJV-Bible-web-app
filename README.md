# KJV RAG

Retrieval-augmented generation (RAG) over the King James Bible. This repo builds the retrieval half: it turns the KJV PDF into clean, searchable passages and measures which embedding model finds the right passage for a question.

## Project structure

```
KJV Rag/
├── data/
│   ├── KJVfile.pdf              # source PDF (not committed)
│   ├── test_questions.json      # 15 questions + the verse that answers each
│   └── ...                      # generated: verses/parents/children .jsonl, embeddings/, faiss/, chroma/
├── app/
│   ├── chunking.py              # PDF -> verses -> parent and child chunks
│   ├── embeddings.py            # MiniLM / mpnet embeddings, cached to data/embeddings/
│   ├── dense_retriever.py       # FAISS and Chroma search, child -> parent lookup
│   ├── evaluation.py            # correct at #1, correct in top 5, MRR
│   └── main.py                  # CLI: python -m app.main <command>
├── tests/test_retrieval.py      # runs without model downloads
├── RAG_Retrieval_walkthrough.ipynb   # the same steps, cell by cell, with explanations and results
├── requirements.txt
└── README.md
```

Planned next: `sparse_retriever.py` (BM25), `hybrid_retriever.py`, `mmr.py` and `reranker.py`.

## Setup

Requires Python 3.14. Put the KJV PDF at `data/KJVfile.pdf`.

With [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Or with pip:

```
pip install -r requirements.txt
```

(With pip, run the commands below without the `uv run` prefix.)

## Usage

```
uv run python -m app.main ingest      # PDF -> data/verses.jsonl, parents.jsonl, children.jsonl
uv run python -m app.main embed       # embed children with both models, build FAISS + Chroma
uv run python -m app.main search "Jonah is swallowed by a big fish" --model mpnet
uv run python -m app.main evaluate    # score every model and store on the test questions
```

`embed` downloads the models on first use and is slow on a CPU (about 6 minutes for MiniLM and 45 minutes for mpnet). The vectors are cached in `data/embeddings/`, so later runs take seconds.

## How it works

1. **Parse.** The PDF's layout is regular (chapter headers, verse numbers, section headings, cross-references), so the text is parsed into 31,102 verses and checked against the known KJV counts (66 books, 1,189 chapters).
2. **Chunk (parent-child).**
   - **Parents** follow the section headings, with at most 10 verses each. They are what the LLM reads.
   - **Children** are windows of 3 verses inside each parent, overlapping by 1 verse. They are what gets embedded and searched.
3. **Embed** the children with `all-MiniLM-L6-v2` and `all-mpnet-base-v2`.
4. **Store** the same vectors in FAISS (exact search) and Chroma (a database with metadata).
5. **Search** the children, then return their parents.

## Results

| Model | Correct at #1 | Correct in top 5 | MRR |
|---|---|---|---|
| all-MiniLM-L6-v2 | 7 / 15 | 14 / 15 | 0.652 |
| **all-mpnet-base-v2** | **9 / 15** | **14 / 15** | **0.719** |

FAISS and Chroma score identically. mpnet is the better model. Both models miss *"For God so loved the world"* (John 3:16): dense search matches meaning rather than exact words, which is what sparse (BM25) and hybrid retrieval are planned to fix. See the notebook for the per-question breakdown.

## Tests

```
uv run pytest
```

The tests use small made-up inputs and fake vectors, so they need no model downloads.
