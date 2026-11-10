# KJV RAG

Retrieval-augmented generation (RAG) over the King James Bible. This repo builds the retrieval half:
- it turns the KJV PDF into clean, searchable passages
- it measures ten retrieval techniques on 54 test questions
- it uses the best one, **Hybrid** (dense search + BM25)

## Project structure

```
KJV Rag/
├── data/
│   ├── KJVfile.pdf              # source PDF (not committed)
│   ├── test_questions.json      # 54 questions + the verses that answer each
│   └── ...                      # generated: verses/parents/children .jsonl, embeddings/, faiss/, chroma/
├── app/
│   ├── chunking.py              # PDF -> verses -> parent and child chunks
│   ├── embeddings.py            # MiniLM / mpnet / SPLADE vectors, cached to data/embeddings/
│   ├── dense_retriever.py       # FAISS and Chroma search, parent-child retrieval
│   ├── sparse_retriever.py      # TF-IDF, BM25 and SPLADE
│   ├── hybrid_retriever.py      # reciprocal rank fusion of several retrievers
│   ├── metadata_filter.py       # search only the book / testament a question names
│   ├── mmr.py                   # Maximum Marginal Relevance re-ranking
│   ├── multi_query.py           # several versions of the question, fused
│   ├── contextual.py            # children embedded with a line saying where they sit
│   ├── books.py                 # the 66 books, their testament and group
│   ├── techniques.py            # builds the ten techniques; best_retriever() = Hybrid
│   ├── evaluation.py            # correct at #1, correct in top 5, MRR, coverage@5
│   └── main.py                  # CLI: python -m app.main <command>
├── tests/test_retrieval.py      # runs without model downloads
├── RAG_Retrieval_walkthrough.ipynb   # the same steps, cell by cell, with explanations and results
├── requirements.txt
└── README.md
```

Planned next: `reranker.py`, then generation.

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
uv run python -m app.main embed       # embed with both models, build FAISS + Chroma, and the extra vectors for compare
uv run python -m app.main search "Jonah is swallowed by a big fish"   # Hybrid (add --technique dense for dense only)
uv run python -m app.main evaluate    # score each embedding model and vector store
uv run python -m app.main compare     # score the ten retrieval techniques
```

`embed` downloads the models the first time it runs, and it is slow on a CPU:

| What `embed` builds | Time on a CPU |
|---|---|
| MiniLM child vectors | about 6 minutes |
| mpnet child vectors | about 45 minutes |
| Extra vectors for `compare` (mpnet parents, SPLADE parents, contextual children) | about 3 hours |

Everything is cached in `data/embeddings/`, so later runs take seconds.

## How it works

1. **Parse.** The PDF has a regular layout of chapter headers, verse numbers, section headings and cross-references. The text is parsed into 31,102 verses and checked against the known KJV counts (66 books, 1,189 chapters).
2. **Chunk (parent-child).**
   - **Parents** follow the section headings, with at most 10 verses each. They are what the LLM reads.
   - **Children** are windows of 3 verses inside each parent, overlapping by 1 verse. They are what gets embedded and searched.
3. **Embed** the children with `all-mpnet-base-v2`. It was chosen over `all-MiniLM-L6-v2`; see Results.
4. **Store** the same vectors in FAISS (exact search) and Chroma (a database with metadata).
5. **Search with Hybrid.** Parent-child dense search (search the children, return their parents) and BM25 over the parents run separately. Their rankings are then merged with reciprocal rank fusion.

## Results

All scores are on the 54 questions in `data/test_questions.json`: 16 exact quotes, 20 paraphrases, 12 questions that name a book, and 6 topics. Every answer was checked against the parsed verse text.

- **Correct at #1 / in top 5:** questions where a passage containing an answer verse came first / in the first five.
- **MRR** (mean reciprocal rank): the average of 1 / rank of the first correct passage. 1.0 means always first.
- **Coverage@5:** the share of each question's answer verses that appear in the top 5.

### 1. Embedding model: mpnet

Both models were tested with parent-child dense search in FAISS. Chroma scores almost the same, at most one question different, because its search is approximate.

| Model | Correct at #1 | Correct in top 5 | MRR | Coverage@5 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 33 | 49 | 0.741 | 0.821 |
| **all-mpnet-base-v2** | **38** | **50** | **0.784** | **0.840** |

mpnet is better on every question type, so every dense part of the ten techniques uses it.

### 2. Ten retrieval techniques: Hybrid wins

Every technique returns the same parent passages, so they are compared on equal terms.

| # | Technique | How it works | Correct at #1 | In top 5 | MRR | Coverage@5 |
|---|---|---|---|---|---|---|
| 1 | **Hybrid** | parent-child dense + BM25, fused with reciprocal rank fusion | **42** | **53** | **0.874** | **0.933** |
| 2 | BM25 | shared words, weighted by rarity, with diminishing returns and a length correction | 37 | 51 | 0.799 | 0.873 |
| 3 | MMR | parent-child, re-ranked to avoid near-duplicate passages | 38 | 48 | 0.788 | 0.758 |
| 4 | Metadata filtering | parent-child, only inside the book or testament the question names | 38 | 50 | 0.784 | 0.840 |
| 5 | Parent-child | search the 3-verse children, return their parents | 38 | 50 | 0.784 | 0.840 |
| 6 | Sparse (SPLADE) | a neural model gives each passage weighted keywords, including related words | 36 | 49 | 0.783 | 0.862 |
| 7 | Multi-query | parent-child with 3 versions of the question (as asked, KJV wording, keywords), fused | 37 | 46 | 0.761 | 0.752 |
| 8 | TF-IDF | shared words, weighted by how rare they are | 30 | 47 | 0.687 | 0.783 |
| 9 | Contextual | parent-child, each child embedded with a line saying where it sits in the Bible | 31 | 42 | 0.661 | 0.705 |
| 10 | Dense vector | mpnet vectors of the whole parent passages | 26 | 41 | 0.610 | 0.675 |

MRR by question type:

| Technique | Quote | Paraphrase | Names a book | Topic |
|---|---|---|---|---|
| **Hybrid** | **0.944** | 0.787 | **0.958** | 0.806 |
| BM25 | 0.938 | 0.656 | 0.875 | 0.756 |
| MMR | 0.710 | 0.885 | 0.845 | 0.561 |
| Metadata filtering | 0.709 | 0.877 | 0.808 | 0.625 |
| Parent-child | 0.709 | 0.877 | 0.808 | 0.625 |
| Sparse (SPLADE) | 0.787 | 0.833 | 0.669 | **0.833** |
| Multi-query | 0.654 | **0.887** | 0.713 | 0.724 |
| TF-IDF | 0.776 | 0.599 | 0.745 | 0.625 |
| Contextual | 0.513 | 0.833 | 0.667 | 0.469 |
| Dense vector | 0.444 | 0.713 | 0.667 | 0.597 |

### 3. Combinations of the best techniques

| Combination | Correct at #1 | In top 5 | MRR | Coverage@5 |
|---|---|---|---|---|
| Multi-query + BM25 | 43 | 54 | 0.883 | 0.952 |
| **Hybrid (parent-child + BM25)** | **42** | **53** | **0.874** | **0.933** |
| Metadata filtering + BM25 | 42 | 53 | 0.874 | 0.933 |
| Hybrid + SPLADE | 42 | 54 | 0.866 | 0.962 |
| Metadata filtering + BM25 + SPLADE | 42 | 54 | 0.866 | 0.962 |

### Why Hybrid was chosen

- **It is clearly the best of the ten.** Its MRR is 0.874, against 0.799 for the next best, and it finds a correct passage in the top 5 for 53 of the 54 questions.
- **Its two halves fail on different questions.**
  - BM25 matches exact words, so it is best on quotes (0.938) but weak on paraphrases (0.656).
  - Dense search matches meaning, so it is strong on paraphrases (0.877) but weaker on quotes (0.709).
  - Together, Hybrid is first or close to first on every question type. For example, it finds *"For God so loved the world"* (John 3:16), which dense search alone doesn't place in its top 5.
- **Plain Hybrid was kept over the combinations.**
  - Multi-query + BM25 is ahead by only one question. Its modern-to-KJV word list (e.g. "Holy Spirit" → "Holy Ghost") shares words with the test questions, so that lead may not hold on new questions.
  - Adding SPLADE lowers MRR (0.866) and means running another neural model on every search.
  - Adding metadata filtering changes nothing (see below).
- **It is simple and fast:** one embedding model, one BM25 index, and no LLM.

### Other findings

- **Search small, return big.**
  - Embedding whole 10-verse passages (Dense vector, 0.610) is the weakest technique.
  - Searching 3-verse children and returning their parents (Parent-child, 0.784) is much better, because the extra verses in a long passage dilute its vector.
- **Metadata filtering scores the same as parent-child.**
  - The filter works, but every chunk already starts with its reference (e.g. "Luke 15:11-32"), so dense search already prefers the book a question names.
  - The filter only changes the lower ranks.
- **Contextual retrieval made results worse.**
  - The added "From the book of …" line is much the same for every chunk, which blurs the differences between them.
  - The original method writes the context with an LLM, which might help, but this project uses no LLM.
- **MMR helps paraphrases a little but lowers coverage.** It pushes out the next part of the same story, which often holds the other answer verses.
- **Multi-query and contextual retrieval are no-LLM versions.** Multi-query uses rule-based rewordings. Contextual retrieval builds its context from the book, chapter and section headings.

## Tests

```
uv run pytest
```

The tests use small made-up inputs and fake vectors, so they need no model downloads.
