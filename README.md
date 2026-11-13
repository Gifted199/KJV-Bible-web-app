# KJV RAG

Retrieval-augmented generation (RAG) over the King James Bible. This repo builds the retrieval half:
- it turns the KJV PDF into clean, searchable passages
- it measures ten retrieval techniques on 54 test questions and picks the best one, **Hybrid** (dense search + BM25)
- it measures three rerankers on top of Hybrid and picks the best one, the **bge cross-encoder**

The final search is **Hybrid + bge reranking**: the right passage comes first for 49 of the 54 test questions and is in the top 5 for all 54.

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
│   ├── reranker.py              # bi-encoder and cross-encoder rerankers
│   ├── techniques.py            # builds the ten techniques; best_retriever() = Hybrid + bge reranking
│   ├── evaluation.py            # precision@5, recall@5, MRR, nDCG@5
│   └── main.py                  # CLI: python -m app.main <command>
├── tests/test_retrieval.py      # runs without model downloads
├── RAG_Retrieval_walkthrough.ipynb   # the same steps, cell by cell, with explanations and results
├── requirements.txt
└── README.md
```

Planned next: generation (an LLM writes an answer from the passages and cites the verses).

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
uv run python -m app.main search "Jonah is swallowed by a big fish"   # Hybrid + bge reranking
uv run python -m app.main evaluate    # score each embedding model and vector store
uv run python -m app.main compare     # score the ten retrieval techniques
uv run python -m app.main rerank      # score Hybrid with each reranker
```

`search` takes `--technique rerank` (the default, most accurate, about 3 seconds per question on a CPU), `--technique hybrid` (about 0.05 seconds) or `--technique dense`. The first search also loads the models, which takes about 30 seconds.

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
6. **Rerank.** The `BAAI/bge-reranker-base` cross-encoder reads the question together with each of Hybrid's top 10 passages, and puts them in a new order.

## Results

All scores are on the 54 questions in `data/test_questions.json`: 16 exact quotes, 20 paraphrases, 12 questions that name a book, and 6 topics. Every answer was checked against the parsed verse text.

A returned passage is **relevant** if it contains at least one answer verse. Four metrics are measured over the top 5 passages, the ones an LLM would read:

| Metric | What it measures | Best possible here |
|---|---|---|
| **Precision@5** | Relevant passages in the top 5, divided by 5 | **0.337**, not 1.0: 37 of the 54 questions have only one relevant passage in the whole Bible, so even a perfect search scores 1/5 on them |
| **Recall@5** | The share of a question's answer verses found in the top 5 | 1.0 |
| **MRR** (mean reciprocal rank) | 1 / the rank of the first relevant passage (1st = 1, 2nd = 0.5, a miss = 0), averaged | 1.0 |
| **nDCG@5** | Every relevant passage in the top 5 counts, but less the lower it ranks (1 / log2(rank + 1)), compared with the best order possible | 1.0 |

Each table also counts the questions where a relevant passage came **first** and where one appeared **in the top 5**.

The tables in sections 1 and 3 were measured before precision@5 and nDCG@5 were added. Their **coverage@5** column is the same measure as recall@5.

### 1. Embedding model: mpnet

Both models were tested with parent-child dense search in FAISS. Chroma scores almost the same, at most one question different, because its search is approximate.

| Model | Correct at #1 | Correct in top 5 | MRR | Coverage@5 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 33 | 49 | 0.741 | 0.821 |
| **all-mpnet-base-v2** | **38** | **50** | **0.784** | **0.840** |

mpnet is better on every question type, so every dense part of the ten techniques uses it.

### 2. Ten retrieval techniques: Hybrid wins

Every technique returns the same parent passages, so they are compared on equal terms.

| # | Technique | How it works |
|---|---|---|
| 1 | **Hybrid** | parent-child dense + BM25, fused with reciprocal rank fusion |
| 2 | BM25 | shared words, weighted by rarity, with diminishing returns and a length correction |
| 3 | MMR | parent-child, re-ranked to avoid near-duplicate passages |
| 4 | Metadata filtering | parent-child, only inside the book or testament the question names |
| 5 | Parent-child | search the 3-verse children, return their parents |
| 6 | Sparse (SPLADE) | a neural model gives each passage weighted keywords, including related words |
| 7 | Multi-query | parent-child with 3 versions of the question (as asked, KJV wording, keywords), fused |
| 8 | TF-IDF | shared words, weighted by how rare they are |
| 9 | Contextual | parent-child, each child embedded with a line saying where it sits in the Bible |
| 10 | Dense vector | mpnet vectors of the whole parent passages |

| # | Technique | Precision@5 | Recall@5 | MRR | nDCG@5 | Correct at #1 | In top 5 |
|---|---|---|---|---|---|---|---|
| 1 | **Hybrid** | **0.296** | **0.933** | **0.874** | **0.865** | **42** | **53** |
| 2 | BM25 | 0.274 | 0.873 | 0.799 | 0.786 | 37 | 51 |
| 3 | MMR | 0.215 | 0.758 | 0.788 | 0.714 | 38 | 48 |
| 4 | Metadata filtering | 0.256 | 0.840 | 0.784 | 0.762 | 38 | 50 |
| 5 | Parent-child | 0.256 | 0.840 | 0.784 | 0.760 | 38 | 50 |
| 6 | Sparse (SPLADE) | 0.281 | 0.862 | 0.783 | 0.785 | 36 | 49 |
| 7 | Multi-query | 0.233 | 0.752 | 0.761 | 0.701 | 37 | 46 |
| 8 | TF-IDF | 0.241 | 0.783 | 0.687 | 0.664 | 30 | 47 |
| 9 | Contextual | 0.215 | 0.705 | 0.661 | 0.644 | 31 | 42 |
| 10 | Dense vector | 0.215 | 0.675 | 0.610 | 0.599 | 26 | 41 |

Hybrid is first on all four metrics.

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

### 4. Reranking: the bge cross-encoder wins

Hybrid is fast and finds good candidates. A **reranker** re-reads only the top candidates, so it can be slower and more careful:

| Type | How it scores a passage | Model tested |
|---|---|---|
| **Bi-encoder** | Embeds the question and the passage **separately**, then compares the two vectors | `multi-qa-mpnet-base-cos-v1` (trained on question → answer pairs) |
| **Cross-encoder** | Reads the question and the passage **together**, and outputs one relevance score | `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, fast) and `BAAI/bge-reranker-base` (larger) |

Each reranker re-orders Hybrid's top 10 passages:

| | Precision@5 | Recall@5 | MRR | nDCG@5 | Correct at #1 | In top 5 | Seconds per question (CPU) |
|---|---|---|---|---|---|---|---|
| **Hybrid + cross-encoder (bge)** | **0.307** | **0.963** | **0.945** | **0.930** | **49** | **54** | ~3.3 |
| Hybrid + cross-encoder (MiniLM) | 0.289 | 0.907 | 0.886 | 0.862 | 44 | 52 | ~0.7 |
| Hybrid (no reranker) | 0.296 | 0.933 | 0.874 | 0.865 | 42 | 53 | ~0.05 |
| Hybrid + bi-encoder | 0.248 | 0.764 | 0.720 | 0.684 | 33 | 45 | ~4.1 |

MRR by question type:

| | Quote | Paraphrase | Names a book | Topic |
|---|---|---|---|---|
| **Hybrid + bge** | **1.000** | **0.910** | 0.903 | **1.000** |
| Hybrid + MiniLM | 0.882 | 0.820 | **1.000** | 0.889 |
| Hybrid (no reranker) | 0.944 | 0.787 | 0.958 | 0.806 |
| Hybrid + bi-encoder | 0.607 | 0.774 | 0.743 | 0.792 |

### Why the bge cross-encoder was chosen

- **It is the best on all four metrics.** MRR rises from 0.874 to 0.945 and nDCG@5 from 0.865 to 0.930. The right passage comes first for 49 of the 54 questions instead of 42, and is in the top 5 for all 54.
- **Every quote and every topic question now has a right passage first** (MRR 1.000).
- **Its precision@5, 0.307, is close to the best possible, 0.337.**
- **The cost is accepted.** It takes about 3 seconds per question on a CPU, against 0.05 seconds for Hybrid alone. The answer-writing step (the LLM) will take a few seconds anyway, and on a GPU bge takes a fraction of a second.

**Two settings make it about 3 times faster with no loss in accuracy:**
- **Re-scoring the top 10 instead of the top 20** gives exactly the same scores, because every passage bge moved up was already in Hybrid's top 10. It halves the time.
- **Scoring one passage at a time** is about 30% faster on a CPU. A batch is padded to the length of its longest passage, which wastes work.

**Why not the others:**
- **The MiniLM cross-encoder** gets two more questions right at #1, but its precision, recall and nDCG@5 are lower or the same, and it is worse on quotes (0.882 vs 0.944).
- **The bi-encoder makes every metric worse.** It drops Hybrid's ordering, including BM25's exact word matching, and replaces it with one vector comparison, which is the weakest kind of search on this text. Quotes fall from 0.944 to 0.607.

## Tests

```
uv run pytest
```

The tests use small made-up inputs and fake vectors, so they need no model downloads.
