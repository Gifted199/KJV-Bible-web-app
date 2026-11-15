"""Command line: python -m app.main <command>

    ingest                       PDF -> data/verses.jsonl, parents.jsonl, children.jsonl
    embed    [--model all]       embed the children, build the FAISS index and Chroma collection
    search   "question"          search with Hybrid + bge reranking (the most accurate) and print the best passages
    evaluate [--model all]       score each model and store on data/test_questions.json
    compare                      score the ten retrieval techniques on data/test_questions.json
    rerank                       score Hybrid with a bi-encoder and two cross-encoder rerankers
    check                        test the connection to Azure OpenAI, AI Search and Langfuse (prints no secrets)
    azure-index                  embed the parents with text-embedding-3-small and upload them to Azure AI Search
    azure-storage                show the index size against the free-tier limits
    azure-compare                score Azure AI Search hybrid, with and without bge, beside the local Hybrid + bge
    azure-depth                  score AI Search hybrid + bge re-scoring the top 10, 20 or 30 candidates
    abstain-scores               best bge score of answerable vs unanswerable questions (sets the abstain threshold)
    ask      "question" ...      answer with verse citations (gpt-5.4-mini) or abstain; a batch needs --yes
"""

import argparse
import sys
import time

from app import chunking, embeddings, evaluation
from app.contextual import contextual_texts
from app.dense_retriever import (
    ChromaRetriever,
    FaissRetriever,
    chroma_collection_name,
    faiss_index_path,
    parents_for,
)

DATA_DIR = "data"
EMBED_PRICE_PER_MILLION = 0.02  # text-embedding-3-small list price, US$ per million tokens
STORES = ["faiss", "chroma"]


def chroma_client():
    import chromadb

    return chromadb.PersistentClient(path=f"{DATA_DIR}/chroma")


def load_chunks():
    children = chunking.read_jsonl(f"{DATA_DIR}/children.jsonl")
    parents = chunking.read_jsonl(f"{DATA_DIR}/parents.jsonl")
    return children, {p["id"]: p for p in parents}


def load_questions(parents_by_id):
    return evaluation.load_test_questions(f"{DATA_DIR}/test_questions.json", parents_by_id.values())


def load_retriever(store, model_name, children):
    encode = embeddings.encoder(model_name)
    if store == "faiss":
        return FaissRetriever.load(faiss_index_path(DATA_DIR, model_name), children, encode)
    return ChromaRetriever.load(chroma_client(), chroma_collection_name(model_name), children, encode)


def chosen(value, options):
    return options if value == "all" else [value]


def cmd_ingest(args):
    verses, parents, children = chunking.ingest(args.pdf, DATA_DIR)
    print(f"{len(verses)} verses -> {len(parents)} parents, {len(children)} children")


def cmd_embed(args):
    children, parents_by_id = load_chunks()
    texts = [c["text"] for c in children]
    for name in chosen(args.model, list(embeddings.MODELS)):
        start = time.perf_counter()
        vectors = embeddings.load_or_embed(name, texts, DATA_DIR)
        print(f"{name}: {vectors.shape[0]} vectors of size {vectors.shape[1]} ({time.perf_counter() - start:.0f}s)")
        encode = embeddings.encoder(name)
        FaissRetriever.build(vectors, children, encode).save(faiss_index_path(DATA_DIR, name))
        ChromaRetriever.build(chroma_client(), chroma_collection_name(name), vectors, children, encode)
        print(f"{name}: FAISS index and Chroma collection saved")

    #extra vectors that the retrieval-technique comparison needs, built once and cached
    parent_texts = [p["text"] for p in parents_by_id.values()]
    extras = {
        "mpnet parents": lambda: embeddings.load_or_embed("mpnet", parent_texts, DATA_DIR, "mpnet_parents"),
        "SPLADE parents": lambda: embeddings.load_or_sparse_embed(parent_texts, DATA_DIR),
        "mpnet contextual children": lambda: embeddings.load_or_embed(
            "mpnet", contextual_texts(children), DATA_DIR, "mpnet_contextual"
        ),
    }
    for label, build in extras.items():
        start = time.perf_counter()
        vectors = build()
        print(f"{label}: {vectors.shape[0]} vectors ({time.perf_counter() - start:.0f}s)")


def cmd_search(args):
    if args.technique == "rerank":
        from app.techniques import best_retriever

        found = best_retriever(DATA_DIR).search(args.question, args.k)
    elif args.technique == "hybrid":
        from app.techniques import build_hybrid

        found = build_hybrid(DATA_DIR).search(args.question, args.k)
    else:
        children, parents_by_id = load_chunks()
        retriever = load_retriever(args.store, args.model, children)
        found = parents_for(retriever.search(args.question), parents_by_id, k=args.k)
    for rank, parent in enumerate(found, start=1):
        print(f"\n#{rank} {parent['text']}")


def cmd_evaluate(args):
    children, parents_by_id = load_chunks()
    questions = load_questions(parents_by_id)
    print(f"{len(questions)} test questions")
    for store in chosen(args.store, STORES):
        for name in chosen(args.model, list(embeddings.MODELS)):
            retriever = load_retriever(store, name, children)
            scores, _ = evaluation.evaluate(lambda q: parents_for(retriever.search(q, 50), parents_by_id, k=10), questions)
            print(f"{store:7} {name:7} " + "  ".join(f"{k}: {v}" for k, v in scores.items()))


def compare_techniques(retrievers, questions):
    """Score every technique; returns (overall table, MRR-by-question-type table), best first.

    "s/question" is the average search time, measured after one warm-up search (which loads any model).
    """
    import pandas as pd

    rows, by_type = [], {}
    for name, retriever in retrievers.items():
        retriever.search(questions[0]["question"], 10)
        start = time.perf_counter()
        scores, details = evaluation.evaluate(lambda q: retriever.search(q, 10), questions)
        seconds = (time.perf_counter() - start) / len(questions)
        rows.append({"technique": name, **scores, "s/question": round(seconds, 2)})
        by_type[name] = {
            t: evaluation.score([d for d in details if d["type"] == t])["MRR"]
            for t in dict.fromkeys(d["type"] for d in details)
        }
    overall = pd.DataFrame(rows).sort_values(["MRR", "nDCG@5"], ascending=False).reset_index(drop=True)
    overall.index += 1
    return overall, pd.DataFrame(by_type).T.loc[overall["technique"]]


def cmd_compare(args):
    import pandas as pd

    from app.techniques import build_techniques

    _, parents_by_id = load_chunks()
    questions = load_questions(parents_by_id)
    overall, by_type = compare_techniques(build_techniques(DATA_DIR), questions)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"{len(questions)} test questions\n\n{overall}\n\nMRR by question type:\n{by_type}")


def cmd_rerank(args):
    import pandas as pd

    from app.reranker import build_rerankers
    from app.techniques import build_hybrid

    _, parents_by_id = load_chunks()
    questions = load_questions(parents_by_id)
    overall, by_type = compare_techniques(build_rerankers(build_hybrid(DATA_DIR), args.candidates), questions)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"{len(questions)} test questions, top {args.candidates} Hybrid results re-ranked\n\n{overall}"
          f"\n\nMRR by question type:\n{by_type}")


def cmd_check(args):
    from app.config import check_connections

    failures = 0
    for service, ok, message in check_connections():
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {service:10} {message}")
    if failures:
        raise SystemExit(1)


def cmd_azure_index(args):
    from app import azure_search, config

    _, parents_by_id = load_chunks()
    parents = list(parents_by_id.values())
    texts = [p["text"] for p in parents]
    tokens = embeddings.count_tokens(texts)
    print(f"{len(texts)} parents, {tokens:,} tokens, about ${tokens / 1e6 * EMBED_PRICE_PER_MILLION:.4f} to embed")
    start = time.perf_counter()
    vectors = embeddings.load_or_azure_embed(texts, DATA_DIR)
    print(f"vectors: {vectors.shape[0]} x {vectors.shape[1]} ({time.perf_counter() - start:.0f}s, cached in "
          f"{embeddings.embeddings_path(DATA_DIR, 'azure_parents')})")

    name = config.search_index_name()
    azure_search.create_index(config.search_index_client(), name, vectors.shape[1])
    documents = azure_search.to_documents(parents, vectors)
    start = time.perf_counter()
    azure_search.upload(config.search_client(), documents)
    count = azure_search.wait_for_documents(config.search_client(), len(documents))
    print(f"index '{name}': {count} documents searchable ({time.perf_counter() - start:.0f}s)")
    print_storage(azure_search.storage_report(config.search_index_client(), name))


def print_storage(report):
    mb = lambda n: f"{n / 1e6:.1f} MB"
    print(f"documents: {report['documents']}")
    print(f"index storage: {mb(report['index storage'])} (of which vector index {mb(report['index vector size'])})")
    for label in ("service storage", "service vector index"):
        used, quota = report[label]
        print(f"{label}: {mb(used)} used of {mb(quota)} ({used / quota:.0%})" if quota else f"{label}: {mb(used)}")
    print("(index statistics can lag a few minutes behind an upload; run `azure-storage` again later to refresh)")


def cmd_azure_storage(args):
    from app import azure_search, config

    print_storage(azure_search.storage_report(config.search_index_client(), config.search_index_name()))


def cmd_azure_compare(args):
    import pandas as pd

    from app import config
    from app.azure_search import AzureSearchRetriever
    from app.reranker import CROSS_ENCODERS, RerankRetriever, cross_encoder_scorer
    from app.techniques import build_hybrid

    _, parents_by_id = load_chunks()
    questions = load_questions(parents_by_id)
    bge = cross_encoder_scorer(CROSS_ENCODERS["bge"])
    local = build_hybrid(DATA_DIR)
    azure = AzureSearchRetriever(config.search_client(), parents_by_id, embeddings.azure_encoder())
    retrievers = {
        "Local Hybrid (mpnet + BM25)": local,
        "Local Hybrid + bge": RerankRetriever(local, bge),
        "AI Search hybrid": azure,
        "AI Search hybrid + bge": RerankRetriever(azure, bge),
    }
    overall, by_type = compare_techniques(retrievers, questions)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"{len(questions)} test questions\n\n{overall}\n\nMRR by question type:\n{by_type}")


def cmd_azure_depth(args):
    """How many AI Search candidates should bge re-score? Questions are embedded once and then read from the cache."""
    import pandas as pd

    from app.reranker import CANDIDATES
    from app.techniques import best_azure_retriever

    _, parents_by_id = load_chunks()
    questions = load_questions(parents_by_id)
    encode = embeddings.azure_query_encoder(DATA_DIR)
    encode.prefetch([q["question"] for q in questions])
    depths = sorted(set(args.depths) | {CANDIDATES})
    retrievers = {f"top {d}" + (" (current)" if d == CANDIDATES else ""): best_azure_retriever(DATA_DIR, encode, d)
                  for d in depths}
    overall, _ = compare_techniques(retrievers, questions)
    overall = overall.set_index("technique").loc[list(retrievers)]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"{len(questions)} test questions, AI Search hybrid + bge; question embedding requests this run: "
          f"{encode.calls}\n\n{overall[['MRR', 'correct at #1', 'correct in top 5', 's/question']]}")
    current = overall["MRR"].iloc[depths.index(CANDIDATES)]
    smallest = next(d for d, mrr in zip(depths, overall["MRR"]) if mrr >= current - 0.01)
    print(f"\nsmallest depth with MRR within 0.01 of the current setting (top {CANDIDATES}): top {smallest}")


def load_unanswerable():
    import json

    with open(f"{DATA_DIR}/unanswerable_questions.json", encoding="utf-8") as f:
        return json.load(f)


def cmd_abstain_scores(args):
    """The best bge score for every answerable and unanswerable question: the evidence for ABSTAIN_THRESHOLD."""
    import numpy as np

    from app.generation import ABSTAIN_THRESHOLD
    from app.techniques import best_azure_retriever

    _, parents_by_id = load_chunks()
    answerable = load_questions(parents_by_id)
    unanswerable = load_unanswerable()
    encode = embeddings.azure_query_encoder(DATA_DIR)
    encode.prefetch([q["question"] for q in answerable + unanswerable])
    retriever = best_azure_retriever(DATA_DIR, encode)
    top = lambda q: retriever.search_with_scores(q["question"], 1)[0][1]
    groups = {"answerable": [(top(q), q) for q in answerable], "unanswerable": [(top(q), q) for q in unanswerable]}

    print(f"best bge score per question (question embedding requests this run: {encode.calls})\n")
    for name, scored in groups.items():
        scores = np.array([s for s, _ in scored])
        p = np.percentile(scores, [0, 5, 10, 25, 50]).round(4)
        print(f"{name} ({len(scores)}): min {p[0]}  p5 {p[1]}  p10 {p[2]}  p25 {p[3]}  median {p[4]}  "
              f"max {scores.max():.4f}")
    edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0001]
    print("\nscore range        answerable  unanswerable")
    for low, high in zip(edges, edges[1:]):
        counts = [sum(low <= s < high for s, _ in scored) for scored in groups.values()]
        print(f"{low:<7} - {min(high, 1):<7}   {counts[0]:>10}  {counts[1]:>12}")
    print("\nlowest answerable:")
    for score, q in sorted(groups["answerable"], key=lambda pair: pair[0])[:6]:
        print(f"  {score:.4f}  [{q['type']}] {q['question']}")
    print("unanswerable, highest first:")
    for score, q in sorted(groups["unanswerable"], key=lambda pair: pair[0], reverse=True):
        print(f"  {score:.4f}  [{q['type']}] {q['question']}")
    t = args.threshold if args.threshold is not None else ABSTAIN_THRESHOLD
    lost = sum(s < t for s, _ in groups["answerable"])
    caught = sum(s < t for s, _ in groups["unanswerable"])
    print(f"\nthreshold {t}: wrongly abstains on {lost}/{len(answerable)} answerable, "
          f"abstains before the LLM on {caught}/{len(unanswerable)} unanswerable")


def print_answer(answer):
    print(f"\nQ: {answer.question}")
    if answer.abstained:
        print(f"A: {answer.text}\n   abstained: yes ({answer.reason})")
    else:
        for number, part in enumerate(answer.parts, start=1):
            print(f"A{number}: {part['text']}")
            for c in part["citations"]:
                print(f"     [{c['reference']}] \"{c['quote']}\"")
        print("   abstained: no")
    u = answer.usage
    print(f"   best passage score {answer.top_score:.4f} | LLM calls {u.calls} | tokens in {u.input_tokens} "
          f"(cached {u.cached_input_tokens}) out {u.output_tokens} (reasoning {u.reasoning_tokens}) | "
          f"${u.cost:.5f} | {answer.seconds:.1f}s")
    if answer.problems:
        print(f"   last citation problems: {answer.problems}")


def cmd_ask(args):
    from app import generation
    from app.techniques import best_azure_retriever

    if len(args.questions) > 1:
        e = generation.estimate(args.questions)
        print(f"batch: {e['questions']} questions -> {e['calls'][0]}-{e['calls'][1]} LLM calls (fewer if some abstain "
              f"before the LLM), estimated ${e['cost_usd'][0]:.4f}-${e['cost_usd'][1]:.4f}")
        if not args.yes:
            print("not run: add --yes to run it")
            return
    retriever = best_azure_retriever(DATA_DIR)
    total = generation.Usage()
    for question in args.questions:
        answer = generation.answer_question(question, retriever)
        generation.append_log(answer, args.log)
        print_answer(answer)
        total.calls += answer.usage.calls
        total.input_tokens += answer.usage.input_tokens
        total.cached_input_tokens += answer.usage.cached_input_tokens
        total.output_tokens += answer.usage.output_tokens
        total.reasoning_tokens += answer.usage.reasoning_tokens
    if len(args.questions) > 1:
        print(f"\ntotal: {total.calls} LLM calls, {total.input_tokens} tokens in, {total.output_tokens} out, "
              f"${total.cost:.5f}; logged to {args.log}")


def main():
    parser = argparse.ArgumentParser(prog="python -m app.main", description="KJV RAG pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="parse the PDF into verses, parents and children")
    ingest.add_argument("--pdf", default=f"{DATA_DIR}/KJVfile.pdf")
    ingest.set_defaults(run=cmd_ingest)

    embed = commands.add_parser("embed", help="embed the children and build the vector stores")
    embed.add_argument("--model", default="all", choices=["all", *embeddings.MODELS])
    embed.set_defaults(run=cmd_embed)

    search = commands.add_parser("search", help="find the passages that best answer a question")
    search.add_argument("question")
    search.add_argument("--technique", default="rerank", choices=["rerank", "hybrid", "dense"],
                        help="rerank (Hybrid + bge cross-encoder, the most accurate), hybrid (mpnet + BM25, "
                             "fastest good option) or dense only")
    search.add_argument("--model", default="mpnet", choices=list(embeddings.MODELS), help="dense only")
    search.add_argument("--store", default="faiss", choices=STORES, help="dense only")
    search.add_argument("-k", type=int, default=3, help="how many passages to show")
    search.set_defaults(run=cmd_search)

    evaluate = commands.add_parser("evaluate", help="score models and stores on the test questions")
    evaluate.add_argument("--model", default="all", choices=["all", *embeddings.MODELS])
    evaluate.add_argument("--store", default="all", choices=["all", *STORES])
    evaluate.set_defaults(run=cmd_evaluate)

    compare = commands.add_parser("compare", help="score the ten retrieval techniques on the test questions")
    compare.set_defaults(run=cmd_compare)

    rerank = commands.add_parser("rerank", help="score Hybrid with and without each reranker on the test questions")
    rerank.add_argument("--candidates", type=int, default=10, help="how many Hybrid results get re-scored")
    rerank.set_defaults(run=cmd_rerank)

    check = commands.add_parser("check", help="test the connection to every cloud service")
    check.set_defaults(run=cmd_check)

    azure_index = commands.add_parser("azure-index", help="embed the parents and upload them to Azure AI Search")
    azure_index.set_defaults(run=cmd_azure_index)

    azure_storage = commands.add_parser("azure-storage", help="show the index size against the free-tier limits")
    azure_storage.set_defaults(run=cmd_azure_storage)

    azure_compare = commands.add_parser("azure-compare", help="score AI Search hybrid, with and without bge, "
                                                              "beside the local Hybrid + bge")
    azure_compare.set_defaults(run=cmd_azure_compare)

    azure_depth = commands.add_parser("azure-depth", help="score AI Search hybrid + bge re-scoring different numbers "
                                                          "of candidates")
    azure_depth.add_argument("--depths", type=int, nargs="+", default=[10, 20, 30])
    azure_depth.set_defaults(run=cmd_azure_depth)

    abstain = commands.add_parser("abstain-scores", help="best bge score of answerable vs unanswerable questions")
    abstain.add_argument("--threshold", type=float, default=None, help="threshold to test (default: the current one)")
    abstain.set_defaults(run=cmd_abstain_scores)

    ask = commands.add_parser("ask", help="answer questions with cited verses (gpt-5.4-mini), or abstain")
    ask.add_argument("questions", nargs="+")
    ask.add_argument("--yes", action="store_true", help="run a batch of several questions after the estimate")
    ask.add_argument("--log", default=f"{DATA_DIR}/runs/answers.jsonl")
    ask.set_defaults(run=cmd_ask)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # KJV quotes use curly apostrophes
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
