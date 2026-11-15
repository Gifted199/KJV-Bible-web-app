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
"""

import argparse
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

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
