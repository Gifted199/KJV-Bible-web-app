"""Command line: python -m app.main <command>

    ingest                       PDF -> data/verses.jsonl, parents.jsonl, children.jsonl
    embed    [--model all]       embed the children, build the FAISS index and Chroma collection
    search   "question"          search with Hybrid (the best technique) and print the best passages
    evaluate [--model all]       score each model and store on data/test_questions.json
    compare                      score the ten retrieval techniques on data/test_questions.json
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
STORES = ["faiss", "chroma"]


def chroma_client():
    import chromadb

    return chromadb.PersistentClient(path=f"{DATA_DIR}/chroma")


def load_chunks():
    children = chunking.read_jsonl(f"{DATA_DIR}/children.jsonl")
    parents = chunking.read_jsonl(f"{DATA_DIR}/parents.jsonl")
    return children, {p["id"]: p for p in parents}


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
    if args.technique == "hybrid":
        from app.techniques import best_retriever

        found = best_retriever(DATA_DIR).search(args.question, args.k)
    else:
        children, parents_by_id = load_chunks()
        retriever = load_retriever(args.store, args.model, children)
        found = parents_for(retriever.search(args.question), parents_by_id, k=args.k)
    for rank, parent in enumerate(found, start=1):
        print(f"\n#{rank} {parent['text']}")


def cmd_evaluate(args):
    children, parents_by_id = load_chunks()
    questions = evaluation.load_test_questions(f"{DATA_DIR}/test_questions.json")
    print(f"{len(questions)} test questions")
    for store in chosen(args.store, STORES):
        for name in chosen(args.model, list(embeddings.MODELS)):
            retriever = load_retriever(store, name, children)
            scores, _ = evaluation.evaluate(lambda q: parents_for(retriever.search(q), parents_by_id), questions)
            print(f"{store:7} {name:7} " + "  ".join(f"{k}: {v}" for k, v in scores.items()))


def compare_techniques(retrievers, questions):
    """Score every technique; returns (overall table, MRR-by-question-type table), best first."""
    import pandas as pd

    rows, by_type = [], {}
    for name, retriever in retrievers.items():
        scores, details = evaluation.evaluate(lambda q: retriever.search(q, 10), questions)
        rows.append({"technique": name, **scores})
        by_type[name] = {
            t: evaluation.score([d for d in details if d["type"] == t])["MRR"]
            for t in dict.fromkeys(d["type"] for d in details)
        }
    overall = pd.DataFrame(rows).sort_values(["MRR", "correct in top 5"], ascending=False).reset_index(drop=True)
    overall.index += 1
    return overall, pd.DataFrame(by_type).T.loc[overall["technique"]]


def cmd_compare(args):
    import pandas as pd

    from app.techniques import build_techniques

    questions = evaluation.load_test_questions(f"{DATA_DIR}/test_questions.json")
    overall, by_type = compare_techniques(build_techniques(DATA_DIR), questions)
    pd.set_option("display.width", 200)
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
    search.add_argument("--technique", default="hybrid", choices=["hybrid", "dense"],
                        help="hybrid (the best in `compare`: mpnet + BM25) or dense only")
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

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
