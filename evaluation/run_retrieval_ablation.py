"""
Retrieval ablation: which embedding model, chunk size and retrieval mode finds the
right passage most often? No LLM is involved, so this runs quickly.

    python -m evaluation.run_retrieval_ablation
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ingest import ingest
from rag_chain import RagPipeline
from retriever import MODES
from evaluation.metrics import first_hit_rank, load_dataset, retrieval_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WORK_DIR = config.BASE_DIR / "chroma_eval"
MAX_K = 5


def evaluate_mode(db_dir, embedding_model, mode, dataset):
    pipeline = RagPipeline(
        db_dir=db_dir, k=MAX_K, embedding_model=embedding_model, retrieval_mode=mode
    )
    ranks, top1_source = [], []
    for item in dataset:
        if item["type"] != "answerable":
            continue
        docs = pipeline.retrieve(item["question"])
        ranks.append(first_hit_rank([d.page_content for d in docs], item["evidence"]))
        top1_source.append(Path(docs[0].metadata["source"]).name == item["source"])
    metrics = retrieval_metrics(ranks)
    metrics["top1_source_accuracy"] = sum(top1_source) / len(top1_source)
    return metrics


def to_markdown(rows):
    lines = [
        "| Embedding model | Chunk size | Mode | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | Top-1 source acc. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['embedding_model'].split('/')[-1]} | {r['chunk_size']} | {r['mode']} | "
            f"{r['n_chunks']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | "
            f"{r['mrr']:.2f} | {r['top1_source_accuracy']:.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="path to a dataset json (default: dev set)")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 800, 1200])
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--output", default="retrieval_ablation")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset) if args.dataset else load_dataset()
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    rows = []
    for model in args.embedding_models:
        for chunk_size in args.chunk_sizes:
            overlap = int(chunk_size * args.overlap_ratio)
            db_dir = WORK_DIR / f"{model.split('/')[-1]}_{chunk_size}"
            print(f"Building {model.split('/')[-1]} chunk_size={chunk_size} ...", flush=True)
            _, n_chunks = ingest(config.DOCUMENTS_DIR, db_dir, chunk_size, overlap, model)
            for mode in args.modes:
                metrics = evaluate_mode(db_dir, model, mode, dataset)
                row = {
                    "embedding_model": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "mode": mode,
                    "n_chunks": n_chunks,
                    **{key: round(value, 3) for key, value in metrics.items()},
                }
                rows.append(row)
                print(f"  {mode:14s} hit@1={row['hit@1']:.2f} hit@3={row['hit@3']:.2f} "
                      f"hit@5={row['hit@5']:.2f} mrr={row['mrr']:.2f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.output}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = to_markdown(rows)
    (RESULTS_DIR / f"{args.output}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
