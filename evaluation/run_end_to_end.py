"""
End-to-end evaluation: question -> retrieval -> LLM answer, scored automatically.

    python -m evaluation.run_end_to_end --dataset evaluation/dataset.json --output dev
    python -m evaluation.run_end_to_end --dataset evaluation/dataset_heldout.json --output heldout

Per answerable question it checks:
  * the answer contains the key facts (see metrics.answer_is_correct)
  * the cited sources include the correct paper
and per unanswerable question it checks that the system refuses and cites nothing.
Failures are classified as retrieval misses (the evidence never reached the LLM)
or generation errors (the evidence was retrieved but the answer is still wrong).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from rag_chain import RagPipeline, is_refusal
from evaluation.metrics import answer_is_correct, contains_evidence, load_dataset

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", default=str(config.CHROMA_DIR))
    parser.add_argument("--llm", default=config.OLLAMA_MODEL)
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE)
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME)
    parser.add_argument("--baseline", action="store_true", help="also run the LLM with no retrieval")
    parser.add_argument("--output", default="end_to_end")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset) if args.dataset else load_dataset()
    pipeline = RagPipeline(
        db_dir=args.db,
        k=args.k,
        embedding_model=args.embedding_model,
        llm_model=args.llm,
        retrieval_mode=args.mode,
    )

    records = []
    for number, item in enumerate(dataset, start=1):
        started = time.time()
        docs = pipeline.retrieve(item["question"])
        result = pipeline.answer(item["question"])
        answer, sources = result["answer"], [Path(s).name for s in result["sources"]]

        record = {"id": item["id"], "type": item["type"], "question": item["question"],
                  "answer": answer, "sources": sources}

        if item["type"] == "answerable":
            retrieved = any(contains_evidence(d.page_content, item["evidence"]) for d in docs)
            correct = answer_is_correct(answer, item) and not is_refusal(answer)
            source_ok = item["source"] in sources
            record.update(retrieval_hit=retrieved, answer_correct=correct,
                          source_correct=source_ok, passed=correct and source_ok)
            if not record["passed"]:
                record["failure"] = "retrieval_miss" if not retrieved else "generation_error"
        else:
            refused = is_refusal(answer)
            record.update(refused=refused, passed=refused and sources == [])
            if not record["passed"]:
                record["failure"] = "should_have_refused"

        if args.baseline and item["type"] == "answerable":
            record["baseline_answer"] = pipeline.answer_without_retrieval(item["question"])
            record["baseline_correct"] = (
                answer_is_correct(record["baseline_answer"], item)
                and not is_refusal(record["baseline_answer"])
            )

        records.append(record)
        print(f"[{'PASS' if record['passed'] else 'FAIL'}] {number:02d} {item['id']:10s} "
              f"({time.time() - started:.0f}s) {item['question']}", flush=True)
        if not record["passed"]:
            print(f"        -> {record.get('failure')}: {' '.join(answer.split())[:160]}", flush=True)

    answerable = [r for r in records if r["type"] == "answerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    summary = {
        "llm": args.llm, "retrieval_mode": args.mode, "k": args.k,
        "embedding_model": args.embedding_model, "db": args.db,
        "answerable_passed": sum(r["passed"] for r in answerable),
        "answerable_total": len(answerable),
        "unanswerable_passed": sum(r["passed"] for r in unanswerable),
        "unanswerable_total": len(unanswerable),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in answerable) / len(answerable), 3),
        "failures_retrieval_miss": sum(r.get("failure") == "retrieval_miss" for r in records),
        "failures_generation_error": sum(r.get("failure") == "generation_error" for r in records),
    }
    if args.baseline:
        summary["baseline_correct"] = sum(r["baseline_correct"] for r in answerable)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.output}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8"
    )
    total_pass = summary["answerable_passed"] + summary["unanswerable_passed"]
    total = summary["answerable_total"] + summary["unanswerable_total"]
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nOVERALL: {total_pass}/{total} passed")


if __name__ == "__main__":
    main()
