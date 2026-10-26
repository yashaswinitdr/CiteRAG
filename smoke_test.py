"""
Quick end-to-end check on the two tiny sample documents in samples/.
It verifies answers, answer style, and source attribution.

    python ingest.py --docs samples --db chroma_samples
    python smoke_test.py
"""

import config
from rag_chain import RagPipeline

VACATION = "vacation_policy.txt"
FAQ = "product_faq.txt"

YES_NO_STARTERS = ("is ", "are ", "can ", "does ", "do ", "will ")

# (question, check on the lowercased answer, expected source file or None)
CASES = [
    (
        "How many vacation days do full-time employees get per year, and how many can they carry over?",
        lambda a: "18" in a and "5" in a,
        VACATION,
    ),
    (
        "How much notice do I need for a vacation request longer than 3 consecutive days?",
        lambda a: "10 business days" in a,
        VACATION,
    ),
    (
        "A part-time employee works 20 hours per week. How many vacation days do they accrue per year?",
        lambda a: "9" in a,
        VACATION,
    ),
    (
        "What approval is needed for vacation in the last two weeks of December?",
        lambda a: "manager" in a and "30 days" in a,
        VACATION,
    ),
    (
        "How long does the TidyBot 3000 run on a full charge, and how long does it take to recharge?",
        lambda a: "90 minutes" in a and "3 hours" in a,
        FAQ,
    ),
    (
        "What is the warranty period for the TidyBot 3000?",
        lambda a: "2-year" in a or "2 year" in a or "two-year" in a,
        FAQ,
    ),
    (
        "Can the TidyBot 3000 clean stairs?",
        lambda a: a.lstrip().startswith("no"),
        FAQ,
    ),
    (
        "Is the TidyBot 3000 safe to use around pets?",
        lambda a: a.lstrip().startswith("yes"),
        FAQ,
    ),
    (
        "What is the capital of France?",
        lambda a: "paris" not in a and "do not contain" in a,
        None,
    ),
]


def main():
    pipeline = RagPipeline(db_dir=config.BASE_DIR / "chroma_samples", k=3)
    passed = 0

    for number, (question, check, expected_source) in enumerate(CASES, start=1):
        result = pipeline.answer(question)
        answer = result["answer"].lower()
        sources = [s.replace("\\", "/").split("/")[-1] for s in result["sources"]]

        answer_ok = check(answer)
        is_yes_no = question.lower().startswith(YES_NO_STARTERS)
        style_ok = is_yes_no or not answer.lstrip().startswith(("yes", "no"))
        if expected_source is None:
            source_ok = sources == []
        else:
            source_ok = expected_source in sources
        ok = answer_ok and style_ok and source_ok
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {number}. {question}")
        print(f"       answer : {result['answer']}")
        print(f"       sources: {sources}")
        if not answer_ok:
            print("       -> answer check failed")
        if not style_ok:
            print("       -> a non yes/no question was answered with Yes/No")
        if not source_ok:
            print(f"       -> expected source {expected_source}")
        print()

    print(f"{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
