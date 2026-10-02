import argparse
import csv
import time
from pathlib import Path

from app.factory import build_generator
from app.verifier import verify
from app.datasets.contract_nli_loader import load_contract_nli

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="benchmarks/generation_results.csv")
    args = parser.parse_args()

    examples = load_contract_nli(limit=args.limit)
    generator = build_generator()
    rows = []

    for idx, ex in enumerate(examples, start=1):
        question = (
            "Answer using only the supplied contract evidence. "
            f"Claim to assess: {ex.claim}"
        )

        start = time.perf_counter()
        answer = generator.generate(question, [ex.evidence])
        generation_ms = (time.perf_counter() - start) * 1000

        check = verify(answer, ex.evidence)

        rows.append({
            "id": idx,
            "gold_claim_label": ex.label,
            "claim": ex.claim,
            "generated_answer": answer,
            "answer_verdict": check.verdict,
            "verifier": check.verifier,
            "confidence": check.confidence,
            "generation_latency_ms": round(generation_ms, 3),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
