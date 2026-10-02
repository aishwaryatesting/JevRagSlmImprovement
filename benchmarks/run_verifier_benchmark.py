import argparse
import csv
import time
from pathlib import Path

from app.verifier import verify
from app.datasets.contract_nli_loader import load_contract_nli
from app.datasets.fever_loader import load_fever

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]

def report(rows):
    total = len(rows)
    correct = sum(r["gold"] == r["pred"] for r in rows)
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}

    for row in rows:
        gold, pred = row["gold"], row["pred"]
        for label in LABELS:
            if pred == label and gold == label:
                stats[label]["tp"] += 1
            elif pred == label and gold != label:
                stats[label]["fp"] += 1
            elif pred != label and gold == label:
                stats[label]["fn"] += 1

    print(f"Examples: {total}")
    print(f"Accuracy: {correct / max(total, 1):.4f}")

    f1s = []
    for label, s in stats.items():
        precision = s["tp"] / max(1, s["tp"] + s["fp"])
        recall = s["tp"] / max(1, s["tp"] + s["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        print(
            f"{label:13s} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )

    print(f"Macro-F1: {sum(f1s)/len(f1s):.4f}")
    avg_latency = sum(r["latency_ms"] for r in rows) / max(1, total)
    print(f"Average latency: {avg_latency:.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["contract_nli", "fever"], default="contract_nli")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/results.csv")
    args = parser.parse_args()

    examples = (
        load_contract_nli(limit=args.limit)
        if args.dataset == "contract_nli"
        else load_fever(limit=args.limit)
    )

    if not examples:
        raise RuntimeError(
            f"No usable examples loaded from {args.dataset}. "
            "For the first real benchmark, use ContractNLI."
        )

    rows = []
    for idx, ex in enumerate(examples, start=1):
        start = time.perf_counter()
        result = verify(ex.claim, ex.evidence)
        latency_ms = (time.perf_counter() - start) * 1000

        rows.append({
            "id": idx,
            "dataset": ex.source,
            "gold": ex.label,
            "pred": result.verdict,
            "confidence": result.confidence,
            "verifier": result.verifier,
            "latency_ms": round(latency_ms, 3),
            "claim": ex.claim,
            "evidence": ex.evidence,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report(rows)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
