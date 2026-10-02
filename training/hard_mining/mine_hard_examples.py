import argparse
import csv
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results.csv")
    parser.add_argument("--output", default="data/hard_examples/hard_examples.jsonl")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    args = parser.parse_args()

    hard = []

    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold = row.get("gold")
            pred = row.get("pred")
            confidence = float(row.get("confidence", "0") or 0)

            if pred != gold or confidence < args.confidence_threshold:
                hard.append({
                    "instruction": (
                        "Given the evidence and claim, return a verdict, "
                        "a short rationale, and a grounded response."
                    ),
                    "context": row.get("evidence", ""),
                    "question": row.get("claim", ""),
                    "label": gold,
                    "previous_prediction": pred,
                    "previous_confidence": confidence,
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in hard:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(hard)} hard examples to {out}")

if __name__ == "__main__":
    main()
