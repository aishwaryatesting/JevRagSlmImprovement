import csv
from pathlib import Path
from app.verifier import verify

DATA = Path(__file__).parent / "data" / "ground_truth.csv"

labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
total = correct = 0
by_label = {x: {"tp": 0, "fp": 0, "fn": 0} for x in labels}

with DATA.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        pred = verify(row["claim"], row["evidence"]).verdict
        gold = row["label"]
        total += 1
        correct += int(pred == gold)

        for label in labels:
            if pred == label and gold == label:
                by_label[label]["tp"] += 1
            elif pred == label and gold != label:
                by_label[label]["fp"] += 1
            elif pred != label and gold == label:
                by_label[label]["fn"] += 1

print(f"Accuracy: {correct}/{total} = {correct / max(total,1):.3f}")
for label, m in by_label.items():
    p = m["tp"] / max(1, m["tp"] + m["fp"])
    r = m["tp"] / max(1, m["tp"] + m["fn"])
    f1 = 2*p*r / max(1e-12, p+r)
    print(f"{label:13s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
