import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
policies = json.loads((ROOT / "data" / "policies.json").read_text(encoding="utf-8"))

samples = [
    {
        "instruction": "Answer using only the supplied evidence.",
        "context": policies[0]["text"],
        "question": "Can I return opened headphones after opening them?",
        "answer": "Yes, but only if the headphones are defective."
    },
    {
        "instruction": "Answer using only the supplied evidence.",
        "context": policies[1]["text"],
        "question": "How long do I have to request a refund?",
        "answer": "Refund requests are accepted within 30 days of purchase."
    },
    {
        "instruction": "Answer using only the supplied evidence.",
        "context": policies[2]["text"],
        "question": "What is the standard warranty period?",
        "answer": "The standard warranty period is one year from the date of purchase."
    },
    {
        "instruction": "Answer using only the supplied evidence.",
        "context": policies[3]["text"],
        "question": "Is international shipping free?",
        "answer": "The provided evidence does not say that international shipping is free."
    }
]

out = ROOT / "data" / "student_sft.jsonl"
with out.open("w", encoding="utf-8") as f:
    for row in samples:
        f.write(json.dumps(row) + "\\n")

print(f"Wrote {len(samples)} examples to {out}")
