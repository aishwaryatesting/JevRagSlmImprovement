import argparse
import json
from pathlib import Path
from datasets import load_dataset

TEXT_MAP = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT",
    "supported": "SUPPORTED",
    "contradicted": "CONTRADICTED",
    "insufficient": "INSUFFICIENT",
}

def resolve_label(ds, raw):
    try:
        feature = ds.features["label"]
        if hasattr(feature, "int2str") and isinstance(raw, int):
            return TEXT_MAP.get(feature.int2str(raw).strip().lower())
    except Exception:
        pass
    return TEXT_MAP.get(str(raw).strip().lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", default="data/processed/contract_nli_train.jsonl")
    args = parser.parse_args()

    ds = load_dataset("tasksource/contract-nli", "contractnli_a", split=args.split)

    out_rows = []
    for row in ds:
        evidence = (
            row.get("premise")
            or row.get("contract")
            or row.get("context")
            or row.get("document")
        )
        claim = row.get("hypothesis") or row.get("claim")
        label = resolve_label(ds, row.get("label"))

        if not evidence or not claim or not label:
            continue

        out_rows.append({
            "instruction": (
                "Given the evidence and claim, return a verdict, "
                "a short rationale, and a grounded response."
            ),
            "context": str(evidence),
            "question": str(claim),
            "label": label,
        })

        if len(out_rows) >= args.limit:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {out}")

if __name__ == "__main__":
    main()
