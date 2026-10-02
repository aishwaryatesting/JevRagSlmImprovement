import argparse
import json
import random
from pathlib import Path

def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/enriched/teacher_enriched.jsonl")
    parser.add_argument("--hard", default="data/hard_examples/hard_examples.jsonl")
    parser.add_argument("--output", default="data/processed/mixed_training.jsonl")
    parser.add_argument("--hard-ratio", type=float, default=0.20)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    clean = read_jsonl(args.clean)
    hard = read_jsonl(args.hard)

    if not clean:
        raise RuntimeError("Clean dataset is empty.")
    if not hard:
        raise RuntimeError("Hard-example dataset is empty.")

    hard_n = max(1, int(args.size * args.hard_ratio))
    clean_n = max(1, args.size - hard_n)

    mixed = (
        random.choices(clean, k=clean_n)
        + random.choices(hard, k=hard_n)
    )
    random.shuffle(mixed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mixed)} mixed examples to {out}")

if __name__ == "__main__":
    main()
