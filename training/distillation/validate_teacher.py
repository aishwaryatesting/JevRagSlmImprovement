"""
Validate teacher outputs before distillation.

Validation:
1. Parse the teacher verdict.
2. Teacher verdict must match the authoritative human label.
3. Optional Jev check:
   Jev independently classifies the ORIGINAL claim against the ORIGINAL context.
   Jev verdict must match the same human label.
4. Jev confidence is recorded. A confidence threshold is optional; it is not
   used unless --jev-threshold is explicitly set above 0.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.verifier import verify


VERDICT_PATTERNS = [
    re.compile(
        r"\bVerdict\s*:\s*[\"']?(SUPPORTED|CONTRADICTED|INSUFFICIENT)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\"']verdict[\"']\s*:\s*[\"']?(SUPPORTED|CONTRADICTED|INSUFFICIENT)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(SUPPORTED|CONTRADICTED|INSUFFICIENT)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def extract_verdict(text: str):
    text = text or ""
    for pattern in VERDICT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/enriched/teacher_enriched.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/enriched/teacher_validated.jsonl",
    )
    parser.add_argument(
        "--rejects",
        default="data/enriched/teacher_rejected.jsonl",
    )
    parser.add_argument(
        "--require-jev",
        action="store_true",
        help=(
            "Ask Jev to independently classify the original claim against "
            "the original context."
        ),
    )
    parser.add_argument(
        "--jev-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional minimum Jev confidence. Default 0.0 means do not reject "
            "based on confidence; record confidence for later analysis."
        ),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    accepted = []
    rejected = []
    reason_counts = Counter()

    for row in rows:
        human_label = row["label"].upper()
        teacher_answer = row.get("answer", "")
        teacher_verdict = extract_verdict(teacher_answer)

        reasons = []

        if teacher_verdict is None:
            reasons.append("teacher verdict could not be parsed")
        elif teacher_verdict != human_label:
            reasons.append(
                f"teacher={teacher_verdict} != human={human_label}"
            )

        jev_result = None

        if args.require_jev:
            # Jev verifies the ORIGINAL claim, not the teacher's whole response.
            jev_result = verify(
                row["question"],
                row["context"],
            )

            if jev_result.verdict != human_label:
                reasons.append(
                    f"jev={jev_result.verdict} != human={human_label}"
                )

            if (
                args.jev_threshold > 0
                and jev_result.confidence < args.jev_threshold
            ):
                reasons.append(
                    f"jev confidence={jev_result.confidence:.3f} "
                    f"< threshold={args.jev_threshold:.3f}"
                )

        out_row = dict(row)
        out_row["teacher_verdict"] = teacher_verdict

        if jev_result:
            out_row["teacher_validation_jev"] = {
                "verdict": jev_result.verdict,
                "confidence": jev_result.confidence,
                "verifier": jev_result.verifier,
            }

        if reasons:
            out_row["validation_status"] = "REJECTED"
            out_row["validation_reasons"] = reasons
            rejected.append(out_row)
            reason_counts.update(reasons)
        else:
            out_row["validation_status"] = "ACCEPTED"
            accepted.append(out_row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reject_path = Path(args.rejects)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    with reject_path.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Validated dataset: {out_path}")
    print(f"Rejected dataset: {reject_path}")

    if reason_counts:
        print("\nRejection summary:")
        for reason, count in reason_counts.most_common():
            print(f"  {count:>3}  {reason}")


if __name__ == "__main__":
    main()
