"""
Teacher enrichment for verifier-guided distillation.

Important:
- The teacher NEVER sees the human ground-truth label.
- The human label remains hidden and is used only later for validation.
- The teacher must independently classify context + claim.
"""

import argparse
import json
import re
from pathlib import Path


def clean_generated_answer(text: str) -> str:
    """Trim common Qwen chatter and keep the generated answer compact."""
    text = (text or "").strip()

    # If the model produced the requested structured block somewhere later,
    # start from the first Verdict occurrence.
    m = re.search(r"\bVerdict\s*:", text, flags=re.IGNORECASE)
    if m:
        text = text[m.start():]

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input", default="data/processed/contract_nli_train.jsonl")
    parser.add_argument("--output", default="data/enriched/teacher_enriched.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out_rows = []

    system_message = (
        "You are a contract claim verifier. "
        "Use only the supplied contract text. "
        "Do not mention human labels, datasets, training data, or evaluation. "
        "Do not invent facts."
    )

    for i, row in enumerate(rows, start=1):
        context = row["context"]
        claim = row["question"]

        # CRITICAL: row['label'] is deliberately NOT included in the prompt.
        user_message = f"""Classify the claim using exactly one of:
SUPPORTED
CONTRADICTED
INSUFFICIENT

Definitions:
SUPPORTED: the contract text establishes the claim.
CONTRADICTED: the contract text conflicts with the claim.
INSUFFICIENT: the contract text does not establish or contradict the claim.

Return exactly three lines:
Verdict: <SUPPORTED|CONTRADICTED|INSUFFICIENT>
Rationale: <one concise sentence using only the contract text>
Grounded response: <one concise sentence grounded only in the contract text>

Contract text:
{context}

Claim:
{claim}
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = system_message + "\n\n" + user_message

        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        answer = clean_generated_answer(answer)

        out_rows.append(
            {
                "instruction": row.get(
                    "instruction",
                    "Given the contract text and claim, classify the claim.",
                ),
                "context": context,
                "question": claim,
                # Ground truth is stored for later validation, but was NOT shown
                # to the teacher.
                "label": row["label"],
                "answer": answer,
                "teacher_model": args.teacher,
                "teacher_saw_ground_truth": False,
            }
        )

        print(f"Enriched {i}/{len(rows)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} enriched rows to {out}")


if __name__ == "__main__":
    main()
