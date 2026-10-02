"""
Teacher-response distillation.

This script uses a larger local Hugging Face model as the teacher to create
high-quality grounded answers. You can later train the smaller student on
those answers.

For a 4 GB GPU, choose a modest teacher or run on CPU. Large teachers may
require a cloud GPU.
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--input", default="data/student_sft.jsonl")
    parser.add_argument("--output", default="data/distilled_sft.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    out_rows = []

    for row in rows:
        prompt = (
            "You are the teacher model. Produce a concise answer using only the evidence. "
            "Preserve conditions and exceptions. If unsupported, say the evidence is insufficient.\\n\\n"
            f"Evidence: {row['context']}\\n"
            f"Question: {row['question']}\\n"
            "Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        answer_tokens = output[0][inputs["input_ids"].shape[1]:]
        teacher_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        out_rows.append({
            "instruction": row["instruction"],
            "context": row["context"],
            "question": row["question"],
            "answer": teacher_answer,
            "teacher_model": args.teacher,
        })

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\\n")

    print(f"Wrote {len(out_rows)} distilled examples to {out_path}")

if __name__ == "__main__":
    main()
