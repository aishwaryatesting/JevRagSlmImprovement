"""
LoRA fine-tuning for the student SLM.

Default student is intentionally small so the code is more likely to run on
consumer hardware. For the real benchmark, swap in the student you want.
"""

import argparse
import json
from pathlib import Path

def format_example(row):
    return (
        f"Instruction: {row['instruction']}\\n"
        f"Evidence: {row['context']}\\n"
        f"Question: {row['question']}\\n"
        f"Answer: {row['answer']}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/distilled_sft.jsonl")
    parser.add_argument("--output", default="artifacts/student-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    dataset = Dataset.from_list([{"text": format_example(r)} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")

if __name__ == "__main__":
    main()
