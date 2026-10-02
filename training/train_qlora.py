"""
Optional QLoRA training.

Requires bitsandbytes and a compatible CUDA setup. On some Windows setups,
plain LoRA is easier. Use this only when your environment supports it.
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
    parser.add_argument("--output", default="artifacts/student-qlora")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    dataset = Dataset.from_list([{"text": format_example(r)} for r in rows])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args_train = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=args_train,
        processing_class=tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
