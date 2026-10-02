import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def build_text(row):
    instruction = row.get(
        "instruction",
        "Given the contract text and claim, classify the claim."
    )
    return f"""Instruction:
{instruction}

Contract:
{row["context"]}

Claim:
{row["question"]}

Answer:
{row["answer"]}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/enriched/teacher_validated.jsonl")
    parser.add_argument("--output", default="artifacts/student-qlora-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = [
        json.loads(x)
        for x in Path(args.data).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if not rows:
        raise RuntimeError(f"No training rows found in {args.data}")

    ds = Dataset.from_list([{"text": build_text(r)} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if use_cuda:
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=qconfig,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        compute_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=args.output,
        dataset_text_field="text",
        max_length=args.max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",

        # Critical fix:
        # BF16 does not use FP16 GradScaler, so BF16 gradients are not
        # passed into scaler.unscale_().
        bf16=use_bf16,
        fp16=(use_cuda and not use_bf16),

        gradient_checkpointing=use_cuda,
        optim="adamw_torch",
        dataloader_pin_memory=use_cuda,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"Training rows: {len(ds)}")
    print(f"Base model: {args.model}")
    print(f"CUDA available: {use_cuda}")
    print(f"BF16 supported: {use_bf16}")
    print(f"Training dtype: {compute_dtype}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {args.output}")

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("")
    print("QLoRA training completed.")
    print(f"Adapter saved to: {args.output}")


if __name__ == "__main__":
    main()
