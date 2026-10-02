# Teacher → Student Distillation

This repository implements **response distillation**:

1. Start with grounded examples: evidence + question.
2. A larger teacher LLM produces a grounded answer.
3. Store the teacher answer as the target.
4. Fine-tune the smaller student SLM using LoRA or QLoRA.
5. Use Jev/Ragas/another judge to evaluate whether the student remains faithful to evidence.

This works across model families because it distills **outputs**, not internal logits.

Example:

```powershell
python training/make_student_dataset.py

python training/distill_generate.py `
  --teacher Qwen/Qwen2.5-1.5B-Instruct

python training/train_lora.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --data data/distilled_sft.jsonl
```

For a truly large teacher such as 7B/14B/32B+, use a GPU machine/cloud endpoint.
A 4 GB laptop GPU is suitable for the small student experiments, not for training
a large teacher.

## Why response distillation instead of logits?

Logit distillation normally requires access to the teacher model's token
probabilities and compatible tokenization. Response distillation only needs
teacher-generated answers, so the teacher can be:

- a larger local Hugging Face model
- a hosted API model
- another instruction model from a different family
