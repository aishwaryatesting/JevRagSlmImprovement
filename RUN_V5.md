# V5 End-to-End Runbook

## 0. Setup

```powershell
cd D:\jev_rag_verifier_v5

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-training.txt
pip install -r requirements-benchmark.txt
pip install -r requirements-jev.txt
pip install -r requirements-qlora.txt
```

## 1. Build ContractNLI dataset

```powershell
python training/datasets/build_contractnli_training.py `
  --limit 2000
```

Creates:

```text
data/processed/contract_nli_train.jsonl
```

## 2. Teacher enrichment

```powershell
python training/distillation/teacher_enrich.py `
  --teacher Qwen/Qwen2.5-7B-Instruct `
  --input data/processed/contract_nli_train.jsonl `
  --output data/enriched/teacher_enriched.jsonl
```

The human label remains the ground truth.
The teacher adds rationale + grounded response.

If 7B is too large locally, run this step on a stronger GPU or choose a smaller teacher.



## 2B. Validate teacher outputs before QLoRA

First ensure the teacher's verdict matches the authoritative human label:

```powershell
python training/distillation/validate_teacher.py `
  --input data/enriched/teacher_enriched.jsonl `
  --output data/enriched/teacher_validated.jsonl
```

To also require Jev support:

```powershell
$env:VERIFIER_MODE="jev"

python training/distillation/validate_teacher.py `
  --input data/enriched/teacher_enriched.jsonl `
  --output data/enriched/teacher_validated.jsonl `
  --require-jev `
  --jev-threshold 0.80
```

Only accepted teacher examples should be used for QLoRA.

## 3. QLoRA train SLM v1

```powershell
python training/finetuning/train_qlora.py `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --data data/enriched/teacher_validated.jsonl `
  --output artifacts/student-qlora-v1
```

## 4. Configure Jev safely

Do NOT paste the key into source code or chat.

PowerShell:

```powershell
$env:TYPESAFE_API_KEY="YOUR_REAL_KEY"
$env:VERIFIER_MODE="jev"
```

## 5. Run the application

```powershell
$env:GENERATOR_MODE="hf"
$env:HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
$env:HF_ADAPTER="artifacts/student-qlora-v1"

python run_cli.py "Can I return opened headphones?"
```

## 6. Benchmark

```powershell
python benchmarks/run_verifier_benchmark.py `
  --dataset contract_nli `
  --limit 100 `
  --output benchmarks/results.csv
```

## 7. Mine hard failures

```powershell
python training/hard_mining/mine_hard_examples.py `
  --input benchmarks/results.csv `
  --output data/hard_examples/hard_examples.jsonl
```

## 8. Teacher-enrich the hard examples too

```powershell
python training/distillation/teacher_enrich.py `
  --teacher Qwen/Qwen2.5-7B-Instruct `
  --input data/hard_examples/hard_examples.jsonl `
  --output data/enriched/hard_teacher_enriched.jsonl
```

## 9. Build mixed clean + hard dataset

```powershell
python training/hard_mining/build_mixed_dataset.py `
  --clean data/enriched/teacher_enriched.jsonl `
  --hard data/enriched/hard_teacher_enriched.jsonl `
  --output data/processed/mixed_training.jsonl `
  --hard-ratio 0.20
```

## 10. QLoRA train SLM v2

```powershell
python training/finetuning/train_qlora.py `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --data data/processed/mixed_training.jsonl `
  --output artifacts/student-qlora-v2
```

## 11. Compare v1 vs v2

Run the same held-out benchmark for both adapters.

Only promote v2 if it improves the metrics you care about, especially:
- macro-F1
- accuracy
- false acceptance rate
- faithfulness
- latency

## 12. Promote v2

```powershell
python scripts/promote_model.py `
  --version v2 `
  --path artifacts/student-qlora-v2 `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --macro-f1 0.92
```

## 13. Check current production model

```powershell
python scripts/show_current_model.py
```

Current version is stored in:

```text
artifacts/current_model.json
```

History is stored in:

```text
artifacts/model_history.json
```
