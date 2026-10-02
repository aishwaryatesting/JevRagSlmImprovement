# Hugging Face benchmark layer

Install:

```powershell
pip install -r requirements-benchmark.txt
```

## 1. ContractNLI verifier benchmark

```powershell
$env:VERIFIER_MODE="offline"

python benchmarks/run_verifier_benchmark.py `
  --dataset contract_nli `
  --limit 100
```

To benchmark real Jev on the same dataset:

```powershell
$env:TYPESAFE_API_KEY="YOUR_KEY"
$env:VERIFIER_MODE="jev"

python benchmarks/run_verifier_benchmark.py `
  --dataset contract_nli `
  --limit 100 `
  --output benchmarks/jev_contract_nli.csv
```

## 2. Base SLM vs distilled SLM

Base student:

```powershell
$env:GENERATOR_MODE="hf"
$env:HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
Remove-Item Env:HF_ADAPTER -ErrorAction SilentlyContinue

python benchmarks/run_generation_benchmark.py `
  --limit 50 `
  --output benchmarks/base_slm.csv
```

Distilled student:

```powershell
$env:HF_ADAPTER="artifacts/student-lora"

python benchmarks/run_generation_benchmark.py `
  --limit 50 `
  --output benchmarks/distilled_slm.csv
```

## Metrics

Verifier benchmark:
- accuracy
- per-class precision
- per-class recall
- per-class F1
- macro-F1
- average latency

The CSV files retain the full claim, evidence, gold label, prediction and confidence
for error analysis.

## FEVER

A basic FEVER loader is included, but FEVER often stores evidence as Wikipedia
sentence references rather than already-resolved text. ContractNLI is therefore
the recommended first benchmark.
