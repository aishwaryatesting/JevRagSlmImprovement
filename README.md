# SLM + Jev RAG Verifier

A small, runnable project for testing the architecture:

**RAG retrieval → small-model answer → claim splitting → Jev verification → accept/remove/escalate**

The project runs immediately in **offline mode**, so you do not need any API key for the demo.

When you have a TypeSafe API key, switch to **live Jev mode**.

## 1. Put it on D:\

Extract this folder as:

```powershell
D:\jev_rag_verifier
```

Then:

```powershell
cd D:\jev_rag_verifier
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run the CLI

```powershell
python run_cli.py "Can I return opened headphones?"
```

## 3. Run the tests

```powershell
pytest -q
```

## 4. Run the evaluation benchmark

```powershell
python evaluate.py
```

This compares verifier predictions against human-labelled ground truth.

## 5. Run the FastAPI service

```powershell
uvicorn app.main:app --reload --port 8000
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Can I return opened headphones?"}'
```

## 6. Run the Streamlit UI

```powershell
streamlit run app/ui.py
```

If direct execution reports a package import problem, use:

```powershell
python -m streamlit run app/ui.py
```

## 7. Enable live Jev

Install the SDK:

```powershell
pip install -r requirements-jev.txt
```

Then in PowerShell:

```powershell
$env:TYPESAFE_API_KEY="YOUR_KEY"
$env:VERIFIER_MODE="jev"
python run_cli.py "Can I return opened headphones?"
```

The implementation uses the TypeSafe SDK pattern:

```python
from typesafe_sdk import Choice, TypeSafeClient
```

and sends both:

- the **claim**
- the **retrieved evidence**

inside Jev's state.

Jev then makes a `Choice` among:

- `SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT`

## Project structure

```text
jev_rag_verifier/
├── app/
│   ├── generator.py      # deterministic SLM stand-in
│   ├── retriever.py      # tiny lexical RAG retriever
│   ├── verifier.py       # offline + live Jev adapter
│   ├── pipeline.py       # orchestration
│   ├── main.py           # FastAPI
│   └── ui.py             # Streamlit
├── data/
│   ├── policies.json
│   └── ground_truth.csv
├── tests/
│   └── test_pipeline.py
├── evaluate.py
├── run_cli.py
├── requirements.txt
├── requirements-jev.txt
└── .env.example
```

## Where the real SLM goes

`app/generator.py` intentionally uses a deterministic local function so the repository runs without another model.

Replace `draft_answer()` with:

- Phi
- Qwen
- Gemma
- Ollama
- vLLM
- Hugging Face Transformers
- an OpenAI-compatible endpoint

without changing the verifier pipeline.

## Article experiment

For a Medium/GitHub benchmark, run the same labelled dataset with:

1. SLM only
2. SLM + Jev
3. SLM + Ragas / LLM judge
4. larger LLM baseline

Track:

- accuracy
- precision
- recall
- F1
- false acceptance rate
- false rejection rate
- latency
- cost per 1,000 claims

**Important:** the offline verifier is only a runnable demo. Its scores must not be presented as Jev benchmark results.


# V2: Real vector RAG + student fine-tuning + teacher distillation

## Build the FAISS RAG database

```powershell
pip install -r requirements-rag.txt
python scripts/build_faiss.py
$env:RETRIEVER_MODE="faiss"
python run_cli.py "Can I return opened headphones?"
```

The vector store uses:

```text
SentenceTransformer embeddings
        ↓
FAISS cosine/IP index
        ↓
Top-k evidence
```

## Run a real Hugging Face SLM

```powershell
pip install -r requirements-training.txt

$env:GENERATOR_MODE="hf"
$env:HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

python run_cli.py "What is the warranty period?"
```

## Teacher → Student distillation

Create seed examples:

```powershell
python training/make_student_dataset.py
```

Generate teacher answers:

```powershell
python training/distill_generate.py `
  --teacher Qwen/Qwen2.5-1.5B-Instruct
```

Train the smaller student using LoRA:

```powershell
python training/train_lora.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --data data/distilled_sft.jsonl `
  --output artifacts/student-lora
```

Use the trained adapter in the RAG pipeline:

```powershell
$env:RETRIEVER_MODE="faiss"
$env:GENERATOR_MODE="hf"
$env:HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
$env:HF_ADAPTER="artifacts/student-lora"

python run_cli.py "Can I return opened headphones?"
```

## Optional QLoRA

```powershell
pip install -r requirements-qlora.txt
python training/train_qlora.py
```

`bitsandbytes` can be environment-sensitive on Windows. Plain LoRA is the
recommended first path for this project.

## Recommended experiment matrix

| Experiment | Retriever | Generator | Training | Verifier |
|---|---|---|---|---|
| A | FAISS | Base SLM | none | none/offline |
| B | FAISS | Base SLM | none | Jev |
| C | FAISS | Student SLM | teacher-distilled LoRA | none/offline |
| D | FAISS | Student SLM | teacher-distilled LoRA | Jev |
| E | FAISS | Student SLM | teacher-distilled LoRA | Ragas/LLM judge |

Measure accuracy, faithfulness, hallucination rate, F1, latency, and cost.

## Architecture

```text
Documents
   ↓
Chunk / embed
   ↓
FAISS
   ↓
Top-k evidence
   ↓
Student SLM
   ↓
Claims
   ↓
Jev / Ragas / other verifier
   ↓
Final answer
```

Teacher-distillation path:

```text
Evidence + Question
        ↓
Large Teacher LLM
        ↓
Grounded teacher answer
        ↓
Distillation dataset
        ↓
LoRA / QLoRA
        ↓
Small Student LLM
```


# V3: Hugging Face benchmark datasets

Install:

```powershell
pip install -r requirements-benchmark.txt
```

Recommended first real benchmark:

```powershell
python benchmarks/run_verifier_benchmark.py `
  --dataset contract_nli `
  --limit 100
```

This normalizes Hugging Face examples into:

```text
evidence + claim + human ground-truth label
```

and maps NLI labels to:

```text
entailment    -> SUPPORTED
contradiction -> CONTRADICTED
neutral       -> INSUFFICIENT
```

Use the same dataset to compare the offline verifier and real Jev.

For SLM experiments:

```powershell
python benchmarks/run_generation_benchmark.py --limit 50
```

Then set:

```powershell
$env:HF_ADAPTER="artifacts/student-lora"
```

and rerun to compare the distilled student against the base SLM.


# V4 Accuracy Improvements
See `README_V4.md` and `training/README_IMPROVEMENT_LOOP.md`.


## V5
See `RUN_V5.md` for the complete teacher-enrichment → QLoRA → Jev → hard-mining → retraining → promotion workflow.


## Teacher validation
Teacher-enriched examples are now validated against human labels before QLoRA. Optional Jev validation can filter unsupported teacher outputs.
