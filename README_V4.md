# V4 accuracy-focused pipeline

Adds: balanced/hard data workflow, teacher enrichment support, hybrid BM25+dense retrieval, cross-encoder reranking, task-specific LoRA/QLoRA, hard-example mining, confidence gating, Jev verification, top-k context discipline, deterministic decoding, and configurable 0.5B/1.5B/3B student models.

Recommended production experiment:

```text
User -> Hybrid RAG -> Cross-encoder reranker -> Top-3 evidence -> Fine-tuned 1.5B SLM -> Jev -> confidence gate
                                                                      |-> return
                                                                      |-> retrieve more
                                                                      `-> large LLM/human
```

Install advanced dependencies:

```powershell
pip install -r requirements-advanced.txt
$env:RETRIEVER_MODE="hybrid"
$env:RERANKER_MODE="cross_encoder"
$env:GENERATOR_MODE="hf"
$env:HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
python run_cli.py "Can I return opened headphones?"
```

For iterative hard-example training see `training/README_IMPROVEMENT_LOOP.md`.
