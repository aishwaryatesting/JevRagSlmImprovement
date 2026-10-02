# Accuracy improvement loop

1. Start with clean, balanced FEVER/ContractNLI data.
2. Add negations, exceptions, dates/numbers, conditional statements and hard negatives.
3. Optionally enrich with a stronger teacher for rationales/corrections.
4. QLoRA fine-tune a 1.5B or 3B student on evidence + claim -> verdict + explanation.
5. Evaluate on a held-out set.
6. Mine mistakes/low-confidence cases with `mine_hard_examples.py`.
7. Mix about 80% clean + 20% hard examples with `build_mixed_training_set.py`.
8. QLoRA fine-tune again.
9. In production use hybrid retrieval -> reranker -> top-k context -> SLM -> Jev -> confidence routing.

Do not train only on hard examples; retain clean/general data to reduce overfitting and forgetting.
