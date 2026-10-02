from app.datasets.common import VerificationExample

TEXT_MAP = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT",
    "supported": "SUPPORTED",
    "contradicted": "CONTRADICTED",
    "insufficient": "INSUFFICIENT",
}

def _label_name(ds, raw_label):
    try:
        feature = ds.features["label"]
        if hasattr(feature, "int2str") and isinstance(raw_label, int):
            return feature.int2str(raw_label)
    except Exception:
        pass
    return str(raw_label)

def load_contract_nli(limit: int = 100, split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset("tasksource/contract-nli", "contractnli_a", split=split)
    examples = []

    for row in ds:
        evidence = (
            row.get("premise")
            or row.get("contract")
            or row.get("context")
            or row.get("document")
        )
        claim = row.get("hypothesis") or row.get("claim")

        raw = row.get("label")
        label_text = _label_name(ds, raw).strip().lower()
        label = TEXT_MAP.get(label_text)

        if not evidence or not claim or not label:
            continue

        examples.append(
            VerificationExample(
                evidence=str(evidence),
                claim=str(claim),
                label=label,
                source="contract_nli",
            )
        )
        if len(examples) >= limit:
            break

    return examples
