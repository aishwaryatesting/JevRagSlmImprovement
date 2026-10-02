from app.datasets.common import VerificationExample

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "INSUFFICIENT",
}

def load_fever(limit: int = 100, split: str = "labelled_dev"):
    """
    Loads FEVER from Hugging Face when textual evidence is directly available.
    Some FEVER configurations expose Wikipedia sentence IDs rather than resolved
    text. Those rows are skipped by this simple loader.
    """
    from datasets import load_dataset

    ds = load_dataset("fever", "v1.0", split=split)
    examples = []

    for row in ds:
        claim = row.get("claim")
        label = row.get("label")
        mapped = LABEL_MAP.get(label)

        if not claim or not mapped:
            continue

        evidence_texts = []
        ev = row.get("evidence")

        if isinstance(ev, str) and ev.strip():
            evidence_texts.append(ev.strip())
        elif isinstance(ev, list):
            for item in ev:
                if isinstance(item, str) and item.strip():
                    evidence_texts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("sentence")
                    if text:
                        evidence_texts.append(str(text).strip())

        if not evidence_texts:
            continue

        examples.append(
            VerificationExample(
                evidence="\n".join(evidence_texts),
                claim=str(claim),
                label=mapped,
                source="fever",
            )
        )
        if len(examples) >= limit:
            break

    return examples
