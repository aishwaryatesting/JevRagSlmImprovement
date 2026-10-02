import re

def draft_answer(question: str, evidence: list[str]) -> str:
    q = question.lower()
    joined = " ".join(evidence)

    # Deterministic "small model" stand-in so the project runs without an LLM key.
    # Replace this function later with Phi/Qwen/Gemma/OpenAI-compatible inference.
    if "headphone" in q and "return" in q:
        return "Opened headphones can be returned only if they are defective."
    if "refund" in q and "day" in q:
        return "Refund requests are accepted within 30 days of purchase."
    if "warranty" in q:
        return "The standard warranty period is one year."
    if "shipping" in q:
        return "I do not have enough evidence to state that international shipping is free."

    # Evidence-grounded fallback
    first = evidence[0].strip()
    return f"Based on the available policy: {first}"

def split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]
