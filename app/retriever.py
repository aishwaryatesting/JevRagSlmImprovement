import json
import re
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "policies.json"

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def load_policies():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))

def retrieve(question: str, k: int = 3) -> list[str]:
    q = _tokens(question)
    scored = []
    for item in load_policies():
        text = item["text"]
        score = len(q & _tokens(text))
        scored.append((score, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [text for score, text in scored[:k] if score > 0]
    return results or [load_policies()[0]["text"]]
