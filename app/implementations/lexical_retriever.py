import json
import re
from pathlib import Path
from app.interfaces.retriever import Retriever

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "policies.json"

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

class LexicalRetriever(Retriever):
    def __init__(self, data_file: Path | None = None):
        self.data_file = data_file or DATA_FILE
        self.documents = [x["text"] for x in json.loads(self.data_file.read_text(encoding="utf-8"))]

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q = _tokens(query)
        scored = [(len(q & _tokens(doc)), doc) for doc in self.documents]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [doc for score, doc in scored[:k] if score > 0]
        return out or self.documents[:1]
