import json
from pathlib import Path
import numpy as np
from app.interfaces.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "artifacts" / "policies.faiss"
DEFAULT_META = ROOT / "artifacts" / "policies_meta.json"

class FaissRetriever(Retriever):
    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX,
        metadata_path: str | Path = DEFAULT_META,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_path))
        self.documents = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        emb = self.model.encode([query], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        _, ids = self.index.search(emb, min(k, len(self.documents)))
        return [self.documents[i] for i in ids[0] if i >= 0]
