import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
data_file = ROOT / "data" / "policies.json"
out_dir = ROOT / "artifacts"
out_dir.mkdir(exist_ok=True)

docs = [x["text"] for x in json.loads(data_file.read_text(encoding="utf-8"))]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(out_dir / "policies.faiss"))
(out_dir / "policies_meta.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")

print(f"Built FAISS index with {len(docs)} documents")
