from fastapi import FastAPI
from .models import AskRequest, AskResponse
from .pipeline import run_pipeline

app = FastAPI(title="SLM + Jev RAG Verifier", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return run_pipeline(req.question)
