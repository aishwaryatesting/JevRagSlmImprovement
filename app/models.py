from pydantic import BaseModel
from typing import List, Literal, Optional

Verdict = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]

class AskRequest(BaseModel):
    question: str

class ClaimCheck(BaseModel):
    claim: str
    evidence: str
    verdict: Verdict
    confidence: float
    verifier: str

class AskResponse(BaseModel):
    question: str
    retrieved_evidence: List[str]
    draft_answer: str
    checks: List[ClaimCheck]
    final_answer: str
