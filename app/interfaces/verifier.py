from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class VerificationResult:
    verdict: str
    confidence: float
    verifier: str

class Verifier(ABC):
    @abstractmethod
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        raise NotImplementedError
