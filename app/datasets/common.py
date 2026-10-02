from dataclasses import dataclass

@dataclass
class VerificationExample:
    evidence: str
    claim: str
    label: str
    source: str
