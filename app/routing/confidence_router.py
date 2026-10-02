from dataclasses import dataclass


@dataclass
class RoutingDecision:
    action: str
    reason: str


class ConfidenceRouter:
    def __init__(self, supported_threshold=0.85, insufficient_threshold=0.70):
        self.supported_threshold = supported_threshold
        self.insufficient_threshold = insufficient_threshold

    def decide(self, verdict, confidence):
        if verdict == "SUPPORTED" and confidence >= self.supported_threshold:
            return RoutingDecision("RETURN", "supported with sufficient confidence")
        if verdict == "INSUFFICIENT" and confidence >= self.insufficient_threshold:
            return RoutingDecision("RETRIEVE", "evidence is insufficient")
        return RoutingDecision("ESCALATE", "low confidence or contradiction")
