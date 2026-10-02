import re
import os

from .factory import build_retriever, build_reranker, build_generator
from .verifier import verify
from .context_utils import trim_evidence
from .routing.confidence_router import ConfidenceRouter


def split_claims(answer):
    return [
        p.strip()
        for p in re.split(r"(?<=[.!?])\s+", answer.strip())
        if p.strip()
    ]


def aggregate_action(actions):
    """
    Produce one overall route for the complete answer.

    Priority:
      ESCALATE > RETRIEVE > RETURN
    """
    if "ESCALATE" in actions:
        return "ESCALATE"
    if "RETRIEVE" in actions:
        return "RETRIEVE"
    return "RETURN"


def build_grounded_final_answer(evidence, checks, overall_action):
    """
    Convert claim-level Jev checks into one final grounded response.

    We do NOT concatenate one routing message per claim.

    If every claim is safe to return, preserve the verified claims.
    Otherwise fall back to the retrieved evidence itself, which is the
    safest deterministic response because it does not introduce new claims.
    """
    supported_claims = [
        check["claim"]
        for check in checks
        if check["route"] == "RETURN"
        and check["verdict"] == "SUPPORTED"
    ]

    if overall_action == "RETURN" and supported_claims:
        return " ".join(supported_claims)

    # Deduplicate evidence while preserving retrieval order.
    unique_evidence = []
    seen = set()

    for item in evidence:
        clean = str(item).strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique_evidence.append(clean)

    if unique_evidence:
        # Keep final output concise and grounded.
        grounded = " ".join(unique_evidence[:2])
        return f"Based on the retrieved evidence: {grounded}"

    if overall_action == "RETRIEVE":
        return (
            "The available evidence is insufficient to answer reliably. "
            "Additional retrieval is required."
        )

    return (
        "The generated answer could not be verified reliably and requires "
        "escalation to a stronger model or human review."
    )


def run_pipeline(question):
    retriever = build_retriever()
    reranker = build_reranker()
    generator = build_generator()

    retrieved = retriever.retrieve(
        question,
        k=int(os.getenv("RETRIEVAL_K", "8")),
    )

    if reranker:
        retrieved = reranker.rerank(
            question,
            retrieved,
            top_k=int(os.getenv("RERANK_TOP_K", "3")),
        )

    evidence = trim_evidence(
        retrieved,
        int(os.getenv("MAX_CONTEXT_CHARS", "4000")),
        int(os.getenv("CONTEXT_TOP_K", "3")),
    )

    draft = generator.generate(question, evidence)
    claims = split_claims(draft)

    router = ConfidenceRouter(
        float(os.getenv("SUPPORTED_THRESHOLD", "0.85")),
        float(os.getenv("INSUFFICIENT_THRESHOLD", "0.70")),
    )

    checks = []
    actions = []
    ev = "\n".join(evidence)

    for claim in claims:
        result = verify(claim, ev)
        decision = router.decide(
            result.verdict,
            result.confidence,
        )

        checks.append(
            {
                "claim": claim,
                "evidence": ev,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "verifier": result.verifier,
                "route": decision.action,
                "route_reason": decision.reason,
            }
        )

        actions.append(decision.action)

    overall_action = aggregate_action(actions)

    final_answer = build_grounded_final_answer(
        evidence=evidence,
        checks=checks,
        overall_action=overall_action,
    )

    route_counts = {
        "RETURN": actions.count("RETURN"),
        "RETRIEVE": actions.count("RETRIEVE"),
        "ESCALATE": actions.count("ESCALATE"),
    }

    return {
        "question": question,
        "retrieved_evidence": evidence,
        "draft_answer": draft,
        "checks": checks,
        "aggregate_route": overall_action,
        "route_counts": route_counts,
        "final_answer": final_answer,
        "routing_summary": actions,
    }
