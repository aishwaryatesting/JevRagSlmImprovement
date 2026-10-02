from app.interfaces.generator import Generator

class DeterministicGenerator(Generator):
    def generate(self, question: str, evidence: list[str]) -> str:
        q = question.lower()
        if "headphone" in q and "return" in q:
            return "Opened headphones can be returned only if they are defective."
        if "refund" in q and "day" in q:
            return "Refund requests are accepted within 30 days of purchase."
        if "warranty" in q:
            return "The standard warranty period is one year."
        if "shipping" in q:
            return "I do not have enough evidence to state that international shipping is free."
        return f"Based on the available policy: {evidence[0]}"
