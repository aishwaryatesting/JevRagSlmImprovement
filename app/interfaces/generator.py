from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, evidence: list[str]) -> str:
        raise NotImplementedError
