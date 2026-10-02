from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        raise NotImplementedError
