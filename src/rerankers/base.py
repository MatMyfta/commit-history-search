from abc import ABC, abstractmethod

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        """Evaluates candidate documents against a query and optimizes their ranking order."""
        pass
