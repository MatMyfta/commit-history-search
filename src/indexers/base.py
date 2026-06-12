from abc import ABC, abstractmethod

class BaseIndexer(ABC):
    @abstractmethod
    def build_index(self) -> None:
        """Parses the raw document data and builds the primary retrieval index."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int) -> list:
        """Executes a search query and returns the top_k candidate documents."""
        pass
