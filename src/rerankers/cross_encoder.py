import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.rerankers.base import BaseReranker

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _get_relevance_score(self, query: str, commit_message: str) -> float:
        pairs = [[query, commit_message]]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            logits = self.model(**inputs).logits.view(-1).float()
            probabilities = torch.sigmoid(logits)
        return probabilities[0].item()

    def rerank(self, query: str, candidates: list, top_n: int = 10) -> list:
        scored_candidates = []
        for commit in candidates:
            score = self._get_relevance_score(query, commit["message"])
            scored_candidates.append((commit, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [commit for commit, score in scored_candidates[:top_n]]
