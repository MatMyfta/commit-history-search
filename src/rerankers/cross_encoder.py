import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def _get_relevance_scores(self, text_pairs: list, max_context_length: int) -> list:
        inputs = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            max_length=max_context_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.view(-1).float()
            return logits.cpu().numpy().tolist()

    def rerank(self, query: str, candidates: list, max_context_length: int = 1024) -> list:
        if not candidates:
            return []

        try:
            text_pairs = []
            for c in candidates:
                summary = c.get("summary", "")
                body = c.get("body", "")
                doc_representation = f"Summary: {summary}\nBody: {body}".strip()
                text_pairs.append([query, doc_representation])

            scores = self._get_relevance_scores(text_pairs, max_context_length)

            isolated_mutated_candidates = []
            for candidate, score in zip(candidates, scores):
                mutated_copy = candidate.copy()
                mutated_copy["search_score"] = score
                isolated_mutated_candidates.append(mutated_copy)

            return sorted(isolated_mutated_candidates, key=lambda x: x["search_score"], reverse=True)

        except Exception as e:
            print(f"\n[CRITICAL ERROR in CrossEncoder]: {str(e)}\n")
            return sorted(candidates, key=lambda x: x.get("search_score", 0.0), reverse=True)
