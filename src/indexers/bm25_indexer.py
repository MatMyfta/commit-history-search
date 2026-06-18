import json
import re
from rank_bm25 import BM25Okapi

class BM25Indexer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.bm25 = None
        self.corpus = []
        self._load_and_initialize()

    def _load_and_initialize(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_commits_json = json.load(f)
        self.build_index(raw_commits_json)

    def _advanced_tokenize(self, text: str) -> list:
        text = re.sub(r'\(?Closes gh-\d+\)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[0-9a-f]{7,40}\b', '', text)

        raw_tokens = re.findall(r'[a-zA-Z0-9]+', text)

        final_tokens = []
        for token in raw_tokens:
            splits = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            for split in splits:
                final_tokens.append(split.lower())

        return final_tokens

    def build_index(self, raw_commits_json: list) -> None:
        processed_corpus = []
        self.corpus = raw_commits_json

        for commit in raw_commits_json:
            document_body = commit.get("body", "").strip()
            tokenized_document = self._advanced_tokenize(document_body)
            processed_corpus.append(tokenized_document)

        self.bm25 = BM25Okapi(processed_corpus)

    def search(self, query: str, top_k: int = 25) -> list:
        if not self.bm25 or not self.corpus:
            return []

        tokenized_query = self._advanced_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_candidates = []
        for doc_idx, score in enumerate(scores):
            candidate_copy = self.corpus[doc_idx].copy()
            candidate_copy["search_score"] = float(score)
            scored_candidates.append(candidate_copy)

        sorted_candidates = sorted(scored_candidates, key=lambda x: x["search_score"], reverse=True)
        return sorted_candidates[:top_k]
