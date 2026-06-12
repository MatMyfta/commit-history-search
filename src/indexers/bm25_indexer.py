import json
import re
from rank_bm25 import BM25Okapi
from src.indexers.base import BaseIndexer

class BM25Indexer(BaseIndexer):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.commits = []
        self.bm25 = None
        self.build_index()

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

    def build_index(self) -> None:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.commits = json.load(f)

        tokenized_corpus = [self._advanced_tokenize(commit["message"]) for commit in self.commits]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 100) -> list:
        if self.bm25 is None:
                    print("Warning: BM25 index is not initialized.")
                    return []

        query_tokens = self._advanced_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_pairs = sorted(zip(self.commits, scores), key=lambda x: x[1], reverse=True)
        return [commit for commit, score in ranked_pairs[:top_k]]
