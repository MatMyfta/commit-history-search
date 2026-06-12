import json
import re

class BenchmarkGenerator:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())

    def generate(self, size: int = 20) -> list:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)

        eval_set = []
        for commit in commits:
            summary = commit["summary"].strip()
            tokens = self._tokenize(summary)

            if len(tokens) >= 5 and not summary.startswith("[") and not "test" in summary.lower():
                eval_set.append({
                    "query": summary,
                    "expected_id": commit["id"]
                })

        return eval_set[:size]
