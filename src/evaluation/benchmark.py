import json
import re
import requests

class BenchmarkGenerator:
    def __init__(self, data_path: str, ollama_url: str = "http://localhost:11434/api/generate"):
        self.data_path = data_path
        self.ollama_url = ollama_url

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())

    def _mutate_query_via_llm(self, summary: str) -> str:
        prompt = (
            f"You are a software engineer searching a git repository.\n"
            f"Transform this commit message summary into a casual, conversational search query "
            f"that a developer would type into a search engine. Do NOT use the exact same words or class names.\n"
            f"Commit summary: '{summary}'\n"
            f"Search query:"
        )

        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                mutated_query = response.json().get("response", "").strip()
                return mutated_query.strip('"').strip("'")
        except Exception:
            pass

        return summary # Fallback if local LLM is offline

    def generate(self, size: int = 20) -> list:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)

        eval_set = []
        for commit in commits:
            summary = commit.get("summary", "").strip()
            body = commit.get("body", "").strip()
            tokens = self._tokenize(summary)

            if len(tokens) >= 5 and len(body) > 20 and not summary.startswith("[") and not "test" in summary.lower():
                print(f"Mutating query {len(eval_set) + 1}/{size}: {summary}")
                mutated_query = self._mutate_query_via_llm(summary)

                eval_set.append({
                    "query": mutated_query,
                    "original_summary": summary,
                    "expected_id": commit["id"]
                })

            if len(eval_set) >= size:
                break

        return eval_set
