import sys
from src.config import Config
from src.indexers.bm25_indexer import BM25Indexer
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.rerankers.llm_reranker import LLMReranker

def print_results(title, results):
    print(f"\n===== {title.upper()} TOTAL RESULTS =====")
    for idx, commit in enumerate(results[:5]):
        print(f"[{idx+1}] Score/Rank | ID: {commit['id'][:8]} | Author: {commit['author']}")
        print(f"    Summary: {commit['summary']}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py 'your search query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Querying IR systems for: '{query}'\n")

    print("Loading BM25 Index...")
    indexer = BM25Indexer(Config.OUTPUT_FILE)

    bm25_results = indexer.search(query, top_k=5)
    print_results("System 1: Pure BM25 Baseline", bm25_results)

    candidates = indexer.search(query, top_k=50)

    print("\nRunning Cross-Encoder Reranker...")
    ce_reranker = CrossEncoderReranker()
    ce_results = ce_reranker.rerank(query, candidates, top_n=5)
    print_results("System 2: Cross-Encoder Hybrid", ce_results)

    print("\nRunning Local LLM Reranker (Ollama)...")
    llm_reranker = LLMReranker(model_name=Config.LLM_MODEL, base_url=Config.OPENAI_BASE_URL)
    llm_results = llm_reranker.rerank(query, candidates, top_n=5)
    print_results("System 3: Local LLM Hybrid", llm_results)

if __name__ == "__main__":
    main()
