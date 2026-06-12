import argparse
from src.config import Config
from src.indexers.bm25_indexer import BM25Indexer
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.rerankers.llm_reranker import LLMReranker
from src.evaluation.benchmark import BenchmarkGenerator
from src.evaluation.metrics import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Git History IR System Evaluator")
    parser.add_argument("--strategy", choices=["bm25", "llm-hybrid", "ce-hybrid"], default="bm25")
    args = parser.parse_args()

    indexer = BM25Indexer(Config.OUTPUT_FILE)
    benchmark = BenchmarkGenerator(Config.OUTPUT_FILE).generate(size=20)
    evaluator = Evaluator(benchmark)

    if args.strategy == "bm25":
        score = evaluator.calculate_mrr(search_pipeline_func=indexer.search)
        print(f"BM25 MRR@10: {score:.4f}")

    elif args.strategy == "llm-hybrid":
        reranker = LLMReranker(
            model_name=Config.LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL
        )

        def hybrid_pipeline(query, top_k=5):
            candidates = indexer.search(query, top_k=100)
            return reranker.rerank(query, candidates, top_n=top_k)

        score = evaluator.calculate_mrr(
            search_pipeline_func=lambda q: reranker.rerank(q, indexer.search(q, top_k=100), top_n=5)
        )
        print(f"LLM Hybrid MRR@10: {score:.4f}")

    elif args.strategy == "ce-hybrid":
        reranker = CrossEncoderReranker()

        def ce_pipeline(query, top_k=5):
            candidates = indexer.search(query, top_k=100)
            return reranker.rerank(query, candidates, top_n=top_k)

        score = evaluator.calculate_mrr(
            search_pipeline_func=lambda q: reranker.rerank(q, indexer.search(q, top_k=100), top_n=5)
        )
        print(f"Cross-Encoder Hybrid MRR@10: {score:.4f}")

if __name__ == "__main__":
    main()
