import argparse
from src.config import Config
from src.data_extractor import GitDataExtractor
from src.indexers.bm25_indexer import BM25Indexer
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.rerankers.llm_reranker import LLMReranker
from src.evaluation.benchmark import BenchmarkGenerator
from src.evaluation.metrics import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Git History IR System Evaluator")
    parser.add_argument("--strategy", choices=["bm25", "llm", "ce"], default="bm25")
    parser.add_argument("--fresh", action="store_true", help="Force fresh extraction of git logs")
    args = parser.parse_args()

    if args.fresh:
        extractor = GitDataExtractor(
            repo_url=Config.REPO_URL,
            repo_dir=Config.REPO_DIR,
            output_file=Config.OUTPUT_FILE
        )
        extractor.extract()

    indexer = BM25Indexer(Config.OUTPUT_FILE)
    benchmark = BenchmarkGenerator(Config.OUTPUT_FILE).generate(size=Config.BENCHMARK_SIZE)
    evaluator = Evaluator(benchmark)

    if args.strategy == "bm25":
        score = evaluator.calculate_mrr(search_pipeline_func=indexer.search)
        print(f"BM25 MRR@10: {score:.4f}")

    elif args.strategy == "llm":
        reranker = LLMReranker(
            model_name=Config.LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            api_key=Config.API_KEY
        )
        score = evaluator.calculate_mrr(
            search_pipeline_func=lambda q: reranker.rerank(q, indexer.search(q, top_k=25), top_n=10)
        )
        print(f"LLM MRR@10: {score:.4f}")

    elif args.strategy == "ce":
        reranker = CrossEncoderReranker()
        score = evaluator.calculate_mrr(
            search_pipeline_func=lambda q: reranker.rerank(q, indexer.search(q, top_k=25), max_context_length=256)[:10]
        )
        print(f"Cross-Encoder MRR@10: {score:.4f}")

if __name__ == "__main__":
    main()
