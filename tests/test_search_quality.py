import os
import pytest
from src.config import Config
from src.indexers.bm25_indexer import BM25Indexer
from src.rerankers.cross_encoder import CrossEncoderReranker

@pytest.fixture(scope="module")
def search_system():
    """Initializes the index and reranker once for all search quality tests."""
    if not os.path.exists(Config.OUTPUT_FILE):
        pytest.skip(f"Data file missing at {Config.OUTPUT_FILE}. Run extraction first.")

    indexer = BM25Indexer(Config.OUTPUT_FILE)
    reranker = CrossEncoderReranker()
    return indexer, reranker

QUALITY_TEST_CASES = [
    (
        "fixing errors with infinite numbers or NaN values",
        "6c0566bd", # Substring of the expected GSON commit hash resolving NaN
        5
    ),
    (
        "handling non-English characters encoding issues",
        "46b73632", # Example hash snippet for Unicode/escape handling patches
        5
    ),
    (
        "custom layout for rendering object fields to JSON",
        "bf549f05", # Example hash snippet for custom TypeAdapters
        5
    )
]

@pytest.mark.parametrize("query,expected_id_snippet,top_n", QUALITY_TEST_CASES)
def test_cross_encoder_search_quality(search_system, query, expected_id_snippet, top_n):
    indexer, reranker = search_system

    candidates = indexer.search(query, top_k=50)

    results = reranker.rerank(query, candidates, top_n=top_n)

    returned_ids = [commit["id"] for commit in results]

    is_found = any(expected_id_snippet in rid for rid in returned_ids)

    assert is_found, (
        f"Search Quality Regression Failure!\n"
        f"Query: '{query}'\n"
        f"Expected commit snippet '{expected_id_snippet}' to be in Top {top_n}.\n"
        f"Instead got Top {top_n}: {[rid[:8] for rid in returned_ids]}"
    )
