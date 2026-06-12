class Evaluator:
    def __init__(self, eval_sample: list):
        self.eval_sample = eval_sample

    def calculate_mrr(self, search_pipeline_func, top_k=10) -> float:
        rr_scores = []
        for item in self.eval_sample:
            results = search_pipeline_func(item["query"])

            rank = -1
            for rank_idx, commit in enumerate(results):
                if commit["id"] == item["expected_id"]:
                    rank = rank_idx + 1
                    break

            rr_scores.append(1.0 / rank if rank != -1 else 0.0)
        return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
