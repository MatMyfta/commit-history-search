# Commit History IR: Code Repository Search Engine

An advanced Information Retrieval (IR) pipeline designed to navigate and mine historical git commit logs using natural language. This project implements a **Multi-Stage Cascading Architecture** to solve the vocabulary mismatch problem in software engineering repos, matching abstract developer intents to concrete source code modifications and bug fixes.

---

## Architecture Overview

The system processes queries using a two-stage computational funnel to balance retrieval speeds with high-precision semantic understanding:

1. **Stage 1: Lexical Candidate Pruning (High Recall)**
   * Utilizes an inverted index backed by the **BM25 (Okapi)** statistical term-frequency algorithm.
   * Sweeps thousands of raw git commits in milliseconds, filtering the repository down to a candidate pool of the top 100 most relevant items.
2. **Stage 2: Neural Re-ranking (High Precision)**
   * Intercepts the top candidates and passes them to a deep learning engine to evaluate deep semantic dependencies.
   * Supports two decoupled execution modules: a discriminative **Cross-Encoder** running locally (`bge-reranker-v2-m3`) or a generative **Listwise LLM** pipeline routed via cloud infrastructure.
   * Renders the final globally optimized sequence bounded at `top_n = 10`.

---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/MatMyfta/commit-history-search.git
cd git-history
```

### Configure Environment

Create a virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Instructions

Execute the primary runtime driver main.py using the --strategy flag to benchmark different components of the search configuration.

Run Lexical Baseline (BM25)

```bash
python main.py --strategy bm25
```

Run Cross-Encoder Re-ranker

```bash
python main.py --strategy ce
```

Run Listwise LLM Re-ranker

```bash
python main.py --strategy llm
```
