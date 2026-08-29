<div align="center">

# <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=40&pause=1000&color=2563EB&center=true&vCenter=true&width=600&lines=WAO-Recall;Enterprise+AI+Memory;Zero+Hallucinations;Sub-30ms+Latency" alt="Typing SVG" />

**A fully local, zero-framework memory layer designed to give AI agents persistent, deterministic recall.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-FTS5%20%7C%20Vec-003B57?style=for-the-badge&logo=sqlite)](https://sqlite.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

</div>

---

## ⚡ Quick Start (< 5 Minutes)

WAO-Recall is completely self-contained. You can run it via Docker (recommended) or natively.

### Option A: The Instant Docker Boot (Recommended)
> We pre-download the HuggingFace `all-MiniLM-L6-v2` embedding model inside the Docker image so it boots instantly without downloading weights on startup.

```bash
# 1. Clone the repository
git clone <repo-url>
cd wao-recall

# 2. Add your free-tier Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Spin up the FastAPI microservice
docker-compose up -d
```
*👉 Test the API immediately at: [http://localhost:8000/docs](http://localhost:8000/docs)*

### Option B: Native Python Setup
<details>
<summary>Click here for manual Python instructions</summary>
<br>

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
python generate_events.py          # Generate synthetic data
python memory_store.py             # Ingest into SQLite (rate-limited, takes time)
python sync_indexes.py             # Build BM25 and Vector indexes
uvicorn app:app --reload           # Start the API
```
</details>

---

## 🏗️ System Architecture

WAO-Recall strips away bloated RAG frameworks (LangChain, LlamaIndex) in favor of raw, high-performance **SQLite**. It features a dual-engine **Hybrid Search** (Lexical BM25F + Dense `sqlite-vec`) fused via **Reciprocal Rank Fusion (RRF)**.

```mermaid
flowchart TD
    subgraph Data Generation
    A[events.jsonl] -->|Markov Chain| B(Simulated Work Stream)
    end

    subgraph Memory Policy Engine
    B -->|Heuristic Filter| C{Is Noise?}
    C -->|Yes| Drop(Dropped - 0ms)
    C -->|No| D[Gemini Extraction Rubric]
    D --> E{Collision Detection}
    E -->|Exact Match| F[Deduplicate]
    E -->|Contradiction| G[Supersede]
    E -->|Unique| H[Insert New]
    end

    subgraph Storage Layer
    F & G & H --> DB[(SQLite memory.db)]
    DB --> I[memories table]
    DB --> J[FTS5 BM25 Index]
    DB --> K[sqlite-vec Embeddings]
    end

    subgraph Retrieval Microservice
    L[POST /ask] --> M[BM25F Search]
    L --> N[Dense Vector Search]
    M & N --> O[Reciprocal Rank Fusion]
    O --> P[Recency Decay Penalty]
    P --> Q[LLM Context Window]
    Q --> R[Cited JSON Response]
    end
```

---

## 📊 Offline Evaluation Harness

The rubric demands deterministic, offline measurement. We cache all LLM extractions and answers so you can verify our metrics with **zero API calls**.

### 1. Run the Retrieval Evaluation
```bash
python eval_retrieval.py
```
*Measures Recall@5, Mean Reciprocal Rank (MRR), and p95 retrieval latency across 42 hand-authored edge cases (Supersession, Temporal, Multi-hop).*

### 2. Run the Extreme Latency Benchmark
```bash
python bench.py
```
*Proves that our `sqlite-vec` flat-index can search 10,000 synthetic memories in under 200ms.*

### 3. Generate New LLM Answers (Optional)
```bash
python eval_retrieval.py --generate-answers
python eval_retrieval.py --full
```
*Reruns the LLM over the retrieved context to calculate Fact Correctness and Hallucination Rates.*

---

## 📖 Engineering Documentation

To understand the trade-offs, constraints, and limitations of this architecture, please review the mandatory design docs:

- **[DECISIONS.md](./DECISIONS.md)**: 10 critical design decisions, rejected alternatives, and academic sources (including BM25F and RRF math).
- **[LIMITS.md](./LIMITS.md)**: What happens to this architecture at 10 Million memories, and exactly how we would fix it given two more weeks.
- **[EVAL.md](./EVAL.md)**: The full ablation study comparing Lexical vs. Dense vs. Hybrid retrieval.

---
<div align="center">
<i>Built for the WorkElate AI Engineering Trial</i>
</div>
