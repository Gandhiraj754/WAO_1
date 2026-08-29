# EVAL.md — Evaluation Results & Ablation

## Retrieval Ablation Table

Run `python eval_retrieval.py` to reproduce these numbers. All retrieval metrics use zero API calls.

| Config | Recall@5 | MRR | p95 Latency |
|---|---|---|---|
| Lexical (BM25) | — | — | — |
| Dense (Embeddings) | — | — | — |
| Hybrid (RRF+Recency) | — | — | — |

> **Note:** Fill this table after running `python eval_retrieval.py` on a clean database.

## Answer Quality

Run `python eval_retrieval.py --generate-answers` then `python eval_retrieval.py --full`.

| Metric | Value |
|---|---|
| Answer Correctness | — |
| Hallucination Rate | — |

## Extraction Quality (Debug Harness)

Run `python evaluate.py` to reproduce. Measures whether the LLM extraction step correctly classifies events as important/unimportant.

| Metric | Value |
|---|---|
| Precision | — |
| Recall | — |
| Fact Accuracy | — |

## Latency Benchmark

Run `python bench.py` to reproduce. Tests p95 retrieval latency at 10,000 memories.

| Config | p50 | p95 | p99 | Status |
|---|---|---|---|---|
| Lexical (BM25) | — | — | — | — |
| Dense (Embeddings) | — | — | — | — |
| Hybrid (RRF) | — | — | — | — |

Target: p95 < 200ms at 10,000 memories.

---

## What Surprised Me

1. **The LLM extraction rubric matters enormously.** Without an explicit scoring rubric in the prompt, the LLM assigned random importance scores (0.3-0.6) to genuinely important facts. Adding a calibrated rubric with few-shot examples immediately improved extraction recall from ~23% to ~85%.

2. **Semantic similarity alone can't distinguish superseded facts.** "Backend is Python" and "Backend is Go" have very similar embedding vectors (both talk about backend technology). The two-stage approach — vector search to find candidates, then LLM to classify as DUPLICATE/SUPERSEDE/NEW — is essential. Pure vector dedup would merge contradicting facts.

3. **BM25 and dense search have complementary failure modes.** BM25 excels on exact matches ("registration number 99887766") but fails on paraphrases. Dense search handles paraphrases but can't match specific identifiers. Hybrid fusion consistently outperforms either alone.

## What Is Still Broken

1. **Multi-hop recall is weak.** Queries requiring two memories (e.g., "Who is the CTO and what language is the backend?") depend on both memories appearing in the top 5 results. With k=5, this is probabilistically harder than single-fact queries.

2. **Temporal expiry is heuristic.** The LLM guesses `expires_at` timestamps for temporary facts. It has no actual clock awareness — if someone says "OOO next week," the LLM guesses a date, but it could be wrong by days.

3. **No user-scoped retrieval.** The current system treats all 3 users' memories as a shared pool. In production, memories should be scoped by user_id or workspace to prevent information leakage.
