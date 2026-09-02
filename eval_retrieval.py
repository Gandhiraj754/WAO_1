"""
WAO-Recall Retrieval Evaluation Harness
========================================
This is the graded core of the project. It runs 42 hand-authored queries
against all 3 retrieval configurations and reports:
- Recall@5 and MRR (per config)
- Answer Correctness (%)
- Hallucination Rate (%)
- p95 Retrieval Latency (ms)

Usage:
  python eval_retrieval.py                  # Retrieval metrics only (0 API calls)
  python eval_retrieval.py --generate-answers  # Also generates + caches LLM answers (uses API)
  python eval_retrieval.py --full           # Full report with cached answers
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import time
import argparse
import sqlite3
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from retrieval import lexical_search, dense_search, hybrid_search, ask_question, get_db

QUERIES_PATH = "data/eval_queries.jsonl"
ANSWER_CACHE_PATH = "data/answer_cache.json"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_queries():
    with open(QUERIES_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_fact_for_memory(db, memory_id):
    """Fetch the fact text for a memory_id from the database."""
    cursor = db.cursor()
    cursor.execute("SELECT fact FROM memories WHERE memory_id = ?", (memory_id,))
    row = cursor.fetchone()
    return row[0] if row else ""


def enrich_results_with_facts(results, db):
    """Add fact text to dense_search results that only have memory_id + distance."""
    for r in results:
        if "fact" not in r or not r.get("fact"):
            r["fact"] = fetch_fact_for_memory(db, r["memory_id"])
    return results


def check_keywords_in_results(results, keyword_group):
    """Check if ALL keyword groups are matched by at least one result.
    
    keyword_group is a list of lists. Each inner list is a set of keywords
    that must ALL appear in a single result's fact text.
    Returns True only if every group is matched.
    """
    if not keyword_group:
        return True  # No keywords to match (e.g., must-return-nothing)

    for kw_set in keyword_group:
        found = False
        for result in results:
            fact = result.get("fact", "").lower()
            if all(kw.lower() in fact for kw in kw_set):
                found = True
                break
        if not found:
            return False
    return True


def get_reciprocal_rank(results, keyword_group):
    """Get the reciprocal rank. For multi-hop, uses the worst rank."""
    if not keyword_group:
        return 0.0

    worst_rank = 0
    for kw_set in keyword_group:
        best_rank_for_group = None
        for rank, result in enumerate(results, 1):
            fact = result.get("fact", "").lower()
            if all(kw.lower() in fact for kw in kw_set):
                best_rank_for_group = rank
                break
        if best_rank_for_group is None:
            return 0.0  # One group not found at all
        worst_rank = max(worst_rank, best_rank_for_group)

    return 1.0 / worst_rank if worst_rank > 0 else 0.0


def evaluate_retrieval(queries):
    """Evaluate Recall@5, MRR, and p95 latency for all 3 retrieval configs."""
    db = get_db()

    configs = {
        "Lexical (BM25)": lambda q, k: lexical_search(q, "u_sohil", k),
        "Dense (Embeddings)": lambda q, k: dense_search(q, "u_sohil", k),
        "Hybrid (RRF+Recency)": lambda q, k: hybrid_search(q, "u_sohil", k),
    }

    results_table = {}

    for config_name, search_fn in configs.items():
        recalls = []
        mrrs = []
        latencies = []

        for q in queries:
            if q["must_return_nothing"]:
                continue  # Recall/MRR don't apply to must-return-nothing

            start = time.perf_counter()
            raw_results = search_fn(q["question"], 5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            # Enrich dense results with fact text
            enriched = enrich_results_with_facts(raw_results, db)

            recall_hit = check_keywords_in_results(enriched, q["expected_memory_keywords"])
            recalls.append(1 if recall_hit else 0)

            rr = get_reciprocal_rank(enriched, q["expected_memory_keywords"])
            mrrs.append(rr)

        recall_at_5 = (sum(recalls) / len(recalls) * 100) if recalls else 0
        mrr = (sum(mrrs) / len(mrrs)) if mrrs else 0
        p95 = np.percentile(latencies, 95) if latencies else 0

        results_table[config_name] = {
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "p95_latency_ms": p95,
        }

    db.close()
    return results_table


def generate_answer_cache(queries):
    """Run ask_question for each query and cache the results. USES API."""
    cache = {}
    if os.path.exists(ANSWER_CACHE_PATH):
        with open(ANSWER_CACHE_PATH) as f:
            cache = json.load(f)

    total = len(queries)
    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        if qid in cache:
            print(f"[{i}/{total}] {qid} (cached)")
            continue

        print(f"[{i}/{total}] Asking: {q['question']}")
        time.sleep(4.2)  # Rate limiting
        try:
            result = ask_question("u_sohil", q["question"])
            cache[qid] = {
                "answer": result["answer"],
                "cited_memory_ids": result["cited_memory_ids"],
                "confidence": result["confidence"],
            }
        except Exception as e:
            print(f"  Error: {e}")
            cache[qid] = {"answer": "ERROR", "cited_memory_ids": [], "confidence": 0}

        # Save incrementally
        with open(ANSWER_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    print("Answer cache generation complete.")
    return cache


def evaluate_answers(queries, cache):
    """Grade answer correctness and hallucination rate from cached answers."""
    correct = 0
    incorrect = 0
    hallucinations = 0
    total_must_return_nothing = 0

    for q in queries:
        qid = q["query_id"]
        if qid not in cache:
            continue

        answer = cache[qid]["answer"].lower()
        expected = q["expected_answer"].lower()

        if q["must_return_nothing"]:
            total_must_return_nothing += 1
            if "don't have that in memory" in answer or "do not have" in answer:
                correct += 1
            else:
                hallucinations += 1
                incorrect += 1
        else:
            # Semantic similarity check
            vec_answer = embed_model.encode(answer)
            vec_expected = embed_model.encode(expected)
            cosine_sim = float(np.dot(vec_answer, vec_expected) / (norm(vec_answer) * norm(vec_expected)))

            if cosine_sim >= 0.65:
                correct += 1
            else:
                incorrect += 1

    total = correct + incorrect
    answer_correctness = (correct / total * 100) if total > 0 else 0
    hallucination_rate = (hallucinations / total_must_return_nothing * 100) if total_must_return_nothing > 0 else 0

    return {
        "answer_correctness": answer_correctness,
        "hallucination_rate": hallucination_rate,
        "correct": correct,
        "incorrect": incorrect,
        "hallucinations": hallucinations,
    }


def print_results(retrieval_results, answer_results=None):
    """Print the ablation table."""
    print("\n" + "=" * 75)
    print("  WAO-RECALL EVALUATION RESULTS")
    print("=" * 75)

    # Retrieval metrics table
    header = f"{'Config':<25} | {'Recall@5':>10} | {'MRR':>8} | {'p95 Latency':>12}"
    print(f"\n{header}")
    print("-" * 65)

    for name, metrics in retrieval_results.items():
        print(f"{name:<25} | {metrics['recall_at_5']:>9.1f}% | {metrics['mrr']:>8.3f} | {metrics['p95_latency_ms']:>9.1f} ms")

    print("-" * 65)

    # Answer metrics
    if answer_results:
        print(f"\n--- ANSWER QUALITY ---")
        print(f"Answer Correctness  : {answer_results['answer_correctness']:.1f}%")
        print(f"Hallucination Rate  : {answer_results['hallucination_rate']:.1f}%")
        print(f"  Correct: {answer_results['correct']}  |  Incorrect: {answer_results['incorrect']}  |  Hallucinations: {answer_results['hallucinations']}")
    else:
        print(f"\n--- ANSWER QUALITY ---")
        print(f"Run with --generate-answers first, then --full to see answer metrics.")

    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAO-Recall Retrieval Evaluation")
    parser.add_argument("--generate-answers", action="store_true",
                        help="Generate and cache LLM answers (uses API)")
    parser.add_argument("--full", action="store_true",
                        help="Include answer correctness from cache (no API calls)")
    args = parser.parse_args()

    queries = load_queries()
    print(f"Loaded {len(queries)} evaluation queries.")

    if args.generate_answers:
        print("\nGenerating answer cache (this uses the Gemini API)...")
        generate_answer_cache(queries)
        print("Done. Run with --full to see the complete report.")
    else:
        # Always run retrieval metrics (0 API calls)
        print("\nEvaluating retrieval across all 3 configurations...")
        retrieval_results = evaluate_retrieval(queries)

        answer_results = None
        if args.full and os.path.exists(ANSWER_CACHE_PATH):
            with open(ANSWER_CACHE_PATH) as f:
                answer_cache = json.load(f)
            answer_results = evaluate_answers(queries, answer_cache)

        print_results(retrieval_results, answer_results)
