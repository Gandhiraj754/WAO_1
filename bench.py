"""
WAO-Recall Latency Benchmark
==============================
Proves that p95 retrieval latency stays under 200ms at 10,000 memories.
Inserts 10,000 synthetic memories into a temporary SQLite database,
then runs 100 random queries across all 3 search configurations.

Usage:
  python bench.py

Zero API calls. Pure local computation.
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sqlite3
import sqlite_vec
import struct
import time
import random
import string
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

BENCH_DB = "data/bench_temp.db"
NUM_MEMORIES = 10_000
NUM_QUERIES = 100
VECTOR_DIM = 384

model = SentenceTransformer('all-MiniLM-L6-v2')


def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def random_fact():
    """Generate a random fact string for benchmarking."""
    subjects = ["backend", "frontend", "database", "API", "server", "cache", "queue", "auth", "payment", "search"]
    attributes = ["technology", "status", "version", "provider", "language", "framework", "latency", "uptime"]
    values = ["Python", "Go", "Rust", "PostgreSQL", "Redis", "running", "v2.1", "AWS", "stable", "degraded"]
    return f"{random.choice(subjects)}'s {random.choice(attributes)} is {random.choice(values)}"


def random_query():
    """Generate a random query string for benchmarking."""
    questions = [
        "What technology does the backend use?",
        "What is the database status?",
        "Who is the lead engineer?",
        "What payment provider do we use?",
        "When was the company founded?",
        "What is the current deployment status?",
        "Which cloud provider are we on?",
        "What framework does the frontend use?",
        "Is the cache layer working?",
        "What version is the API on?",
    ]
    return random.choice(questions)


def setup_bench_db():
    """Create a temporary database with 10,000 memories."""
    if os.path.exists(BENCH_DB):
        os.remove(BENCH_DB)

    db = sqlite3.connect(BENCH_DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cursor = db.cursor()

    # Create tables matching production schema
    cursor.execute("""
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY,
            fact TEXT,
            entity TEXT,
            attribute TEXT,
            value TEXT,
            status TEXT DEFAULT 'CURRENT',
            superseded_by TEXT,
            created_at TEXT,
            expires_at TEXT
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            memory_id UNINDEXED,
            fact
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE memories_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding float[384]
        )
    """)

    print(f"Inserting {NUM_MEMORIES} synthetic memories...")
    random.seed(42)
    np.random.seed(42)

    for i in range(NUM_MEMORIES):
        mem_id = f"bench_mem_{i:05d}"
        fact = random_fact()

        # Use real embeddings for first 100, random vectors for rest (speed)
        if i < 100:
            vec = model.encode(fact).tolist()
        else:
            vec = np.random.randn(VECTOR_DIM).astype(np.float32).tolist()

        cursor.execute(
            "INSERT INTO memories (memory_id, fact, status, created_at) VALUES (?, ?, 'CURRENT', '2026-01-01T00:00:00Z')",
            (mem_id, fact)
        )
        cursor.execute(
            "INSERT INTO memories_fts (memory_id, fact) VALUES (?, ?)",
            (mem_id, fact)
        )
        cursor.execute(
            "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
            (mem_id, serialize_f32(vec))
        )

        if (i + 1) % 2000 == 0:
            print(f"  Inserted {i + 1}/{NUM_MEMORIES}...")

    db.commit()
    print(f"Database ready with {NUM_MEMORIES} memories.\n")
    return db


def bench_lexical(db, query):
    """BM25 search benchmark."""
    import re
    cursor = db.cursor()
    safe_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip()
    if not safe_query:
        safe_query = "dummy"
    cursor.execute("""
        SELECT f.memory_id, f.fact, bm25(f.memories_fts) as score
        FROM memories_fts f
        JOIN memories m ON f.memory_id = m.memory_id
        WHERE f.memories_fts MATCH ?
        AND m.status = 'CURRENT'
        ORDER BY score
        LIMIT 5
    """, (safe_query,))
    return cursor.fetchall()


def bench_dense(db, query):
    """Semantic search benchmark."""
    cursor = db.cursor()
    query_vec = model.encode(query).tolist()
    cursor.execute("""
        SELECT v.memory_id, v.distance
        FROM memories_vec v
        JOIN memories m ON v.memory_id = m.memory_id
        WHERE v.embedding MATCH ? AND v.k = 5
        AND m.status = 'CURRENT'
        ORDER BY v.distance
    """, (serialize_f32(query_vec),))
    return cursor.fetchall()


def bench_hybrid(db, query):
    """Hybrid search (simplified RRF) benchmark."""
    lex = bench_lexical(db, query)
    dense = bench_dense(db, query)

    scores = {}
    for rank, row in enumerate(lex, 1):
        mid = row[0]
        scores[mid] = scores.get(mid, 0) + 1.0 / (60 + rank)
    for rank, row in enumerate(dense, 1):
        mid = row[0]
        scores[mid] = scores.get(mid, 0) + 1.0 / (60 + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:5]


def run_benchmark():
    db = setup_bench_db()

    configs = {
        "Lexical (BM25)": lambda q: bench_lexical(db, q),
        "Dense (Embeddings)": lambda q: bench_dense(db, q),
        "Hybrid (RRF)": lambda q: bench_hybrid(db, q),
    }

    print(f"Running {NUM_QUERIES} queries per configuration...\n")

    print(f"{'Config':<25} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | {'Status':>10}")
    print("-" * 75)

    random.seed(123)
    for name, search_fn in configs.items():
        latencies = []
        for _ in range(NUM_QUERIES):
            q = random_query()
            start = time.perf_counter()
            search_fn(q)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        status = "✓ PASS" if p95 < 200 else "✗ FAIL"

        print(f"{name:<25} | {p50:>9.1f}ms | {p95:>9.1f}ms | {p99:>9.1f}ms | {status:>10}")

    print("-" * 75)
    print(f"\nTarget: p95 < 200ms at {NUM_MEMORIES:,} memories")

    db.close()

    # Cleanup temp database
    if os.path.exists(BENCH_DB):
        os.remove(BENCH_DB)
    print("Benchmark complete. Temp database cleaned up.")


if __name__ == "__main__":
    run_benchmark()
