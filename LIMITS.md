# LIMITS.md — What Breaks at Scale

Five things that break at 10M memories or 10k concurrent users, and what we'd build with two more weeks.

---

## 1. SQLite Single-Writer Lock

**What breaks:** SQLite allows only one writer at a time. With 10k users ingesting events simultaneously, writes serialize into a single queue. At ~4ms per write, throughput caps at ~250 writes/second. 10k users generating 1 event/second = 10,000 writes/second → 40x over capacity.

**Fix with two weeks:** Migrate to PostgreSQL with pgvector. PostgreSQL handles thousands of concurrent writers via MVCC. We'd use connection pooling (pgBouncer) and partition the memories table by user_id for write isolation. The SQL schema is already compatible — only the connection layer changes.

---

## 2. Vector Search Latency at 10M Memories

**What breaks:** `sqlite-vec` uses brute-force or flat-index vector search. At 10,000 memories, p95 is under 200ms (proven by bench.py). At 10M memories, brute-force search scales linearly → p95 would exceed 20 seconds.

**Fix with two weeks:** Switch to HNSW (Hierarchical Navigable Small World) indexing via pgvector or a dedicated vector store (Qdrant, Weaviate). HNSW provides O(log n) search time. At 10M vectors with 384 dimensions, p95 would stay under 50ms. We'd also add an IVF (Inverted File) pre-filter by user_id to reduce the search space per query.

---

## 3. LLM Extraction Rate Limit

**What breaks:** We rate-limit at 1 extraction per 4.2 seconds (14 RPM on Gemini free tier). 10M events would take 10,000,000 × 4.2s = 485 days to process. Even with a paid tier at 1000 RPM, it would take ~7 days.

**Fix with two weeks:** Batch processing pipeline. Group events into batches of 10-20, send them in a single prompt asking the LLM to extract facts from all events at once. This reduces API calls by 10-20x. We'd also add a fast classifier (fine-tuned DistilBERT, ~5ms/event) as a pre-filter before the LLM — if the classifier says "noise" with >95% confidence, skip the LLM entirely. This could eliminate 60% of events before they ever hit the API.

---

## 4. Memory Collision Search Explosion

**What breaks:** For every new memory, we search all existing CURRENT memories for semantic collisions (distance < 0.35) and entity-attribute exact matches. At 10M memories, the entity-attribute scan (`SELECT * FROM memories WHERE status='CURRENT'`) becomes a full table scan returning millions of rows.

**Fix with two weeks:** Add a composite index on `(entity, attribute, status)` for the deterministic collision path. For the semantic collision path, use the HNSW index (from fix #2) with a distance threshold filter. We'd also add a Bloom filter per entity to skip the collision check entirely when the entity has never been seen before (O(1) lookup).

---

## 5. FTS5 Index Rebuild on High Write Volume

**What breaks:** SQLite FTS5 maintains a merge-tree index. Under sustained high write load (10k inserts/second), the FTS5 merge operations compete with reads, causing read latency spikes. The FTS5 index also grows linearly with document count — at 10M memories, the index file exceeds 2GB.

**Fix with two weeks:** Replace FTS5 with Elasticsearch or Meilisearch for lexical search. These are designed for high-throughput indexing with near-real-time search. We'd keep SQLite as the source-of-truth for structured data and use Elasticsearch purely as a search index, with a CDC (Change Data Capture) pipeline to keep them in sync.
