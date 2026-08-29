# LIMITS.md — What Breaks at Scale

As requested by the specification, here are the five critical bottlenecks that will cause this system to fail at 10M memories or 10,000 concurrent users. 

### Scale Limitations

| Component | What Breaks at 10M Memories / 10k Users |
| :--- | :--- |
| **1. SQLite Single-Writer Lock** | SQLite uses file-level locking. With 10k users, writes serialize into a single queue, capping throughput at ~250 writes/second. The database will quickly throw timeout exceptions. |
| **2. Vector Search Latency** | `sqlite-vec` uses a brute-force flat index. Searching 10M vectors linearly (O(n)) will take 15–20 seconds per query, completely destroying the user experience. |
| **3. API Rate Limiting** | The free-tier LLM throttles at 15 Requests/Min. Processing 10M events synchronously would take 486 days. Even paid tiers will struggle with sequential extraction. |
| **4. Collision Detection Scan** | To deduplicate memories, we run an exact-match SQL scan on all `CURRENT` memories. At 10M rows, this full-table scan degrades database I/O performance. |
| **5. FTS5 Index Rebuilds** | SQLite's FTS5 maintains a merge-tree index. Under heavy write loads (10k inserts/sec), background merges choke the disk, severely spiking read latencies. |

---

### The Two-Week Engineering Fix

If given two more weeks to prepare this architecture for enterprise production, we would fundamentally decouple the read/write paths and migrate to an asynchronous, distributed system:

1. **Storage Migration (PostgreSQL):** We will replace SQLite with PostgreSQL (`pgvector`) to utilize Multiversion Concurrency Control (MVCC) and `pgBouncer`, entirely solving the concurrent writer bottleneck.
2. **O(log n) Search Infrastructure:** We will implement HNSW (Hierarchical Navigable Small World) indexing for dense vectors, and migrate the lexical BM25 index to a dedicated **Elasticsearch** cluster via Debezium CDC. This ensures read queries take <50ms without being blocked by writes.
3. **Event-Driven Ingestion:** We will deploy **Apache Kafka** to queue incoming events. Worker nodes will batch events and use a cheap, local Small Language Model (e.g., DistilBERT) to instantly drop noise without hitting the Gemini API.
4. **O(1) Collision Avoidance:** We will deploy **Redis** and use Bloom Filters scoped by `user_id`. This allows us to definitively know if an entity exists in O(1) time before ever touching the SQL database for a collision check.
