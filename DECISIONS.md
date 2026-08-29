# DECISIONS.md — Design Decisions Log

Every non-trivial design choice is recorded here with what was decided, what was rejected and why, the source learned from, and the tradeoff accepted.

---

## 1. SQLite as the Unified Storage Layer

**Decision:** Use SQLite with `sqlite-vec` and FTS5 extensions as the single storage engine for events, memories, vectors, and full-text search.

**Rejected:** PostgreSQL + pgvector, FAISS + separate file storage.

**Why:** SQLite is zero-config, runs on any machine, and eliminates Docker/network complexity. With `sqlite-vec` (C extension by Alex Garcia), we get production-grade ANN vector search directly in SQL. FTS5 gives us BM25 lexical search. One database file holds everything.

**Source:** [sqlite-vec documentation](https://alexgarcia.xyz/sqlite-vec/) and [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)

**Tradeoff:** SQLite's write concurrency is limited to one writer at a time. At enterprise scale (10k+ concurrent users), we'd need to migrate to PostgreSQL + pgvector. For this trial's scale (3 users, ~600 events), SQLite is the right tool.

---

## 2. BM25F (BM25 for Fields) instead of Standard BM25

**Decision:** Implement BM25F by separating `entity`, `attribute`, and `fact` into distinct columns in the FTS5 virtual table, and weighting them `(5.0, 3.0, 1.0)` during retrieval.

**Rejected:** Standard BM25 on a single concatenated text block.

**Why:** Workplace queries are highly targeted (e.g., "What is the backend language?"). Standard BM25 treats the word "backend" the same regardless of where it appears. By extracting structured data (entity="backend", attribute="language") during ingestion and indexing them as separate fields, we can apply massive score multipliers when the user's query exactly hits an entity or attribute. This solves the "long document penalty" flaw of standard BM25 and drastically improves Lexical precision.

**Source:** [The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, 2009)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)

**Tradeoff:** Increased storage overhead (duplicating entity/attribute strings in the index) and slightly slower ingestion time, which is acceptable for the massive leap in retrieval precision.

---

## 3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

**Decision:** Implement three retrieval configurations (Lexical BM25, Dense Embeddings, Hybrid RRF+Recency) and compare them.

**Rejected:** Using only dense search, using only BM25, using a learned reranker.

**Why:** BM25 excels at exact keyword matches ("registration number 99887766") while dense search captures semantic similarity ("core product" ↔ "AI memory layer"). RRF fuses both without requiring training data. The formula `1/(60+rank)` is parameter-free and robust.

**Source:** [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormack_rrf_sigir09.pdf) and [Elasticsearch RRF docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)

**Tradeoff:** RRF is a simple score fusion. A learned cross-encoder reranker (e.g., ms-marco-MiniLM) would likely improve MRR by 5-10%, but adds inference latency and model complexity.

---

## 4. all-MiniLM-L6-v2 as the Embedding Model

**Decision:** Use `all-MiniLM-L6-v2` (384-dim) for all local dense embeddings.

**Rejected:** OpenAI `text-embedding-3-small`, Cohere embeddings, `all-mpnet-base-v2`.

**Why:** Budget constraint is ₹0 / no paid APIs. MiniLM-L6-v2 runs locally on CPU, produces 384-dim vectors (compact for sqlite-vec), and ranks in the top 10 on MTEB benchmarks for its size class. It encodes in ~5ms per sentence on CPU.

**Source:** [SBERT Model Card](https://www.sbert.net/docs/pretrained_models.html) and [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

**Tradeoff:** 384-dim vectors sacrifice ~3% recall vs. 768-dim models (mpnet-base-v2) but halve storage and improve query latency.

---

## 5. LLM-Based Extraction with Scoring Rubric

**Decision:** Use the LLM (Gemini Flash Lite) with an explicit scoring rubric and few-shot examples for memory extraction.

**Rejected:** Rule-based extraction (regex/keyword matching), zero-shot prompting without rubric.

**Why:** Workplace events are ambiguous. "Blocked on design assets" is important (state change), but "Can someone review my PR?" is not (process request). Rules can't distinguish these reliably. The scoring rubric (0.8-1.0 for core facts, 0.5-0.7 for state changes, 0.0-0.4 for noise) gives the LLM explicit calibration guidance, producing consistent importance scores.

**Source:** [Gemini API structured output docs](https://ai.google.dev/gemini-api/docs/structured-output) and prompt engineering best practices from [OpenAI Cookbook](https://cookbook.openai.com/)

**Tradeoff:** LLM extraction adds ~4 seconds per event (rate-limited). For the 609-event dataset this means ~40 minutes of ingestion time. At production scale, we'd batch events and use async processing.

---

## 6. Semantic Collision Detection for Dedup/Supersession

**Decision:** Before storing a new memory, search sqlite-vec for similar existing memories (distance < 0.35). If found, ask the LLM to classify as DUPLICATE, SUPERSEDE, or NEW.

**Rejected:** Pure entity-attribute key matching, no dedup at all, embedding-only dedup without LLM verification.

**Why:** Embedding distance alone can't reliably distinguish "Backend is Python" from "Backend is Go" (both are about "backend technology" with similar vectors). The two-stage approach (fast vector search to find candidates, then LLM to resolve) gives us precision. Entity-attribute exact matching handles deterministic cases without burning an API call.

**Source:** [Approximate Nearest Neighbor search in practice](https://arxiv.org/abs/1702.08734) and the WAO-Recall trial specification (Section 02B).

**Tradeoff:** Each collision check costs one LLM call. In the worst case (every new memory collides), ingestion time doubles. We mitigate this with the entity-attribute deterministic shortcut.

---

## 7. Deterministic Event Generation with Markov Chains

**Decision:** Use a Markov Chain transition matrix to generate realistic event type sequences, seeded with `random.seed(42)` for reproducibility.

**Rejected:** Uniform random event types, hand-written event scripts, real anonymized data.

**Why:** Uniform random produces unrealistic sequences (email → calendar → email → calendar). The Markov chain models realistic workflows: after a chat_message, the next event is 50% chat, 20% doc_edit, 20% task_update, 10% email. Fixed seed ensures `data/events.jsonl` is identical across runs.

**Source:** [Markov Chains for sequence generation](https://en.wikipedia.org/wiki/Markov_chain) and the WAO-Recall spec requirement: "deterministic under a fixed seed."

**Tradeoff:** A Markov chain is memoryless — it doesn't model multi-turn conversations or time-of-day patterns. For a more realistic simulation, we'd use a Hidden Markov Model or a simple LLM-based generator.

---

## 8. Recency Decay in Hybrid Search

**Decision:** Apply a subtle time-decay multiplier to RRF scores: `decay = 1 / (1 + age_in_days * 0.01)`.

**Rejected:** No recency bias, hard recency cutoff, exponential decay.

**Why:** When a user asks "What database are we using?", the most recent memory should rank higher than an older one about the same topic. A 0.01/day decay rate means a 30-day-old memory loses only ~23% of its score — subtle enough to not bury old but important facts.

**Source:** [Time-aware information retrieval](https://dl.acm.org/doi/10.1145/1571941.1572085) and temporal IR research.

**Tradeoff:** The decay constant (0.01) is hand-tuned. In production, we'd learn this from user click-through data.

---

## 9. Cached, Offline Evaluation Harness

**Decision:** Cache all LLM responses (extraction cache, answer cache) as JSON files committed to the repo. Evaluation runs deterministically from cache.

**Rejected:** Live evaluation that calls the LLM each time, mock LLM responses.

**Why:** The WAO-Recall spec explicitly requires: "Cache and commit LLM responses so our numbers match yours exactly." By caching extraction results in `data/extraction_cache.json` and answer results in `data/answer_cache.json`, anyone can run `python eval_retrieval.py` and get identical numbers without an API key.

**Source:** WAO-Recall Trial Brief, Section 02E.

**Tradeoff:** If the prompt changes, the cache must be regenerated. We document this clearly in the README.

---

## 10. The Ambiguity Resolution — "Design decision buried in the spec"

**Decision:** The spec says events must include `source_app` and lists five event types. It also says "a decision made in email and referenced later in task_update." We interpreted this as requiring cross-source provenance tracking — the memory system must link facts across different source apps, not just within a single app.

**Rejected:** Treating each source_app as an independent silo.

**Why:** The spec deliberately buries this to test whether candidates read carefully. Our `memory_sources` table tracks multiple event_ids per memory, and our semantic collision detection works across all event types regardless of source_app.

**Source:** WAO-Recall Trial Brief, Section 02A: "READ THIS SECTION TWICE — There is a design decision buried in it."

**Tradeoff:** Cross-source linking increases collision search space. We accept this cost because it's architecturally correct — an enterprise memory system must unify facts across Slack, email, and task trackers.
