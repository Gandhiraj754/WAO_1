# WAO-Recall: Enterprise AI Memory Engine

**WAO-Recall** is a fully local, production-grade memory layer for AI agents. It ingests a stream of workplace events (Slack messages, emails, task updates, doc edits), extracts durable facts, manages supersession and deduplication, and answers natural language questions with cited evidence.

Built entirely from scratch — no LangChain, LlamaIndex, or RAG frameworks. Fully local vector indexing via SQLite.

---

## Setup (< 5 minutes)

### Prerequisites
- Python 3.10+
- A free [Google AI Studio](https://aistudio.google.com/) API key for Gemini

### Install
```bash
git clone <repo-url>
cd wao-recall
pip install -r requirements.txt   # or: pip install google-genai python-dotenv sentence-transformers sqlite-vec numpy
```

### Configure
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run the Full Pipeline
```bash
# 1. Generate synthetic events (deterministic, seed=42)
python generate_events.py          # → data/events.jsonl (609 events)

# 2. Ingest events into the memory database
python memory_store.py             # → data/memory.db (takes ~40 min, rate-limited)

# 3. Sync search indexes
python sync_indexes.py             # → Rebuilds FTS5 + vector indexes

# 4. Run the evaluation
python eval_retrieval.py           # → Recall@5, MRR, p95 latency (0 API calls)
python bench.py                    # → p95 latency at 10k memories (0 API calls)

# 5. Interactive Q&A
python retrieval.py                # → Ask questions, get cited answers
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAO-Recall Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Events Stream (data/events.jsonl)                             │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │  Memory Policy Engine (memory_store.py) │                   │
│  │                                         │                   │
│  │  Stage 1: Heuristic Filter (0ms, $0)    │                   │
│  │  Stage 2: LLM Extraction + Rubric       │                   │
│  │  Stage 3: Semantic Collision Detection   │                   │
│  │           → DUPLICATE / SUPERSEDE / NEW  │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         SQLite (data/memory.db)         │                   │
│  │                                         │                   │
│  │  memories     — facts, entity/attr/val  │                   │
│  │  memories_fts — FTS5 BM25 index         │                   │
│  │  memories_vec — sqlite-vec embeddings   │                   │
│  │  events       — raw event log           │                   │
│  │  memory_sources — provenance links      │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │    Retrieval Engine (retrieval.py)       │                   │
│  │                                         │                   │
│  │  Lexical Search  → BM25 via FTS5        │                   │
│  │  Dense Search    → Cosine via sqlite-vec│                   │
│  │  Hybrid Search   → RRF + Recency Decay  │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │    Agent Surface (POST /ask)            │                   │
│  │    Answer from memory only. No guessing.│                   │
│  │    Every claim cites a [memory_id].     │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation

### Run the eval (deterministic, offline)
```bash
# Retrieval metrics — 0 API calls, runs from local DB
python eval_retrieval.py

# Answer quality — generates + caches LLM answers (uses API once)
python eval_retrieval.py --generate-answers
python eval_retrieval.py --full

# Extraction quality — debug harness for the LLM extraction step
python evaluate.py                 # Uses cached extraction results

# Latency benchmark — proves p95 < 200ms at 10k memories
python bench.py
```

All LLM responses are cached in `data/extraction_cache.json` and `data/answer_cache.json`. Running the eval on a clean machine will produce identical numbers without an API key.

### Test Query Coverage (42 queries)
| Category | Count | Description |
|---|---|---|
| Supersession | 10 | Facts that changed over time (e.g., backend language) |
| Must-Return-Nothing | 8 | Questions about things not in memory |
| Multi-Hop | 8 | Answers requiring 2+ memories combined |
| Temporal | 7 | Questions about time-dependent state |
| Factual | 9 | Direct fact lookup |

See `EVAL.md` for the full ablation table and results.

---

## Key Files

| File | Purpose |
|---|---|
| `generate_events.py` | Synthetic event generator (Markov Chain, seed=42) |
| `memory_store.py` | Memory Policy Engine (extraction, dedup, supersession) |
| `retrieval.py` | Retrieval Engine (BM25, dense, hybrid RRF) + interactive Q&A |
| `eval_retrieval.py` | **Graded evaluation harness** (Recall@5, MRR, latency) |
| `evaluate.py` | Extraction quality debugger (Precision, Recall, Fact Accuracy) |
| `bench.py` | Latency benchmark at 10k memories |
| `DECISIONS.md` | 10 design decisions with sources |
| `EVAL.md` | Evaluation results and ablation table |
| `LIMITS.md` | 5 scale limitations and fixes |

---

## Constraints Met

| Constraint | Status |
|---|---|
| Python | ✓ |
| No frameworks (LangChain, etc.) | ✓ |
| Budget ₹0 (free tier only) | ✓ Gemini free tier + local SentenceTransformer |
| p95 latency < 200ms at 10k | ✓ Proven by bench.py |
| Deterministic under fixed seed | ✓ generate_events.py uses seed=42 |
| Cached LLM responses | ✓ extraction_cache.json + answer_cache.json |
