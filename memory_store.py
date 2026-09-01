import os

# BYPASS: Force Transformers to ignore TensorFlow BEFORE importing it
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sqlite3
import sqlite_vec
import json
import time
import struct
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

DB_PATH = "data/memory.db"
model = SentenceTransformer('all-MiniLM-L6-v2')

def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT,
            timestamp TEXT,
            type TEXT,
            payload JSON,
            source_app TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT,
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
        CREATE TABLE IF NOT EXISTS memory_sources (
            memory_id TEXT,
            event_id TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(memory_id),
            FOREIGN KEY(event_id) REFERENCES events(event_id),
            PRIMARY KEY(memory_id, event_id)
        )
    """)
    
    # Keep Lexical index for retrieval
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED, 
            entity,
            attribute,
            fact
        )
    """)
    
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding float[384]
        )
    """)
    db.commit()
    return db

class MemoryPolicyEngine:
    MIN_TEXT_LENGTH = 10
    LLM_CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.cursor = db.cursor()
        self.fallback_models = [
            {'id': 'gemini-3.1-flash-lite', 'rpm': 14},
            {'id': 'gemini-3.1-flash-lite', 'rpm': 14}
        ]
        self.current_model_index = 0
        
    def heuristic_is_noise(self, event_text: str) -> bool:
        return len(event_text.strip()) < self.MIN_TEXT_LENGTH

    def extract_fact_with_llm(self, text: str) -> Dict:
        """Extract a single fact from event text using the LLM.
        Used by evaluate.py and tune_thresholds.py for offline evaluation."""
        prompt = f"""You are an AI memory extraction engine for a workplace platform. Analyze this single event.

IMPORTANT — extract these (is_important = true):
- People and roles: "Sharath is the CTO"
- Technology decisions: "Building the backend in Python"
- Company facts: incorporation dates, registration numbers, names
- Concrete state changes: blockers, priority bumps, environment outages
- Documentation content updates: sections added/updated, roadmap drafts published
- Temporary tech states: database switches, migrations (set expires_at)

NOT IMPORTANT — ignore these (is_important = false):
- Social chat, process requests, merge notifications, reminders, opinions, intent.

SCORING RUBRIC (importance_score):
0.8 - 1.0 : Core facts — roles, technology, company info, permanent decisions
0.5 - 0.7 : State changes — blockers, errors, doc updates, roadmap, priority changes
0.0 - 0.4 : Noise — social, requests, notifications, opinions, reminders

Respond ONLY with a single JSON object:
{{"is_important": bool, "entity": str, "attribute": str, "value": str, "expires_at": str|null, "importance_score": float}}

Event Text: {text}
"""
        models = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite']
        for model_name in models:
            retries = 0
            while retries < 3:
                try:
                    time.sleep(4.2)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={'response_mime_type': 'application/json', 'temperature': 0.0}
                    )
                    return json.loads(response.text)
                except Exception as e:
                    err = str(e)
                    if '429' in err:
                        print(f"[{model_name}] Quota exceeded. Falling back...")
                        break
                    print(f"API Error with {model_name}: {err[:50]}... Retrying in 10s...")
                    time.sleep(10)
                    retries += 1
        return {"is_important": False, "entity": "", "attribute": "", "value": "", "expires_at": None, "importance_score": 0.0}
        
    def process_batch(self, batch: List[Dict]):
        if not batch: return
        
        prompt = f"""You are an AI memory extraction engine for a workplace platform. Analyze this batch of events.
        
IMPORTANT — extract these (is_important = true):
- People and roles: "Sharath is the CTO"
- Technology decisions: "Building the backend in Python"
- Company facts: incorporation dates, registration numbers, names
- Concrete state changes: blockers, priority bumps, environment outages
- Documentation content updates: sections added/updated, roadmap drafts published
- Temporary tech states: database switches, migrations (set expires_at)

NOT IMPORTANT — ignore these (is_important = false):
- Social chat, process requests, merge notifications, reminders, opinions, intent.

SCORING RUBRIC (importance_score):
0.8 - 1.0 : Core facts — roles, technology, company info, permanent decisions
0.5 - 0.7 : State changes — blockers, errors, doc updates, roadmap, priority changes
0.0 - 0.4 : Noise — social, requests, notifications, opinions, reminders

Respond ONLY with a JSON array of objects. You MUST return exactly {len(batch)} objects, in the exact same order as the input.
Format for each object:
{{"event_id": str, "is_important": bool, "entity": str, "attribute": str, "value": str, "expires_at": str|null, "importance_score": float}}

Input Events:
"""
        for i, ev in enumerate(batch):
            prompt += f"\n[{i+1}] Event ID: {ev['event_id']}\nText: {ev['text']}\n"
            
        models = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite']
        extracted_results = []
        
        for model_name in models:
            retries = 0
            while retries < 3:
                try:
                    time.sleep(4.2)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={'response_mime_type': 'application/json', 'temperature': 0.0}
                    )
                    extracted_results = json.loads(response.text)
                    if len(extracted_results) == len(batch):
                        break
                    else:
                        print(f"Warning: Expected {len(batch)} results, got {len(extracted_results)}. Retrying...")
                        retries += 1
                except Exception as e:
                    err = str(e)
                    if '429' in err:
                        print(f"[{model_name}] Quota exceeded. Falling back to next model...")
                        break
                    print(f"API Error with {model_name}: {err[:50]}... Retrying in 10s...")
                    time.sleep(10)
                    retries += 1
            if len(extracted_results) == len(batch):
                break
                
        if len(extracted_results) != len(batch):
            print("Failed to process batch correctly. Skipping.")
            return

        for ev, extracted in zip(batch, extracted_results):
            if not extracted or not extracted.get("is_important"): 
                continue
                
            importance = float(extracted.get("importance_score", 1.0))
            if importance < self.LLM_CONFIDENCE_THRESHOLD: 
                continue
                
            self._insert_memory_logic(ev["event_id"], ev["timestamp"], extracted, ev["user_id"])

    def resolve_semantic_collision(self, new_fact: str, old_fact: str) -> str:
        """Returns DUPLICATE, SUPERSEDE, or NEW using LLM."""
        prompt = f"""You are an AI resolving a memory collision.
Old Memory: {old_fact}
New Memory: {new_fact}

Compare them:
1. If the new memory is essentially the same fact as the old one (e.g., restated), return DUPLICATE.
2. If the new memory is an update/change to the same subject (e.g. state changed, technology changed), return SUPERSEDE.
3. If they are about completely different subjects despite semantic similarity, return NEW.

Return ONLY one word: DUPLICATE, SUPERSEDE, or NEW.
"""
        models = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite']
        for model_name in models:
            retries = 0
            while retries < 3:
                try:
                    time.sleep(4.2)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={'temperature': 0.0}
                    )
                    action = response.text.strip().upper()
                    if action in ["DUPLICATE", "SUPERSEDE", "NEW"]:
                        return action
                    return "NEW"
                except Exception as e:
                    err = str(e)
                    if '429' in err:
                        print(f"[{model_name}] Quota exceeded in collision check. Falling back...")
                        break
                    print(f"API Error in collision check: {err[:50]}... Retrying in 5s...")
                    time.sleep(5)
                    retries += 1
        return "NEW"

    def _insert_memory_logic(self, event_id: str, timestamp: str, extracted: Dict, user_id: str):
            
        entity = extracted.get("entity")
        attribute = extracted.get("attribute")
        new_value = extracted.get("value")
        expires_at = extracted.get("expires_at")
        fact_text = f"{entity}'s {attribute} is {new_value}"
        
        canon_entity = str(entity).strip().lower() if entity else ""
        canon_attr = str(attribute).strip().lower() if attribute else ""

        
        # 1. Generate Vector
        vector = model.encode(fact_text).tolist()
        vec_bytes = serialize_f32(vector)
        
        # 2. Semantic Search for Collisions (Distance < 0.35)
        self.cursor.execute("""
            SELECT m.memory_id, m.fact, v.distance 
            FROM memories_vec v
            JOIN memories m ON v.memory_id = m.memory_id
            WHERE v.embedding MATCH ? AND v.k = 3 AND m.status = 'CURRENT'
        """, (vec_bytes,))
        similar_memories = self.cursor.fetchall()
        
        candidates = {}
        exact_match_id = None
        
        for old_id, old_fact, dist in similar_memories:
            if dist < 0.35: 
                candidates[old_id] = old_fact
                
        # Also grab exact attribute matches (canonicalized)
        self.cursor.execute("SELECT memory_id, fact, entity, attribute FROM memories WHERE status='CURRENT'")
        for old_id, old_fact, old_ent, old_attr in self.cursor.fetchall():
            o_ent = str(old_ent).strip().lower() if old_ent else ""
            o_attr = str(old_attr).strip().lower() if old_attr else ""
            if o_ent == canon_entity and o_attr == canon_attr and o_ent != "":
                exact_match_id = old_id
            else:
                candidates[old_id] = old_fact
            
        new_memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        # Deterministic Resolution first
        if exact_match_id:
            resolved_action = "SUPERSEDE"
            target_old_id = exact_match_id
        else:
            resolved_action = "NEW"
            target_old_id = None
            for old_id, old_fact in candidates.items():
                action = self.resolve_semantic_collision(fact_text, old_fact)
                if action in ["DUPLICATE", "SUPERSEDE"]:
                    resolved_action = action
                    target_old_id = old_id
                    break
                
        if resolved_action == "NEW":
            print(f"[{event_id}] -> NEW MEMORY: {fact_text}")
            self.cursor.execute("""
                INSERT INTO memories (memory_id, user_id, fact, entity, attribute, value, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (new_memory_id, user_id, fact_text, entity, attribute, new_value, timestamp, expires_at))
            self.cursor.execute("INSERT INTO memory_sources (memory_id, event_id) VALUES (?, ?)", (new_memory_id, event_id))
            self.cursor.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)", (new_memory_id, vec_bytes))
            self.cursor.execute("INSERT INTO memories_fts (memory_id, entity, attribute, fact) VALUES (?, ?, ?, ?)", (new_memory_id, entity, attribute, fact_text))
            
        elif resolved_action == "DUPLICATE":
            print(f"[{event_id}] -> DUPLICATE (Merged into {target_old_id})")
            self.cursor.execute("INSERT OR IGNORE INTO memory_sources (memory_id, event_id) VALUES (?, ?)", (target_old_id, event_id))
            
        elif resolved_action == "SUPERSEDE":
            print(f"[{event_id}] -> SUPERSEDES {target_old_id} with: {fact_text}")
            self.cursor.execute("UPDATE memories SET status = 'SUPERSEDED', superseded_by = ? WHERE memory_id = ?", (new_memory_id, target_old_id))
            self.cursor.execute("""
                INSERT INTO memories (memory_id, user_id, fact, entity, attribute, value, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (new_memory_id, user_id, fact_text, entity, attribute, new_value, timestamp, expires_at))
            self.cursor.execute("INSERT INTO memory_sources (memory_id, event_id) VALUES (?, ?)", (new_memory_id, event_id))
            self.cursor.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)", (new_memory_id, vec_bytes))
            self.cursor.execute("INSERT INTO memories_fts (memory_id, entity, attribute, fact) VALUES (?, ?, ?, ?)", (new_memory_id, entity, attribute, fact_text))
            
        self.db.commit()

def ingest_events(db: sqlite3.Connection, events_file: str):
    cursor = db.cursor()
    engine = MemoryPolicyEngine(db)
    
    print("Starting Semantic Memory Extraction (Batched)...")
    batch = []
    total_processed = 0
    
    with open(events_file, 'r') as f:
        for line in f:
            event = json.loads(line)
            event_id = event["event_id"]
            
            cursor.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,))
            if cursor.fetchone(): continue
                
            cursor.execute("""
                INSERT INTO events (event_id, user_id, timestamp, type, payload, source_app)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_id, event["user_id"], event["timestamp"], event["type"], json.dumps(event["payload"]), event["source_app"]))
            db.commit()
            
            payload_text = event["payload"].get("text", "")
            if payload_text and not engine.heuristic_is_noise(payload_text):
                batch.append({
                    "event_id": event_id,
                    "text": payload_text,
                    "timestamp": event["timestamp"],
                    "user_id": event["user_id"]
                })
                
            total_processed += 1
            if len(batch) >= 15:
                print(f"Processing batch of 15 events (Total read: {total_processed})...")
                engine.process_batch(batch)
                batch = []
                
    if batch:
        print(f"Processing final batch of {len(batch)} events...")
        engine.process_batch(batch)
                
if __name__ == "__main__":
    print("Initializing Memory Store v2...")
    
    # We delete the old database automatically in Python to start fresh!
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Cleared old database. Starting fresh extraction...")
        
    db = init_db()
    ingest_events(db, "data/events.jsonl")
    print("Database schema created and populated successfully with Semantic Resolution!")
