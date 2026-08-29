import sqlite3
import sqlite_vec
import struct
import os
from typing import List

# BYPASS: Force Transformers to ignore TensorFlow so it doesn't crash from lack of disk space
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from sentence_transformers import SentenceTransformer

def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

def sync_indexes():
    db = sqlite3.connect("data/memory.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    
    cursor = db.cursor()
    
    print("Loading local embedding model (all-MiniLM-L6-v2) using PyTorch bypass...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Restore the original float[384] table
    cursor.execute("DROP TABLE IF EXISTS memories_vec")
    cursor.execute("""
        CREATE VIRTUAL TABLE memories_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding float[384]
        )
    """)
    
    cursor.execute("SELECT memory_id, fact, entity, attribute FROM memories WHERE status = 'CURRENT'")
    memories = cursor.fetchall()
    
    print(f"Found {len(memories)} active memories. Syncing indexes...")
    cursor.execute("DELETE FROM memories_fts")
    
    for count, (mem_id, fact, entity, attribute) in enumerate(memories, 1):
        cursor.execute("INSERT INTO memories_fts (memory_id, entity, attribute, fact) VALUES (?, ?, ?, ?)", (mem_id, entity, attribute, fact))
        embedding = model.encode(fact).tolist()
        cursor.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)", (mem_id, serialize_f32(embedding)))
        
        if count % 10 == 0:
            print(f"Synced {count}/{len(memories)} memories...")
            
    db.commit()
    print("Index synchronization complete!")

if __name__ == "__main__":
    sync_indexes()
