import os

# BYPASS: Force Transformers to ignore TensorFlow BEFORE importing it
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import sqlite3
import re
import argparse
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from memory_store import MemoryPolicyEngine

CACHE_PATH = "data/extraction_cache.json"
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'is', 'of'}]
    return ' '.join(words).strip()

def is_exact_match(expected: str, extracted: str) -> bool:
    if expected == extracted: return True
    norm_exp = normalize_text(expected)
    norm_ext = normalize_text(extracted)
    if not norm_exp and not norm_ext: return True
    if not norm_exp or not norm_ext: return False
    return norm_exp == norm_ext or norm_exp in norm_ext or norm_ext in norm_exp

def semantic_match(expected, extracted):
    exp_str = f"{expected[0]} {expected[1]} {expected[2]}".lower()
    ext_str = f"{extracted[0]} {extracted[1]} {extracted[2]}".lower()
    
    vec1 = embed_model.encode(exp_str)
    vec2 = embed_model.encode(ext_str)
    
    cosine_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cosine_sim >= 0.70

def generate_cache():
    db = sqlite3.connect(':memory:')
    engine = MemoryPolicyEngine(db)
    
    with open('data/ground_truth.jsonl', 'r') as f:
        events = [json.loads(line) for line in f]
        
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            cache = json.load(f)
            
    print(f"Generating cache for {len(events)} events. This uses the LLM API...")
    import time
    for event in events:
        event_id = event["event_id"]
        text = event["text"]
        if event_id in cache:
            continue
            
        print(f"Extracting {event_id}...")
        time.sleep(4.2)
        if engine.heuristic_is_noise(text):
            cache[event_id] = {"heuristic_drop": True}
        else:
            extracted = engine.extract_fact_with_llm(text)
            cache[event_id] = {"heuristic_drop": False, "extracted": extracted}
            
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
            
    print("Cache generation complete.")

def run_evaluation():
    if not os.path.exists(CACHE_PATH):
        print("Cache not found. Run with --generate-cache first.")
        return
        
    threshold = MemoryPolicyEngine.LLM_CONFIDENCE_THRESHOLD
        
    with open(CACHE_PATH, 'r') as f:
        cache = json.load(f)
        
    with open('data/ground_truth.jsonl', 'r') as f:
        events = [json.loads(line) for line in f]
        
    tp, tn, fp, fn = 0, 0, 0, 0
    exact_matches, fuzzy_matches, extraction_failures = 0, 0, 0
    
    for event in events:
        event_id = event['event_id']
        expected_imp = str(event.get('expected_is_important', False)).lower() == 'true'
        exp_entity = str(event.get("expected_entity", "")).strip().lower()
        exp_attr = str(event.get("expected_attribute", "")).strip().lower()
        exp_val = str(event.get("expected_value", "")).strip().lower()
        
        if event_id not in cache:
            continue
            
        c = cache[event_id]
        if c.get("heuristic_drop", False):
            actual_imp = False
            importance_score = 0.0
        else:
            extracted = c.get("extracted", {})
            actual_imp = extracted.get('is_important', False)
            importance_score = float(extracted.get('importance_score', 1.0))
            
            ext_entity = str(extracted.get("entity", "")).strip().lower()
            ext_attr = str(extracted.get("attribute", "")).strip().lower()
            ext_val = str(extracted.get("value", "")).strip().lower()
            
        if actual_imp and importance_score < threshold:
            actual_imp = False
            
        if expected_imp and actual_imp:
            tp += 1
        elif not expected_imp and not actual_imp:
            tn += 1
        elif not expected_imp and actual_imp:
            fp += 1
        elif expected_imp and not actual_imp:
            fn += 1

        if expected_imp and actual_imp:
            if ext_entity or ext_attr or ext_val:
                if is_exact_match(exp_entity, ext_entity) and is_exact_match(exp_attr, ext_attr) and is_exact_match(exp_val, ext_val):
                    exact_matches += 1
                elif semantic_match((exp_entity, exp_attr, exp_val), (ext_entity, ext_attr, ext_val)):
                    fuzzy_matches += 1
                else:
                    extraction_failures += 1
            else:
                extraction_failures += 1

    print("\nEVALUATION RESULTS")
    print("="*60)
    print(f"True Positives  : {tp}")
    print(f"True Negatives  : {tn}")
    print(f"False Positives : {fp} (Hallucinations)")
    print(f"False Negatives : {fn} (Misses)")
    
    print("\n--- PERFORMANCE METRICS ---")
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    print(f"Precision : {precision:.1f}% (When it extracts, how often is it right?)")
    print(f"Recall    : {recall:.1f}% (Out of all important facts, how many did it catch?)")
    
    print("\n--- EXTRACTION ACCURACY ---")
    total_valid = exact_matches + fuzzy_matches + extraction_failures
    if total_valid > 0:
        accuracy = (exact_matches + fuzzy_matches) / total_valid * 100
        print(f"Fact Accuracy : {accuracy:.1f}% (Semantic match on Entity/Attribute/Value)")
    else:
        print("Fact Accuracy : N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-cache", action="store_true", help="Generate offline cache")
    args = parser.parse_args()
    
    if args.generate_cache:
        generate_cache()
    else:
        run_evaluation()
