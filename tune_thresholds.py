import json
import sqlite3
from memory_store import MemoryPolicyEngine

def tune_thresholds():
    db = sqlite3.connect(':memory:')
    engine = MemoryPolicyEngine(db)
    
    with open('data/ground_truth.jsonl', 'r') as f:
        events = [json.loads(line) for line in f]
        
    print(f"Starting Threshold Tuning on {len(events)} events...\n")
    print("Step 1: Gathering LLM extractions (this will take a few minutes)...")
    
    results_cache = []
    
    for idx, event in enumerate(events):
        text = event["text"]
        expected_imp = str(event.get("expected_is_important", False)).lower() == 'true'
        
        is_imp_raw = False
        conf = 0.0
        
        if not engine.heuristic_is_noise(text):
            extracted = engine.extract_fact_with_llm(text)
            if extracted:
                is_imp_raw = str(extracted.get("is_important", False)).lower() == 'true'
                conf = float(extracted.get("importance_score", 0.0))
                
        results_cache.append({
            "expected": expected_imp,
            "llm_says_important": is_imp_raw,
            "confidence": conf
        })
        
        print(f"Processed {idx+1}/{len(events)}")
        
    print("\nStep 2: Testing different thresholds instantaneously...\n")
    
    thresholds_to_test = [0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    best_f1 = 0
    best_threshold = 0
    
    print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Notes'}")
    print("-" * 65)
    
    for t in thresholds_to_test:
        tp = 0
        fp = 0
        fn = 0
        
        for res in results_cache:
            # Apply threshold logic
            final_is_imp = False
            if res["llm_says_important"] and res["confidence"] >= t:
                final_is_imp = True
                
            expected = res["expected"]
            
            if expected and final_is_imp:
                tp += 1
            elif expected and not final_is_imp:
                fn += 1
            elif not expected and final_is_imp:
                fp += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = t
            
        print(f"{t:<12.2f} | {precision*100:<9.1f}% | {recall*100:<9.1f}% | {f1_score*100:<9.1f}% | ")
        
    print("-" * 65)
    print(f"💡 RECOMMENDATION: Set LLM_CONFIDENCE_THRESHOLD = {best_threshold:.2f} for the best balance (Highest F1 Score).")

if __name__ == "__main__":
    tune_thresholds()
