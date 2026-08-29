import json

with open('data/extraction_cache.json') as f:
    cache = json.load(f)

with open('data/ground_truth.jsonl') as f:
    events = [json.loads(line) for line in f]

def is_actually_important(cdata, threshold=0.45):
    if cdata.get('heuristic_drop'): 
        return False
    ext = cdata.get('extracted', {})
    if not ext.get('is_important', False): 
        return False
    if float(ext.get('importance_score', 1.0)) < threshold: 
        return False
    return True

print("=== FALSE NEGATIVES (Important facts we missed) ===")
for e in events:
    if str(e.get('expected_is_important', False)).lower() == 'true':
        eid = e['event_id']
        if eid in cache:
            if not is_actually_important(cache[eid]):
                print(f"TEXT: {e['text']}")
                print(f"EXTRACTED BY LLM: {cache[eid].get('extracted')}")
                print("-" * 50)
