import json
import random
import os

def create_ground_truth_sample():
    with open('data/events.jsonl', 'r') as f:
        events = [json.loads(line) for line in f]
        
    # Pick exactly 60 random events
    random.seed(42) # Seed for reproducibility
    sampled_events = random.sample(events, 60)
    
    # Write them out with empty ground truth fields
    with open('data/ground_truth.jsonl', 'w') as out_f:
        for event in sampled_events:
            payload = event.get('payload', {}).get('text', '')
            
            # All fields are strictly null. NO LLM logic is used here.
            gt_record = {
                "event_id": event["event_id"],
                "text": payload,
                "expected_is_important": None,
                "expected_entity": None,
                "expected_attribute": None,
                "expected_value": None
            }
            out_f.write(json.dumps(gt_record) + "\n")

if __name__ == "__main__":
    create_ground_truth_sample()
    print("Successfully created a blank data/ground_truth.jsonl with 60 events. Ready for external labeling.")
