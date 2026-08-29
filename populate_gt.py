import json

mappings = {
    1: {"is_important": True, "entity": "company", "attribute": "focus", "value": "memory systems for AI products"},
    7: {"is_important": True, "entity": "infrastructure migration", "attribute": "scheduled_for", "value": "April"},
    9: {"is_important": True, "entity": "company/project", "attribute": "project", "value": "building an AI that remembers things"},
    10: {"is_important": True, "entity": "Sharath", "attribute": "role", "value": "CTO"},
    13: {"is_important": True, "entity": "migration", "attribute": "database", "value": "MongoDB"},
    14: {"is_important": True, "entity": "startup", "attribute": "focus", "value": "giving memory to AI agents"},
    25: {"is_important": True, "entity": "infrastructure migration", "attribute": "scheduled_for", "value": "April"},
    26: {"is_important": True, "entity": "Sohil", "attribute": "launch_decision_owner", "value": "launch decision"},
    28: {"is_important": True, "entity": "company", "attribute": "incorporation_date", "value": "Jan 1, 2026"},
    31: {"is_important": True, "entity": "company", "attribute": "incorporation_date", "value": "Jan 1, 2026"},
    33: {"is_important": True, "entity": "Sohil", "attribute": "budget_approval", "value": "Q2"},
    38: {"is_important": True, "entity": "backend", "attribute": "programming_language", "value": "Python"},
    43: {"is_important": True, "entity": "project", "attribute": "blocker", "value": "design assets"},
    45: {"is_important": True, "entity": "Sohil", "attribute": "budget_approval", "value": "Q2"},
    54: {"is_important": True, "entity": "company", "attribute": "incorporation_date", "value": "Jan 1, 2026"},
    59: {"is_important": True, "entity": "Gandhi", "attribute": "role", "value": "Lead Engineer"}
}

def populate():
    with open('data/ground_truth.jsonl', 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if i in mappings:
            m = mappings[i]
            record["expected_is_important"] = m["is_important"]
            record["expected_entity"] = m["entity"]
            record["expected_attribute"] = m["attribute"]
            record["expected_value"] = m["value"]
        else:
            record["expected_is_important"] = False
            record["expected_entity"] = None
            record["expected_attribute"] = None
            record["expected_value"] = None
            
        out_lines.append(json.dumps(record))
        
    with open('data/ground_truth.jsonl', 'w') as f:
        for out in out_lines:
            f.write(out + "\n")
            
    print(f"Populated {len(out_lines)} events with ground truth logic!")

if __name__ == "__main__":
    populate()
