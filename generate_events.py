import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ==========================================
# 1. Configuration & Determinism
# ==========================================
SEED = 42
random.seed(SEED)

# Using the users provided by you!
USERS = ["u_sohil", "u_sharath", "u_gandhi"]
START_DATE = datetime(2026, 1, 1, 9, 0, 0)
TOTAL_DAYS = 95 

# ==========================================
# 2. Markov Chain Transition Matrix
# ==========================================
TRANSITION_MATRIX = {
    "chat_message": {"chat_message": 0.5, "doc_edit": 0.2, "task_update": 0.2, "email": 0.1},
    "doc_edit": {"doc_edit": 0.6, "chat_message": 0.3, "task_update": 0.1},
    "email": {"calendar_event": 0.4, "chat_message": 0.4, "email": 0.2},
    "task_update": {"chat_message": 0.5, "doc_edit": 0.3, "task_update": 0.2},
    "calendar_event": {"chat_message": 0.8, "email": 0.2}
}

SOURCE_APP_MAPPING = {
    "chat_message": "slack",
    "doc_edit": "notion",
    "email": "gmail",
    "task_update": "linear",
    "calendar_event": "google_calendar"
}

# ==========================================
# 3. Storylines & Facts (The Startup Scenario)
# ==========================================

# A. Durable Facts (Never change)
DURABLE_FACTS = [
    "Sohil is the CEO.",
    "Sharath is the CTO.",
    "Gandhi is the Lead Engineer.",
    "The company was officially incorporated on Jan 1st, 2026.",
    "The company registration number is 99887766.",
    # Multi-hop & Trap ingredients
    "Sohil owns the launch decision.",
    "The infrastructure migration is scheduled for April.",
    "Sohil approved the budget for Q2.",
    "We are hiring a new frontend developer.",
    "The team had dinner at a Mexican restaurant on Friday."
]

# B. Transient Noise (Friends chatting, irrelevant next day)
TRANSIENT_NOISE = [
    "Anyone want to grab pizza for lunch?",
    "I'm feeling really tired today, might sign off early.",
    "Did you guys watch the match last night?",
    "It's raining so heavily, my internet is lagging.",
    "I'll be 5 minutes late to the sync, grabbing coffee."
]

# C. Supersession Chains (Facts that evolve)
# We track the index to progress the story over the 95 days.
SUPERSESSION_CHAINS = {
    "tech_stack": [
        "We are building the backend in Python.",
        "Python is too slow for this, let's migrate the backend to Node.js.",
        "Node.js is consuming too much memory, we are rewriting the backend in Go."
    ],
    "database": [
        "The database is PostgreSQL.",
        "We're temporarily switching back to MongoDB for the migration.",
        "PostgreSQL is our production database again."
    ],
    "company_name": [
        "Let's call the startup 'Project Alpha'.",
        "I bought the domain for 'NexGenAI', that's our new name.",
        "Actually, 'MemoryLayer Inc' sounds way more professional. That is our final name."
    ],
    "office": [
        "Working out of Sohil's garage for now.",
        "We finally got a small co-working space in HSR Layout."
    ],
    "payments": [
        "Integrating PayPal for our first checkout version.",
        "PayPal's API is annoying, I'm ripping it out and putting in Stripe."
    ],
    "launch_date": [
        "We are aiming to launch in Q1.",
        "Q1 is too tight, pushing the launch to Q2.",
        "Launch is officially delayed to Q3."
    ]
}

# D. The Email -> Task Update Multi-hop decision
MULTI_HOP_DECISION = {
    "email": "Hey team, after reviewing AWS vs GCP pricing, let's definitely go with AWS. Please update the tickets.",
    "task_update": "Migrating our infrastructure to AWS as decided in yesterday's email thread."
}

# E. Fact restated 5 times (Deduplication test)
RESTATED_FACT = [
    "Our core product is an AI memory layer.",
    "We are building an AI that remembers things.",
    "The startup focuses on giving memory to AI agents.",
    "Memory for AI is our primary product.",
    "We develop memory systems for AI products."
]

# F. Realistic Filler Events
FILLER_EVENTS = {
    "chat_message": [
        "Can someone review my PR for the auth service?",
        "Is the staging environment down again?",
        "Merging the hotfix to main now.",
        "Could you check the logs? I'm seeing a weird 500 error.",
        "Let's sync up later today about the frontend rewrite.",
        "The new UI components are looking great.",
        "I just pushed the changes we discussed.",
        "Can we get approval on the new design mocks?",
        "I'll take a look at that bug ticket after lunch."
    ],
    "doc_edit": [
        "Updated the API documentation with the new endpoints.",
        "Added notes from the morning sync.",
        "Drafting the architectural decision record for the cache layer.",
        "Fixing typos in the user onboarding guide.",
        "Expanding the section on environment variables in the README.",
        "Added the new Q3 roadmap draft."
    ],
    "email": [
        "Weekly status update: all metrics are looking solid.",
        "Following up on the vendor contract discussion.",
        "Please review the attached invoice for the cloud services.",
        "Sending over the slide deck for tomorrow's demo.",
        "Summary of our meeting with the investors.",
        "Just a reminder that the all-hands meeting is on Friday."
    ],
    "task_update": [
        "Moved ticket to In Progress.",
        "Blocked on design assets, adding the blocked label.",
        "PR merged, closing this task.",
        "Updated the description to clarify the acceptance criteria.",
        "Assigned to QA for testing.",
        "Bumping the priority on this issue, it's affecting prod."
    ],
    "calendar_event": [
        "Weekly Engineering Sync",
        "1-on-1 Catchup",
        "Product Roadmap Planning",
        "Client Demo Prep",
        "Architecture Review Board",
        "Post-Mortem for yesterday's outage"
    ]
}

def generate_event(user_id: str, timestamp: datetime, event_type: str, state: Dict) -> Dict[str, Any]:
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    
    # 1. Decide what kind of payload this event carries
    rand_val = random.random()
    payload_text = ""
    
    if rand_val < 0.1:
        # 10% chance: Durable Fact
        payload_text = random.choice(DURABLE_FACTS)
    elif rand_val < 0.3:
        # 20% chance: Transient Noise (High because they are friends)
        payload_text = random.choice(TRANSIENT_NOISE)
    elif rand_val < 0.35 and state["restated_count"] < 5:
        # 5% chance: Deduplication fact
        payload_text = RESTATED_FACT[state["restated_count"]]
        state["restated_count"] += 1
    elif rand_val < 0.5:
        # 15% chance: Progress a supersession chain
        chain_key = random.choice(list(SUPERSESSION_CHAINS.keys()))
        chain = SUPERSESSION_CHAINS[chain_key]
        current_idx = state["chain_progress"][chain_key]
        
        if current_idx < len(chain):
            payload_text = chain[current_idx]
            state["chain_progress"][chain_key] += 1
        else:
            payload_text = random.choice(FILLER_EVENTS[event_type])
    elif rand_val < 0.55 and not state["multi_hop_done"]:
        # The specific Email -> Task multi-hop requirement
        if event_type == "email" and not state["email_sent"]:
            payload_text = MULTI_HOP_DECISION["email"]
            state["email_sent"] = True
        elif event_type == "task_update" and state["email_sent"]:
            payload_text = MULTI_HOP_DECISION["task_update"]
            state["multi_hop_done"] = True
        else:
            payload_text = random.choice(FILLER_EVENTS[event_type])
    else:
        # 45% chance: Realistic filler work event
        payload_text = random.choice(FILLER_EVENTS[event_type])

    return {
        "event_id": event_id,
        "user_id": user_id,
        "timestamp": timestamp.isoformat() + "Z",
        "type": event_type,
        "payload": {"text": payload_text},
        "source_app": SOURCE_APP_MAPPING[event_type]
    }

def generate_all_events():
    events = []
    current_time = START_DATE
    
    # Track progress of our storylines
    state = {
        "chain_progress": {k: 0 for k in SUPERSESSION_CHAINS.keys()},
        "restated_count": 0,
        "email_sent": False,
        "multi_hop_done": False,
        "last_event_type": "chat_message" # Starting state for the Markov Chain
    }
    
    # We need a 2+ week quiet stretch. Let's schedule a 16-day vacation around day 40.
    vacation_start_day = 40
    vacation_end_day = 40 + 16

    for day in range(TOTAL_DAYS):
        # The 2+ week quiet stretch requirement
        if vacation_start_day <= day <= vacation_end_day:
            current_time += timedelta(days=1)
            continue

        if day % 7 in [5, 6]: # Skip weekends
            current_time += timedelta(days=1)
            continue
            
        # Simulate 5 to 15 events per active day (to ensure we get well over 200 events)
        num_events_today = random.randint(5, 15)
        
        for _ in range(num_events_today):
            user = random.choice(USERS)
            
            # Using the Markov Chain Transition Matrix weights for realistic flow
            current_event_state = state["last_event_type"]
            possible_next_events = list(TRANSITION_MATRIX[current_event_state].keys())
            probabilities = list(TRANSITION_MATRIX[current_event_state].values())
            
            event_type = random.choices(possible_next_events, weights=probabilities, k=1)[0]
            state["last_event_type"] = event_type
            
            current_time += timedelta(minutes=random.randint(5, 120))
            
            event = generate_event(user, current_time, event_type, state)
            events.append(event)
            
        # Reset to next morning
        current_time = current_time.replace(hour=9, minute=0) + timedelta(days=1)

    events.sort(key=lambda x: x["timestamp"])
    return events

if __name__ == "__main__":
    events = generate_all_events()
    print(f"Generated {len(events)} events.")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    with open("data/events.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    
    print("Saved to data/events.jsonl")
