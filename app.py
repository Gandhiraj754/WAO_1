from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import ask_question
from typing import List

app = FastAPI(title="WAO-Recall API")

@app.on_event("startup")
async def print_clickable_link():
    print("\n" + "="*60)
    print("🚀 API is running perfectly inside Docker!")
    print("👉 CLICK HERE TO TEST THE API: http://localhost:8000/docs")
    print("="*60 + "\n")

class AskRequest(BaseModel):
    user_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    cited_memory_ids: List[str]
    confidence: float

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        # Calls the function from retrieval.py
        result = ask_question(req.user_id, req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
