from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ReAct import autonomous_agent
from fastmcp import FastMCP
import requests
import threading

app = FastAPI(title="Legal Advisor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INTENT_SERVICE_URL = "http://localhost:8001/predict"

class Query(BaseModel):
    question: str

def get_intent(question: str) -> str:
    try:
        response = requests.post(
            INTENT_SERVICE_URL,
            json={"query": question},
            timeout=5
        )
        return response.json().get("intent", "legal_query")
    except Exception:
        print(">>> Intent service unavailable, using default: legal_query")
        return "legal_query"

# ── MCP - runs in background  ──────────────────

mcp = FastMCP("Legal Advisor")

@mcp.tool()
def ask_legal_question(question: str) -> str:
    return autonomous_agent(question)

async def start_mcp():
    thread = threading.Thread(target=mcp.run, daemon=True)
    thread.start()
    print(">>> MCP Server started in background")




@app.get("/health")
def health():
    return {"status": "ok"}

# ── New endpoint — returns intent instantly before agent runs ──────────────────
@app.post("/api/v1/intent")
async def intent_check(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    intent = get_intent(query.question)
    print(f">>> Intent Check: {query.question[:50]} → {intent}")
    return {"intent": intent}

# ── Main endpoint — runs agent with already known intent ───────────────────────
@app.post("/api/v1/ask")
async def ask(query: Query):
    print("\n>>> API HIT")
    print("Query:", query.question)

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    intent = get_intent(query.question)
    print(f">>> Intent: {intent}")

    try:
        answer = autonomous_agent(query.question, intent)
        return {"answer": answer, "intent": intent}
    except Exception as e:
        print(">>> ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Legal Advisor API Running"}