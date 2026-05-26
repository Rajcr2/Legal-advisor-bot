import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api import app

client = TestClient(app)


# API Tests
def test_health():
    assert client.get("/health").status_code == 200

def test_empty_question_ask():
    assert client.post("/api/v1/ask", json={"question": ""}).status_code == 400

def test_empty_question_intent():
    assert client.post("/api/v1/intent", json={"question": ""}).status_code == 400

def test_ask_response_structure():
    with patch("api.autonomous_agent", return_value="Test answer"), \
         patch("api.get_intent", return_value="legal_query"):
        res = client.post("/api/v1/ask", json={"question": "What is section 5?"})
        assert res.status_code == 200
        assert "answer" in res.json()
        assert "intent" in res.json()


# Retriever Tests
def test_retriever_returns_list():
    from ReAct import retriever, State
    state: State = {
        "query": "What are conditions for Hindu marriage?",
        "intent": "legal_query",
        "retrieved_docs": [], "selected_docs": [],
        "final_answer": "", "scratchpad": "",
        "last_docs": [], "next_action": ""
    }
    result = retriever(state)
    assert isinstance(result["retrieved_docs"], list)
    assert len(result["retrieved_docs"]) > 0


# Selector Tests
def test_selector_empty_input():
    from ReAct import selector, State
    state: State = {
        "query": "What is section 5?",
        "intent": "legal_query",
        "retrieved_docs": [], "selected_docs": [],
        "final_answer": "", "scratchpad": "",
        "last_docs": [], "next_action": ""
    }
    result = selector(state)
    assert result["selected_docs"] == []


# Intent Service Tests
def test_intent_fallback_on_failure():
    from api import get_intent
    with patch("api.requests.post", side_effect=Exception("down")):
        assert get_intent("What is section 5?") == "legal_query"

def test_intent_valid_value():
    from api import get_intent
    mock = MagicMock()
    mock.json.return_value = {"intent": "situation_based"}
    with patch("api.requests.post", return_value=mock):
        assert get_intent("My employer harassed me") in ["legal_query", "situation_based"]
