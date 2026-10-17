from fastapi.testclient import TestClient

from arabic_legal_rag import api, rag

client = TestClient(api.app)


def test_health_reports_indexed_documents():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["documents_indexed"] == 1149


def test_empty_question_is_422():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_returns_answer_and_article_numbers(monkeypatch):
    monkeypatch.setattr(rag, "ask", lambda q, k=8: {"answer": "x", "sources": [163, 218]})
    r = client.post("/ask", json={"question": "ما هو الإعذار؟"})
    assert r.status_code == 200
    assert r.json() == {"answer": "x", "sources": [163, 218]}
