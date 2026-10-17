from fastapi.testclient import TestClient

from arabic_legal_rag import api, rag

client = TestClient(api.app)


def test_health_reports_indexed_documents():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "documents_indexed": 1149}


def test_empty_question_is_422():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_returns_answer_and_citations(monkeypatch):
    cites = ["Egyptian Civil Code, Article 163", "Egyptian Civil Code, Article 218"]
    monkeypatch.setattr(rag, "ask", lambda q, k=8: {"answer": "x", "sources": cites})
    r = client.post("/ask", json={"question": "ما هو الإعذار؟"})
    assert r.status_code == 200
    assert r.json() == {"answer": "x", "sources": cites}


def test_rag_ask_cites_article_numbers(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=8: [163, 218])
    monkeypatch.setattr(rag, "generate", lambda prompt: "answer")
    out = rag.ask("ما هو الإعذار؟")
    assert out["sources"] == [
        "Egyptian Civil Code, Article 163",
        "Egyptian Civil Code, Article 218",
    ]
