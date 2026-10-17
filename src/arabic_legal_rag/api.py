"""FastAPI app: POST /ask and GET /health."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from arabic_legal_rag import rag

app = FastAPI(title="Arabic Legal RAG")


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "healthy", "documents_indexed": len(rag.articles())}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag.ask(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
