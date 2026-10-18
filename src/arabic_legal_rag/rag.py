"""RAG pipeline: retrieve articles from the index, then ask Gemini to answer from them only."""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import httpx

from arabic_legal_rag.index import search

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

SYSTEM = (
    "You are a legal assistant for the Egyptian Civil Code. Answer ONLY from the "
    "articles provided below. Always cite article numbers like (Article 163). "
    "If an article is marked REPEALED, say it is repealed. If the provided articles "
    "do not contain the answer, say clearly that the answer was not found in the "
    "retrieved articles; do not use outside knowledge. Reply in the language of the question."
)


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


@lru_cache(maxsize=1)
def articles() -> dict:
    """article_number -> {text_ar, text_en, is_repealed}, joining multi-part chunks."""
    out = {}
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    for c in sorted(chunks, key=lambda x: (x["article_number"], x.get("part", 1))):
        a = out.setdefault(
            c["article_number"],
            {
                "text_ar": "",
                "text_en": "",
                "is_repealed": bool(c.get("is_repealed", False)),
                "citation": c.get("citation") or f"Egyptian Civil Code, Article {c['article_number']}",
            },
        )
        a["text_ar"] += (c.get("text_ar") or "") + "\n"
        a["text_en"] += (c.get("text_en") or "") + "\n"
    return out


def retrieve(question: str, k: int = 8) -> list[int]:
    return [r["article_number"] for r in search(question, k=k)]


def citations_for(numbers: list[int]) -> list[str]:
    arts = articles()
    return [arts[n]["citation"] for n in numbers]


def build_prompt(question: str, numbers: list[int]) -> str:
    arts = articles()
    parts = []
    for n in numbers:
        a = arts[n]
        tag = " [REPEALED]" if a["is_repealed"] else ""
        parts.append(f"{a['citation']}{tag}\nAR: {a['text_ar'].strip()}\nEN: {a['text_en'].strip()}")
    return f"{SYSTEM}\n\n=== ARTICLES ===\n" + "\n\n".join(parts) + f"\n\n=== QUESTION ===\n{question}"


def generate(prompt: str) -> str:
    _load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing (put it in .env)")
    r = httpx.post(
        URL.format(MODEL),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        },
        timeout=90,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")


def ask(question: str, k: int = 8) -> dict:
    numbers = retrieve(question, k)
    answer = generate(build_prompt(question, numbers))
    return {"answer": answer, "sources": citations_for(numbers)}


if __name__ == "__main__":
    res = ask(" ".join(sys.argv[1:]))
    print(res["answer"])
    print("\nsources:", res["sources"])
