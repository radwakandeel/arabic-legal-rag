"""Retrieval eval on eval/hard_questions.json: an answer counts if ANY accepted article is retrieved."""
import json
from pathlib import Path

from arabic_legal_rag.index import search

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "hard_questions.json"


def evaluate(k_max: int = 50, search_fn=search) -> dict:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    ranks = {}
    for q in data:
        res = [r["article_number"] for r in search_fn(q["question"], k=k_max)]
        hits = [i for i, n in enumerate(res, 1) if n in q["articles"]]
        ranks[q["id"]] = hits[0] if hits else None
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks.values() if r and r <= k) / n for k in (1, 5, 10)}
    out["mrr"] = sum(1 / r for r in ranks.values() if r) / n
    out["ranks"] = ranks
    return out


if __name__ == "__main__":
    res = evaluate()
    for k, v in res.items():
        print(k, v if k == "ranks" else round(v, 3))
