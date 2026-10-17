"""Retrieval eval on eval/questions_50.json: a hit means ANY accepted article is retrieved."""
import json
from pathlib import Path

from arabic_legal_rag.index import search

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions_50.json"


def load(split=None, include_weak=True):
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    out = []
    for q in data:
        if q["type"] != "retrieval" or q["status"] != "reviewed":
            continue
        if split and q["split"] != split:
            continue
        if not include_weak and "weak" in q["note"]:
            continue
        out.append(q)
    return out


def summarize(ranks: dict) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in (1, 5, 10):
        out[f"hit@{k}"] = round(sum(1 for r in ranks.values() if r and r <= k) / n, 3)
    out["mrr"] = round(sum(1 / r for r in ranks.values() if r) / n, 3)
    return out


def evaluate(split=None, include_weak=True, k_max=50, search_fn=search) -> dict:
    ranks = {}
    for q in load(split, include_weak):
        res = [r["article_number"] for r in search_fn(q["question"], k=k_max)]
        hits = [i for i, n in enumerate(res, 1) if n in q["articles"]]
        ranks[q["id"]] = hits[0] if hits else None
    return {"summary": summarize(ranks), "ranks": ranks}


if __name__ == "__main__":
    for label, kw in [
        ("dev (all)", dict(split="dev")),
        ("dev (no weak)", dict(split="dev", include_weak=False)),
        ("test (all)", dict(split="test")),
    ]:
        res = evaluate(**kw)
        print(label, res["summary"])
