import json
from pathlib import Path

from arabic_legal_rag.index import search

QUESTIONS = Path("eval/questions.json")


def rank_of(results, expected):
    for i, r in enumerate(results, 1):
        if r["article_number"] == expected:
            return i
    return None


def main():
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    stats = {"ar": [], "en": []}
    top1 = {}
    for it in items:
        for lang in ("ar", "en"):
            res = search(it[f"question_{lang}"], k=10)
            rank = rank_of(res, it["article"])
            stats[lang].append((it["article"], rank))
            top1[(it["article"], lang)] = res[0]["article_number"]
    for lang in ("ar", "en"):
        ranks = [r for _, r in stats[lang]]
        n = len(ranks)
        h1 = sum(1 for r in ranks if r == 1)
        h5 = sum(1 for r in ranks if r and r <= 5)
        h10 = sum(1 for r in ranks if r and r <= 10)
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{lang}: hit@1 {h1}/{n}  hit@5 {h5}/{n}  hit@10 {h10}/{n}  MRR {mrr:.3f}")
    same = sum(1 for it in items if top1[(it["article"], "ar")] == top1[(it["article"], "en")])
    print(f"same top-1 article for ar and en: {same}/{len(items)}")
    print()
    print("articles not in top 5:")
    for lang in ("ar", "en"):
        for art, rank in stats[lang]:
            if rank is None or rank > 5:
                print(f"  {lang} article {art}: rank {rank}, top-1 was {top1[(art, lang)]}")


if __name__ == "__main__":
    main()
