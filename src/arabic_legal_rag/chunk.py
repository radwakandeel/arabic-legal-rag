import argparse
import json
import re
from pathlib import Path

IN = Path("data/processed/articles.json")
OUT = Path("data/processed/chunks.json")
PARA = re.compile(r"(?=^[ \t]*\([ \t]*[0-9\u0660-\u0669]+[ \t]*\))", re.M)


def split_text(text, max_chars):
    pieces = [p for p in PARA.split(text) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur.strip())
            cur = ""
        cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text.strip()]


def build(max_chars):
    arts = json.loads(IN.read_text(encoding="utf-8"))
    out = []
    for a in arts:
        n = a["article_number"]
        base = {
            "article_number": n,
            "citation": a["citation"],
            "is_repealed": a["is_repealed"],
            "ar_incomplete": a.get("ar_incomplete", False),
        }
        if a["is_repealed"]:
            out.append(
                {
                    **base,
                    "id": f"{n}",
                    "part": 1,
                    "text_ar": f"المادة {n} ملغاة.",
                    "text_en": f"Article {n} has been repealed.",
                }
            )
            continue
        parts = split_text(a["text_ar"], max_chars)
        for i, t in enumerate(parts, 1):
            out.append(
                {
                    **base,
                    "id": f"{n}" if len(parts) == 1 else f"{n}-{i}",
                    "part": i,
                    "text_ar": t,
                    "text_en": a["text_en"] if i == 1 else "",
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()
    chunks = build(args.max_chars)
    OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [len(c["text_ar"]) for c in chunks]
    over = [c["id"] for c in chunks if len(c["text_ar"]) > args.max_chars]
    print("chunks:", len(chunks), "articles:", len({c["article_number"] for c in chunks}))
    print("extra parts from splitting:", sum(1 for c in chunks if c["part"] > 1), "max chars:", max(lengths))
    print("over the limit (one long paragraph):", len(over), over[:10])


if __name__ == "__main__":
    main()
