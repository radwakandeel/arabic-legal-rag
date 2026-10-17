import argparse
import json
import re
from pathlib import Path

IN = Path("data/processed/articles.json")
OUT = Path("data/processed/chunks.json")
PARA = re.compile(r"(?=^[ \t]*\([ \t]*[0-9\u0660-\u0669]+[ \t]*\))", re.M)


def split_text(text, max_chars, overlap=0):
    """Pack paragraphs up to max_chars; repeat the last `overlap` paragraphs at the start of the next chunk."""
    pieces = [p for p in PARA.split(text) if p.strip()]
    chunks, cur = [], []
    for p in pieces:
        if cur and sum(len(x) for x in cur) + len(p) > max_chars:
            chunks.append("".join(cur).strip())
            cur = cur[-overlap:] if overlap > 0 else []
        cur.append(p)
    if "".join(cur).strip():
        chunks.append("".join(cur).strip())
    return chunks or [text.strip()]


def build(max_chars, overlap=0):
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
        parts = split_text(a["text_ar"], max_chars, overlap)
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
    ap.add_argument("--overlap", type=int, default=0, help="paragraphs repeated at the start of the next chunk")
    args = ap.parse_args()
    chunks = build(args.max_chars, args.overlap)
    OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [len(c["text_ar"]) for c in chunks]
    over = [c["id"] for c in chunks if len(c["text_ar"]) > args.max_chars]
    print("chunks:", len(chunks), "articles:", len({c["article_number"] for c in chunks}))
    print("extra parts from splitting:", sum(1 for c in chunks if c["part"] > 1), "max chars:", max(lengths))
    print("over the limit (one long paragraph):", len(over), over[:10])


if __name__ == "__main__":
    main()
