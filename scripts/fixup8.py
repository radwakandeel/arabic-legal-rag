import json, re, sys
from pathlib import Path

p = Path("data/processed/articles.json")
records = json.loads(p.read_text(encoding="utf-8"))
END = re.compile(r"[.:;?!؟؛]\s*$")
MAX_LEN, MAX_LINES = 60, 4
SKIP = {936}  # a sentence fragment, not a heading


def split_heading(text):
    lines = (text or "").split("\n")
    k = len(lines)
    while k > 0 and len(lines) - k < MAX_LINES:
        line = lines[k - 1].strip()
        if not line or len(line) > MAX_LEN or line.startswith("(") or END.search(line):
            break
        k -= 1
    if k == len(lines) or k == 0:
        return text, ""
    if not END.search(lines[k - 1].strip()):
        return text, ""
    return "\n".join(lines[:k]).strip(), "\n".join(lines[k:]).strip()


apply = "--apply" in sys.argv
if apply and any("section_en" in r for r in records):
    sys.exit("already applied: rebuild from extract.py first")

cur_ar = cur_en = ""
changed = []
for r in records:
    if r["is_repealed"]:
        continue
    n = r["article_number"]
    if n in SKIP:
        ar, h_ar, en, h_en = r["text_ar"], "", r["text_en"], ""
    else:
        ar, h_ar = split_heading(r["text_ar"])
        en, h_en = split_heading(r["text_en"])
    if apply:
        r["text_ar"], r["text_en"] = ar, en
        r["section_ar"], r["section_en"] = cur_ar, cur_en
    if h_ar or h_en:
        changed.append(n)
    if h_ar:
        cur_ar = h_ar.replace("\n", " / ")
    if h_en:
        cur_en = h_en.replace("\n", " / ")

print(len(changed), "articles with a trailing heading block")
if apply:
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("APPLIED")
else:
    print("dry run only, nothing written")
