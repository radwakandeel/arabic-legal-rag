import json, re, sys
from pathlib import Path

p = Path("data/processed/articles.json")
records = json.loads(p.read_text(encoding="utf-8"))
DIG = r"[0-9\u0660-\u0669]+"
# a paragraph number at the START of a line, with any mix of ( ) and spaces
PAT = re.compile(r"^[ \t]*[()][ \t]*(" + DIG + r")[ \t]*[()]", re.M)

changed = []
for r in records:
    old = r["text_ar"] or ""
    new = PAT.sub(lambda m: "(" + m.group(1) + ")", old)
    if new != old:
        changed.append((r["article_number"], old, new))
        if "--apply" in sys.argv:
            r["text_ar"] = new

print(len(changed), "articles changed")
print("first 15:", [c[0] for c in changed[:15]])
for n, old, new in changed[:3]:
    print("=" * 20, n)
    print("OLD:", repr(old[:60]))
    print("NEW:", repr(new[:60]))

if "--apply" in sys.argv:
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("APPLIED")
else:
    print("dry run only, nothing written")
