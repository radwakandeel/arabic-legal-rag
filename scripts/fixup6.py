import json, re
from pathlib import Path

p = Path("data/processed/articles.json")
records = json.loads(p.read_text(encoding="utf-8"))
by_n = {r["article_number"]: r for r in records}

# 54 is part of the repealed range 54-80
by_n[54].update(text_ar="", text_en="", is_repealed=True)

# 452: heading is "rticle 452" (missing letter) inside 451's English text
t = by_n[451]["text_en"]
m = re.search(r"^\s*rticle\s*452\s*$", t, re.M)
if m and not by_n[452]["text_en"].strip():
    by_n[451]["text_en"] = t[: m.start()].strip()
    by_n[452]["text_en"] = t[m.end():].strip()
else:
    print("WARNING: 452 pattern not found or 452 already has English")

print("452 english starts:", repr(by_n[452]["text_en"][:70]))
print("451 english ends:", repr(by_n[451]["text_en"][-60:]))

p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
live = [r for r in records if not r["is_repealed"]]
print("repealed:", sum(1 for r in records if r["is_repealed"]))
print("missing arabic:", [r["article_number"] for r in live if not r["text_ar"]])
print("missing english:", [r["article_number"] for r in live if not r["text_en"]])
