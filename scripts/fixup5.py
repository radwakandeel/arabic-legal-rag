import json, re
p = "data/processed/articles.json"
arts = json.load(open(p, encoding="utf-8"))
by = {a["article_number"]: a for a in arts}
a1021, a1022 = by[1021], by[1022]
ar = (a1021["text_ar"] or "").rstrip()
if (a1022["text_ar"] or "").strip():
    ar = ar + "\n" + a1022["text_ar"].strip()
m = re.search(r"\n?\(٢\)", ar)
assert m, "marker (٢) not found"
a1021["text_ar"] = ar[:m.start()].strip()
a1022["text_ar"] = ar[m.start():].strip()
a1022["ar_incomplete"] = True
json.dump(arts, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("1021 len:", len(a1021["text_ar"]), "| 1022 len:", len(a1022["text_ar"]))
