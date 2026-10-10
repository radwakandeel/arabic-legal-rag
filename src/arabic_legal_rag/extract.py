import bisect
import json
import re
from pathlib import Path

import pdfplumber

PDF = Path("data/raw/civil_code.pdf")
OUT = Path("data/processed/articles.json")
AR_DIGITS = {0x660 + i: ord(str(i)) for i in range(10)}
DIG = r"[0-9\u0660-\u0669]+"
LETTER = r"[\u0621-\u064A]"
JUNK = r"[()\s\u200e\u200f]*"
EN_HEAD = r"^\s*Article\s*(" + DIG + r")\b\s*[().:]*\s*(.*)$"
AR_HEAD = "^\\s*\u0645\u0627\u062f\u0629" + JUNK + "(" + DIG + ")" + JUNK + "$"
EN_NOTE = re.compile(
    r"^\s*Articles?\s*(" + DIG + r")\s*(?:-|\u2013|to)\s*(" + DIG + r")\s+repealed",
    re.I,
)
AR_NOTE = re.compile(
    "^\\s*\u0627\u0644\u0645\u0648\u0627\u062f\\s+\u0645\u0646\\s+("
    + DIG
    + r")\s+\S+\s+("
    + DIG
    + r")\s*$"
)


def to_int(s):
    return int(s.translate(AR_DIGITS))


def fix_ar_line(line):
    s = line[::-1]
    s = re.sub(DIG, lambda m: m.group()[::-1], s)
    s = re.sub(r"\((" + DIG + r")\(", r"(\1)", s)
    return s


def fix_lam_alef(s):
    s = re.sub("\u0627([\u0625\u0623\u0622])\u0644", "\u0627\u0644\\1", s)
    s = s.replace("\u0627\u0627\u0644", "\u0627\u0644\u0627")
    s = re.sub("(?<!" + LETTER + ")\u0627\u0644(?!" + LETTER + ")", "\u0644\u0627", s)
    return s


def lis_indices(nums):
    tails, tails_idx = [], []
    prev = [-1] * len(nums)
    for i, x in enumerate(nums):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x)
            tails_idx.append(i)
        else:
            tails[j] = x
            tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    k = tails_idx[-1] if tails_idx else -1
    res = []
    while k != -1:
        res.append(k)
        k = prev[k]
    return res[::-1]


def parse_articles(lines, head):
    cands = []
    for i, line in enumerate(lines):
        m = re.match(head, line)
        if m:
            rest = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            cands.append((i, to_int(m.group(1)), rest))
    keep = lis_indices([c[1] for c in cands])
    heads = [cands[k] for k in keep]
    out = {}
    for k, (i, n, rest) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.setdefault(n, "\n".join([rest] + lines[i + 1:end]).strip())
    return out


def rng(nums):
    out, s, p = [], None, None
    for n in sorted(nums):
        if s is None:
            s = p = n
        elif n == p + 1:
            p = n
        else:
            out.append((s, p))
            s = p = n
    if s is not None:
        out.append((s, p))
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def main():
    en_lines, ar_lines = [], []
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            w, h = page.width, page.height
            left = page.crop((0, 0, w / 2, h)).extract_text() or ""
            right = page.crop((w / 2, 0, w, h)).extract_text() or ""
            en_lines += left.splitlines()
            ar_lines += [fix_ar_line(x) for x in right.splitlines()]

    marker = "\u0646\u0635\u0648\u0635 \u0627\u0644\u0642\u0627\u0646\u0648\u0646"
    start = next((i for i, x in enumerate(ar_lines) if marker in x), 0)
    ar_lines = ar_lines[start:]

    notes = {"en": [], "ar": []}
    for line in en_lines:
        m = EN_NOTE.match(line)
        if m:
            notes["en"].append((to_int(m.group(1)), to_int(m.group(2))))
    for line in ar_lines:
        m = AR_NOTE.match(line)
        if m:
            notes["ar"].append((to_int(m.group(1)), to_int(m.group(2))))
    en_lines = [x for x in en_lines if not EN_NOTE.match(x)]
    ar_lines = [x for x in ar_lines if not AR_NOTE.match(x)]

    repealed = set()
    for a, b in notes["en"] + notes["ar"]:
        repealed.update(range(a, b + 1))

    en = parse_articles(en_lines, EN_HEAD)
    ar = parse_articles(ar_lines, AR_HEAD)
    hi = max(set(en) | set(ar))
    nums = sorted(n for n in set(en) | set(ar) | repealed if 1 <= n <= hi)

    records = []
    for n in nums:
        t_ar = fix_lam_alef(ar.get(n, ""))
        t_en = en.get(n, "")
        records.append(
            {
                "article_number": n,
                "text_ar": t_ar,
                "text_en": t_en,
                "is_repealed": n in repealed and not t_ar and not t_en,
                "citation": f"Egyptian Civil Code, Article {n}",
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    live = [r for r in records if not r["is_repealed"]]
    have = {r["article_number"] for r in records}
    print("preamble lines skipped:", start)
    print("articles:", len(records), "first:", nums[0], "last:", nums[-1])
    print("repeal notes en:", notes["en"], "ar:", notes["ar"])
    print("repealed:", rng([r["article_number"] for r in records if r["is_repealed"]]))
    print("no_ar:", rng([r["article_number"] for r in live if not r["text_ar"]]))
    print("no_en:", rng([r["article_number"] for r in live if not r["text_en"]]))
    print("gaps:", rng([n for n in range(1, hi + 1) if n not in have]))
    miss = [r["article_number"] for r in live if not r["text_en"]][:4]
    for n in miss:
        hits = [x for x in en_lines if re.search(r"Article\s*%d\b" % n, x)][:2]
        print("en probe", n, [x[:90] for x in hits])


if __name__ == "__main__":
    main()
