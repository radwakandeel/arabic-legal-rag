import json
from pathlib import Path

ARTICLES = json.loads(Path("data/processed/articles.json").read_text(encoding="utf-8"))
REPEALED = set(range(54, 81)) | set(range(389, 418))
MAX_CHARS = 6000


def test_numbers_are_contiguous():
    nums = [a["article_number"] for a in ARTICLES]
    assert nums == list(range(1, 1150))


def test_every_live_article_has_both_texts():
    for a in ARTICLES:
        if not a["is_repealed"]:
            assert a["text_ar"].strip(), f"article {a['article_number']} has no Arabic"
            assert a["text_en"].strip(), f"article {a['article_number']} has no English"


def test_no_absurdly_long_record():
    for a in ARTICLES:
        n = len(a["text_ar"]) + len(a["text_en"])
        assert n < MAX_CHARS, f"article {a['article_number']} is {n} chars"


def test_repealed_flags():
    flagged = {a["article_number"] for a in ARTICLES if a["is_repealed"]}
    assert flagged == REPEALED


def test_known_incomplete_article():
    a = next(x for x in ARTICLES if x["article_number"] == 1022)
    assert a.get("ar_incomplete") is True
