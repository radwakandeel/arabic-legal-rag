"""Retrieval experiments logged to MLflow.

Brute-force cosine search that mirrors index.search (top 20 per language, then fuse).
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

from arabic_legal_rag.eval_hard import QUESTIONS

CHUNKS = Path("data/processed/chunks.json")
ARTICLES = Path("data/processed/articles.json")
CACHE = Path("data/emb_cache")
N_PER_INDEX = 20
K_MAX = 50
CHUNK_MAX_CHARS = 1500  # default of chunk.py

E5S = "intfloat/multilingual-e5-small"
E5B = "intfloat/multilingual-e5-base"
BGE = "BAAI/bge-m3"

CONFIGS = [
    dict(name="r1-e5small-mindist", model=E5S, fusion="min_distance", sections=True),
    dict(name="r2-e5small-rrf", model=E5S, fusion="rrf", sections=True),
    dict(name="r3-e5small-no-headings", model=E5S, fusion="min_distance", sections=False),
    dict(name="r4-e5base-mindist", model=E5B, fusion="min_distance", sections=True),
    dict(name="r5-bgem3-mindist", model=BGE, fusion="min_distance", sections=True, slow=True),
]

_models = {}


def get_model(name):
    if name not in _models:
        m = SentenceTransformer(name)
        m.max_seq_length = 512
        _models[name] = m
    return _models[name]


def build_docs(chunks, sections, lang, use_sections):
    docs, arts = [], []
    for c in chunks:
        if not (c.get(f"text_{lang}") or "").strip():
            continue
        sec = sections.get(c["article_number"], {}).get(lang, "") if use_sections else ""
        docs.append((sec + "\n" if sec else "") + c[f"text_{lang}"])
        arts.append(c["article_number"])
    return docs, np.array(arts)


def doc_vectors(model_name, lang, use_sections, docs):
    tag = model_name.replace("/", "_")
    f = CACHE / f"{tag}_{lang}_{'sec' if use_sections else 'nosec'}.npy"
    if f.exists():
        v = np.load(f)
        if v.shape[0] == len(docs):
            return v
    print(f"  embedding {len(docs)} {lang} docs with {model_name} ...", flush=True)
    prefix = "passage: " if "e5" in model_name else ""
    v = get_model(model_name).encode(
        [prefix + d for d in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    return v


def query_vector(model_name, text):
    prefix = "query: " if "e5" in model_name else ""
    return get_model(model_name).encode([prefix + text], normalize_embeddings=True)[0]


def ranked_lists(qv, mats, ids):
    out = {}
    for lang in ("ar", "en"):
        d = 1 - mats[lang] @ qv
        order = np.argsort(d)[:N_PER_INDEX]
        out[lang] = [(int(ids[lang][i]), float(d[i])) for i in order]
    return out


def fuse(lists, fusion):
    if fusion == "min_distance":
        best = {}
        for items in lists.values():
            for n, d in items:
                if n not in best or d < best[n]:
                    best[n] = d
        return [n for n, _ in sorted(best.items(), key=lambda kv: kv[1])]
    score = {}
    for items in lists.values():
        seen, r = set(), 0
        for n, _ in items:
            if n in seen:
                continue
            seen.add(n)
            r += 1
            score[n] = score.get(n, 0) + 1 / (60 + r)
    return [n for n, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def metrics(ranks):
    n = len(ranks)
    out = {f"hit_at_{k}": sum(1 for r in ranks.values() if r and r <= k) / n for k in (1, 5, 10)}
    out["mrr"] = sum(1 / r for r in ranks.values() if r) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include slow runs (bge-m3)")
    ap.add_argument("--only", nargs="*", help="run only these run names")
    a = ap.parse_args()

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    arts = json.loads(ARTICLES.read_text(encoding="utf-8"))
    sections = {
        x["article_number"]: {"ar": x.get("section_ar", ""), "en": x.get("section_en", "")}
        for x in arts
    }
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic-legal-rag-retrieval")

    table = []
    for cfg in CONFIGS:
        if a.only:
            if cfg["name"] not in a.only:
                continue
        elif cfg.get("slow") and not a.all:
            continue
        print("RUN", cfg["name"], flush=True)
        t0 = time.time()
        mats, ids = {}, {}
        for lang in ("ar", "en"):
            docs, ids[lang] = build_docs(chunks, sections, lang, cfg["sections"])
            mats[lang] = doc_vectors(cfg["model"], lang, cfg["sections"], docs)
        ranks = {}
        for q in questions:
            qv = query_vector(cfg["model"], q["question"])
            fused = fuse(ranked_lists(qv, mats, ids), cfg["fusion"])[:K_MAX]
            hits = [i for i, n in enumerate(fused, 1) if n in q["articles"]]
            ranks[q["id"]] = hits[0] if hits else None
        m = metrics(ranks)
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.log_params(
                {
                    "embedding_model": cfg["model"],
                    "fusion": cfg["fusion"],
                    "use_section_headings": cfg["sections"],
                    "chunk_max_chars": CHUNK_MAX_CHARS,
                    "n_per_index": N_PER_INDEX,
                    "n_questions": len(questions),
                }
            )
            mlflow.log_metrics(m)
            mlflow.log_dict(ranks, "ranks.json")
        table.append((cfg["name"], m, round(time.time() - t0)))

    print()
    print(f"{'run':26} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'mrr':>6} {'sec':>5}")
    for name, m, secs in table:
        print(
            f"{name:26} {m['hit_at_1']:6.3f} {m['hit_at_5']:6.3f} {m['hit_at_10']:7.3f} {m['mrr']:6.3f} {secs:5d}"
        )


if __name__ == "__main__":
    main()
