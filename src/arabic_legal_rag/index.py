import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/processed/chunks.json")
ARTICLES = Path("data/processed/articles.json")
DB_DIR = "data/chroma"
MODEL = "BAAI/bge-m3"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
        _model.max_seq_length = 512
    return _model


def embed(texts, prefix, batch=32):
    prefix = prefix if "e5" in MODEL else ""  # bge-m3 takes no prefixes
    vecs = get_model().encode(
        [prefix + t for t in texts],
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return vecs.tolist()


def load_sections():
    arts = json.loads(ARTICLES.read_text(encoding="utf-8"))
    return {
        a["article_number"]: {"ar": a.get("section_ar", ""), "en": a.get("section_en", "")}
        for a in arts
    }


def build():
    sections = load_sections()
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=DB_DIR)
    for lang in ("ar", "en"):
        name = f"civil_code_{lang}"
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        rows = [(i, c) for i, c in enumerate(chunks) if (c.get(f"text_{lang}") or "").strip()]
        ids = [f"{lang}-{i}" for i, _ in rows]
        docs = []
        for _, c in rows:
            sec = sections.get(c["article_number"], {}).get(lang, "")
            docs.append((sec + "\n" if sec else "") + c[f"text_{lang}"])
        metas = [
            {
                "article_number": c["article_number"],
                "part": c.get("part", 1),
                "citation": c["citation"],
                "is_repealed": bool(c["is_repealed"]),
                "ar_incomplete": bool(c.get("ar_incomplete", False)),
            }
            for _, c in rows
        ]
        vecs = embed(docs, "passage: ")
        for s in range(0, len(ids), 500):
            col.add(
                ids=ids[s:s + 500],
                documents=docs[s:s + 500],
                metadatas=metas[s:s + 500],
                embeddings=vecs[s:s + 500],
            )
        print(lang, "indexed:", col.count())


def search(question, k=5, n_per_index=20):
    client = chromadb.PersistentClient(path=DB_DIR)
    qvec = embed([question], "query: ")
    best = {}
    for lang in ("ar", "en"):
        col = client.get_collection(f"civil_code_{lang}")
        res = col.query(query_embeddings=qvec, n_results=n_per_index)
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            n = meta["article_number"]
            if n not in best or dist < best[n][0]:
                best[n] = (dist, lang)
    top = sorted(best.items(), key=lambda kv: kv[1][0])[:k]
    return [
        {"article_number": n, "distance": round(d, 4), "matched_lang": lang}
        for n, (d, lang) in top
    ]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("search")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        for r in search(a.question, a.k):
            print(r)


if __name__ == "__main__":
    main()
