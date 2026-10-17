# Arabic Legal RAG

Question answering over the Egyptian Civil Code (bilingual Arabic/English).

## Getting the data (DVC)

The raw PDF, processed corpus and vector index are stored on DagsHub
(`https://dagshub.com/radwakandeel/arabic-legal-rag`). Reading them requires a
free DagsHub account: anonymous access is rejected.

1. Create a free DagsHub account and generate an access token
   (user menu > Settings > Tokens).
2. Inside the cloned repo:

```bash
uv sync
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-token>
uv run dvc pull
```

`--local` keeps your credentials out of git (`.dvc/config.local` is ignored).
Without credentials, `dvc pull` may report the cache files as missing and then
fail on `civil_code.pdf`. That message means authentication failed, not that the
data is missing.

## Rebuild the corpus and index

The DVC pipeline extracts the bilingual Civil Code PDF, applies the checked-in
corpus corrections, chunks by article, and rebuilds the Chroma index:

```bash
uv run dvc repro
```

The pipeline uses `BAAI/bge-m3` for embeddings. Its model weights are downloaded
from Hugging Face on the first index build.

Known limits of the reproducibility claim:

- Rebuilding the corpus from the PDF produced an `articles.json` identical to
  the committed one (checked with `cmp`).
- The `index` stage was not verified end to end through `dvc repro`. The first
  run was killed (exit code 137, probably out of memory on a WSL machine with
  about 3.7GB). `index.py` now adds documents in batches of 64, but that version
  has not been run to completion.
- The index recorded in `dvc.lock` was restored from the cached embeddings (same
  model, same text), not produced by a full `dvc repro`. Retrieval metrics on the
  restored index matched the original index exactly.

## Retrieval behaviour

Retrieval always returns the nearest indexed articles, even for an unrelated
question. The answer model is instructed to say when those retrieved articles
do not contain an answer; this is not a retrieval confidence or relevance gate.
