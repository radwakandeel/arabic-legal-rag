"""RAGAS evaluation on a subset of eval/questions_50.json.

Metrics: faithfulness, answer relevancy, context precision, context recall.
Design choices (also to be written in the README):
  * The judge LLM is the same Gemini model that writes the answers (possible bias).
  * `reference` for context precision/recall = the text of the gold articles in the
    question file (there is no hand-written reference answer).
  * Answers are cached in the work dir, so a re-run never re-spends generation requests.
  * Every HTTP request to Gemini is throttled to --rpm (retries included) and counted.

Usage:
  uv run python scripts/run_ragas.py --probe          # 1 question, prints requests used
  uv run python scripts/run_ragas.py --dry-run        # retrieval only, no LLM calls
  uv run python scripts/run_ragas.py --n 20 --rpm 8   # the full 4-metric run
  uv run python scripts/run_ragas.py --n 20 --rpm 8 --metrics faithfulness --fail-below 0.75   # CI gate
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from arabic_legal_rag import rag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions_50.json"
GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai/"


class Throttle:
    """Allow at most `rpm` requests per minute across threads and event loops."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.next_slot = 0.0
        self.lock = threading.Lock()
        self.count = 0

    def wait_time(self) -> float:
        with self.lock:
            self.count += 1
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
            return slot - now


def pick_questions(split: str, n: int) -> list[dict]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = [q for q in data if q.get("type") == "retrieval" and q.get("split") == split]
    rows.sort(key=lambda q: q["id"])
    return rows[:n]


def article_block(n: int, with_en: bool) -> str:
    a = rag.articles()[n]
    tag = " [REPEALED]" if a["is_repealed"] else ""
    text = f"{a['citation']}{tag}\nAR: {a['text_ar'].strip()}"
    if with_en:
        text += f"\nEN: {a['text_en'].strip()}"
    return text


def reference_text(numbers: list[int]) -> str:
    arts = rag.articles()
    return "\n\n".join(article_block(n, with_en=False) for n in numbers if n in arts)


def build_rows(questions, k, throttle, cache_path, dry_run):
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows, skipped = [], []
    for q in questions:
        numbers = rag.retrieve(q["question"], k=k)
        contexts = [article_block(n, with_en=True) for n in numbers]  # same text the model saw
        key = f"{q['id']}|k={k}|{rag.MODEL}"
        response = cache.get(key, {}).get("response")
        if response is None and not dry_run:
            prompt = rag.build_prompt(q["question"], numbers)
            for attempt in range(4):
                time.sleep(throttle.wait_time())
                try:
                    response = rag.generate(prompt)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  generate failed for {q['id']} (attempt {attempt + 1}): {str(e)[:120]}")
                    time.sleep(20 * (attempt + 1))
            if response is None:
                skipped.append(q["id"])
                continue
            cache[key] = {"response": response, "numbers": numbers}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(
            {
                "id": q["id"],
                "user_input": q["question"],
                "response": response or "",
                "retrieved_contexts": contexts,
                "reference": reference_text(q["articles"]),
            }
        )
    return rows, skipped


def make_llm(throttle, model, timeout):
    from langchain_openai import ChatOpenAI

    def sync_hook(request):
        time.sleep(throttle.wait_time())

    async def async_hook(request):
        await asyncio.sleep(throttle.wait_time())

    return ChatOpenAI(
        model=model,
        api_key=os.environ["GEMINI_API_KEY"],
        base_url=GEMINI_OPENAI,
        temperature=0,
        max_retries=5,
        timeout=timeout,
        http_client=httpx.Client(event_hooks={"request": [sync_hook]}),
        http_async_client=httpx.AsyncClient(event_hooks={"request": [async_hook]}),
    )


def make_embeddings():
    from langchain_core.embeddings import Embeddings

    from arabic_legal_rag.index import embed

    class Bge(Embeddings):
        def embed_documents(self, texts):
            return np.asarray(embed(list(texts), "")).tolist()

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return Bge()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of dev questions")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=8, help="articles given to the model (production default 8)")
    ap.add_argument("--rpm", type=float, default=10, help="max requests per minute")
    ap.add_argument("--strictness", type=int, default=3, help="answer relevancy: generated questions per answer")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workdir", default=str(Path.home() / "scratch" / "arabic-legal-rag" / "ragas"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "ragas_summary.json"))
    ap.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="comma list; fewer metrics = fewer requests",
    )
    ap.add_argument("--fail-below", type=float, default=None, help="exit 1 if mean faithfulness is below this")
    ap.add_argument("--probe", action="store_true", help="run 1 question and report requests used")
    ap.add_argument("--dry-run", action="store_true", help="retrieval only, no LLM calls")
    args = ap.parse_args()

    rag._load_env()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n = 1 if args.probe else args.n
    questions = pick_questions(args.split, n)
    print(f"model={rag.MODEL} split={args.split} questions={len(questions)} k={args.k} rpm={args.rpm}")

    throttle = Throttle(args.rpm)
    rows, skipped = build_rows(questions, args.k, throttle, workdir / "answers.json", args.dry_run)
    gen_requests = throttle.count
    print(f"rows={len(rows)} skipped={skipped} generation_requests={gen_requests}")
    if args.dry_run:
        r0 = rows[0]
        print("example:", r0["id"], "| contexts:", len(r0["retrieved_contexts"]), "| reference chars:", len(r0["reference"]))
        return

    from ragas import EvaluationDataset, evaluate
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=args.timeout, max_retries=3, max_workers=1)
    llm = LangchainLLMWrapper(
        make_llm(throttle, rag.MODEL, args.timeout),
        run_config=run_config,
        cache=DiskCacheBackend(cache_dir=str(workdir / "ragas_cache")),
        bypass_n=True,  # do not rely on Gemini supporting n>1 in the OpenAI-compatible API
    )
    emb = LangchainEmbeddingsWrapper(make_embeddings(), run_config=run_config)
    available = {
        "faithfulness": lambda: Faithfulness(),
        "answer_relevancy": lambda: ResponseRelevancy(strictness=args.strictness),
        "context_precision": lambda: LLMContextPrecisionWithReference(),
        "context_recall": lambda: LLMContextRecall(),
    }
    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in available]
    if unknown:
        sys.exit(f"unknown metric(s): {unknown}; choose from {list(available)}")
    metrics = [available[m]() for m in wanted]
    if args.fail_below is not None and "faithfulness" not in wanted:
        sys.exit("--fail-below needs the faithfulness metric")
    dataset = EvaluationDataset.from_list(
        [{k: v for k, v in r.items() if k != "id"} for r in rows]
    )
    t0 = time.time()
    result = evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config, show_progress=True
    )
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])
    df.drop(columns=["retrieved_contexts", "reference", "response", "user_input"], errors="ignore").to_csv(
        workdir / "ragas_scores.csv", index=False, encoding="utf-8"
    )

    metric_cols = [c for c in df.columns if c not in ("id", "user_input", "response", "retrieved_contexts", "reference")]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_and_generator_model": rag.MODEL,
        "split": args.split,
        "questions_requested": n,
        "questions_scored": len(rows),
        "skipped_generation": skipped,
        "k_articles": args.k,
        "rpm": args.rpm,
        "answer_relevancy_strictness": args.strictness,
        "total_http_requests": throttle.count,
        "generation_requests": gen_requests,
        "seconds": round(time.time() - t0),
        "means": {c: (None if df[c].dropna().empty else round(float(df[c].mean()), 4)) for c in metric_cols},
        "nan_counts": {c: int(df[c].isna().sum()) for c in metric_cols},
        "metrics_run": wanted,
        "note": "reference = text of gold articles; judge = same model as generator",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nrequests: total={throttle.count} (generation={gen_requests}, ragas judge={throttle.count - gen_requests})")
    print(f"per-question scores: {workdir / 'ragas_scores.csv'}")

    if args.fail_below is not None:
        mean = summary["means"].get("faithfulness")
        nans = summary["nan_counts"].get("faithfulness", 0)
        if mean is None:
            print("GATE FAILED: no faithfulness score could be computed")
            sys.exit(1)
        if nans > len(rows) // 4:
            print(f"GATE FAILED: faithfulness is missing for {nans} of {len(rows)} questions (evaluation unreliable)")
            sys.exit(1)
        if mean < args.fail_below:
            print(f"GATE FAILED: faithfulness {mean:.3f} is below {args.fail_below}")
            sys.exit(1)
        print(f"GATE PASSED: faithfulness {mean:.3f} >= {args.fail_below} ({nans} missing of {len(rows)})")


if __name__ == "__main__":
    main()
