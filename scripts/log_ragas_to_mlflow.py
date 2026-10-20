"""Log the RAGAS results into MLflow, on the run that the Registry's `production` alias points to.

Run from the repo root (it uses sqlite:///mlflow.db, like experiment.py):
  uv run python scripts/log_ragas_to_mlflow.py
  uv run python scripts/log_ragas_to_mlflow.py --run-id <run id>      # override the run
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL = "arabic-legal-rag-retrieval-config"
ALIAS = "production"
# summary key -> MLflow metric name
METRIC_NAMES = {
    "faithfulness": "ragas_faithfulness",
    "answer_relevancy": "ragas_answer_relevancy",
    "llm_context_precision_with_reference": "ragas_context_precision",
    "context_recall": "ragas_context_recall",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "reports" / "ragas_summary.json"))
    ap.add_argument("--scores", default=str(Path.home() / "scratch" / "arabic-legal-rag" / "ragas" / "ragas_scores.csv"))
    ap.add_argument("--run-id", default=None, help="default: the run behind the Registry alias")
    args = ap.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    run_id = args.run_id
    if run_id is None:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
        run_id = mv.run_id
        print(f"registry alias '{ALIAS}' -> version {mv.version} -> run {run_id}")

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metrics = {}
    for key, name in METRIC_NAMES.items():
        value = summary["means"].get(key)
        if value is not None:
            metrics[name] = float(value)
        metrics[f"{name}_missing"] = float(summary["nan_counts"].get(key, 0))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "ragas_questions": summary["questions_scored"],
                "ragas_split": summary["split"],
                "ragas_k_articles": summary["k_articles"],
                "ragas_judge_model": summary["judge_and_generator_model"],
            }
        )
        mlflow.set_tags(
            {
                "ragas.note": "separate evaluation run: 20 dev questions, judge = same model as generator, "
                "reference = gold article text; abstentions score 0 (see README)",
                "ragas.timestamp": summary["timestamp"],
            }
        )
        mlflow.log_artifact(args.summary, artifact_path="ragas")
        if Path(args.scores).exists():
            mlflow.log_artifact(args.scores, artifact_path="ragas")
        else:
            print(f"scores csv not found, skipped: {args.scores}")

    run = client.get_run(run_id)
    print("logged. ragas metrics now on the run:")
    for k, v in sorted(run.data.metrics.items()):
        if k.startswith("ragas_"):
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
