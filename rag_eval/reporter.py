from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .metrics import EvalMetrics, CaseMetrics


def _sanitize(obj):
    """Replace NaN/Inf floats with None for valid JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _scalar_summary(em: EvalMetrics) -> dict:
    d = asdict(em)
    d.pop("per_case", None)
    return _sanitize(d)


def _cases_list(em: EvalMetrics) -> list[CaseMetrics]:
    return list(em.per_case)


def save_json(em: EvalMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_scalar_summary(em), f, ensure_ascii=False, indent=2, default=str)


def save_jsonl(cases: list[CaseMetrics] | tuple[CaseMetrics, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(_sanitize(asdict(c)), ensure_ascii=False, default=str) + "\n")


def save_csv(em: EvalMetrics, path: Path) -> None:
    """Per-case CSV compatible with Grafana CSV datasource."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in list(em.per_case):
        rows.append({
            "case_id": c.case_id,
            "language": c.language,
            "difficulty": c.difficulty,
            "should_have_answer": int(c.should_have_answer),
            "correctness": c.correctness,
            "faithfulness": c.faithfulness,
            "relevance": c.relevance,
            "passed": int(c.passed),
            "hallucination": int(c.hallucination),
            "false_refusal": int(c.false_refusal),
            "false_answer": int(c.false_answer),
            "evidence_hit": int(c.evidence_hit) if c.evidence_hit is not None else None,
            "is_refusal": int(c.is_refusal),
            "secondary_correctness": c.secondary_correctness,
            "secondary_faithfulness": c.secondary_faithfulness,
            "correctness_score_delta": c.correctness_score_delta,
            "faithfulness_score_delta": c.faithfulness_score_delta,
            "correctness_label_agree": c.correctness_label_agree,
            "faithfulness_label_agree": c.faithfulness_label_agree,
            "forbidden_claim_hit": int(c.forbidden_claim_hit),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def log_mlflow(em: EvalMetrics, tracking_uri: str, run_name: str, artifacts: list[Path]) -> None:
    """Log EvalMetrics to MLflow. No-op if mlflow is not installed or URI is empty."""
    if not tracking_uri:
        return
    try:
        import math
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("rag-eval")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_name": em.metadata.get("model_name", ""),
                "top_k": em.metadata.get("top_k", ""),
                "winmerge_commit": str(em.metadata.get("winmerge_commit", ""))[:8],
                "dry_run": str(em.metadata.get("dry_run", False)),
            })
            metrics = {}
            for k in ["composite_score", "pass_rate", "mean_correctness",
                      "mean_faithfulness", "evidence_recall", "hallucination_rate",
                      "false_refusal_rate", "false_answer_rate", "forbidden_claim_rate"]:
                v = getattr(em, k, None)
                if isinstance(v, float) and not math.isnan(v):
                    metrics[k] = v
            if em.cohens_kappa_correctness is not None and not math.isnan(em.cohens_kappa_correctness):
                metrics["cohens_kappa_correctness"] = em.cohens_kappa_correctness
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))
    except ImportError:
        pass
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("MLflow logging failed: %s", exc)


def save_comparison_json(delta: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(delta), f, ensure_ascii=False, indent=2, default=str)
