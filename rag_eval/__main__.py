from __future__ import annotations
"""
CLI entry point.

  python -m rag_eval evaluate --dataset data/winmerge_eval.jsonl --report reports/out.json
  python -m rag_eval compare  --baseline reports/run_baseline.json --candidate reports/run_improved.json
  python -m rag_eval calibrate --annotations annotations.json
"""

import io
import json
import logging
import sys
from pathlib import Path

# force UTF-8 output on Windows to avoid charmap errors with non-ASCII
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_eval")


def _load_settings(dry_run: bool | None = None):
    from .config import get_settings
    s = get_settings()
    if dry_run is not None:
        object.__setattr__(s, "dry_run", dry_run)
    return s


@click.group()
def cli():
    pass


# ── evaluate ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dataset",  required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--report",   required=True, type=click.Path(path_type=Path), help="Output aggregate JSON")
@click.option("--top-k",    default=5, show_default=True)
@click.option("--dual-judge/--no-dual-judge", default=True, show_default=True)
@click.option("--dry-run/--no-dry-run", default=None)
@click.option("--epsilon",  default=0.02, show_default=True,
              help="Exit code 1 if composite_score drops below (baseline - epsilon). "
                   "Requires --baseline.")
@click.option("--baseline", default=None, type=click.Path(path_type=Path),
              help="Path to previous run JSON for regression guard.")
@click.option("--thr-correctness",   default=0.7,  show_default=True, help="Pass threshold (tau)")
@click.option("--thr-hallucination", default=0.5,  show_default=True, help="Faithfulness below = hallucination")
@click.option("--thr-false-refusal", default=0.3,  show_default=True, help="Correctness below (pos) = false refusal")
@click.option("--thr-false-answer",  default=0.5,  show_default=True, help="Correctness below (neg) = false answer")
def evaluate(dataset, report, top_k, dual_judge, dry_run, epsilon, baseline,
             thr_correctness, thr_hallucination, thr_false_refusal, thr_false_answer):
    """Run end-to-end RAG evaluation with LLM-as-judge."""
    from .config import get_settings
    from .rag.indexer import build_index
    from .rag.pipeline import make_answer_fn
    from .judge.judge import Judge
    from .judge.agreement import DualJudge
    from .metrics import evaluate_rag_run, compare_runs, SCALAR_METRIC_FIELDS
    from .reporter import save_json, save_jsonl, save_csv, save_comparison_json, log_mlflow
    from .db import save_run

    cfg = get_settings()
    if dry_run is not None:
        cfg = cfg.model_copy(update={"dry_run": dry_run})

    mode = "hybrid (BM25+dense)" if cfg.use_dense else "BM25-only"
    log.info("Building %s index from %s ...", mode, cfg.winmerge_repo_path)
    index = build_index(Path(cfg.winmerge_repo_path), cfg)
    log.info("Index ready: %d chunks", index.size)

    answer_fn = make_answer_fn(cfg, index, top_k=top_k)
    judge = Judge(cfg.model_base_url, cfg.model_api_key, cfg.model_name,
                  timeout=cfg.request_timeout_sec, dry_run=cfg.dry_run,
                  max_tokens=cfg.max_tokens_judge, num_ctx=cfg.num_ctx)

    dj = None
    if dual_judge:
        if not cfg.secondary_model:
            log.warning("SECONDARY_MODEL not set; disabling dual judge")
        else:
            dj = DualJudge(judge, cfg.secondary_base_url, cfg.secondary_api_key,
                           cfg.secondary_model, timeout=cfg.request_timeout_sec,
                           dry_run=cfg.dry_run,
                           max_tokens=cfg.max_tokens_judge, num_ctx=cfg.num_ctx)

    import datetime, subprocess

    def _git_commit(path: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    meta = {
        "model_name": cfg.model_name,
        "dataset_path": str(dataset),
        "top_k": top_k,
        "dry_run": cfg.dry_run,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "winmerge_commit": _git_commit(cfg.winmerge_repo_path),
        "use_dense": cfg.use_dense,
        "retrieval_mode": "hybrid" if cfg.use_dense else "bm25-only",
        "bm25_min_score": cfg.bm25_min_score,
    }

    # Warmup: forces model to load into RAM before timed evaluation begins.
    # On CPU this takes 60-90 s; without warmup the first real case absorbs that overhead.
    if not cfg.dry_run:
        log.info("Warming up model (loading into memory)...")
        try:
            judge.correctness("ping", "ping", "ping", "_warmup")
        except Exception:
            pass
        log.info("Model ready.")

    log.info("Running evaluation (dry_run=%s, top_k=%d, dual_judge=%s)...",
             cfg.dry_run, top_k, dj is not None)

    from .metrics import ThresholdConfig
    thr = ThresholdConfig(
        correctness_pass=thr_correctness,
        hallucination=thr_hallucination,
        false_refusal=thr_false_refusal,
        false_answer=thr_false_answer,
    )
    meta["thresholds"] = {
        "correctness_pass": thr_correctness,
        "hallucination": thr_hallucination,
        "false_refusal": thr_false_refusal,
        "false_answer": thr_false_answer,
    }

    em = evaluate_rag_run(
        dataset, answer_fn,
        judge=judge, dual_judge=dj,
        correctness_threshold=thr_correctness,
        thresholds=thr,
        max_workers=cfg.max_workers,
        metadata=meta,
    )

    log.info("composite=%.3f  pass_rate=%.3f  evidence_recall=%.3f",
             em.composite_score, em.pass_rate, em.evidence_recall)

    save_json(em, report)
    cases_path = report.with_name(report.stem + "_cases.jsonl")
    csv_path   = report.with_suffix(".csv")
    save_jsonl(em.per_case, cases_path)
    save_csv(em, csv_path)
    log.info("Saved: %s  %s  %s", report, cases_path, csv_path)

    # SQLite (Grafana datasource)
    import uuid as _uuid
    run_id = report.stem + "_" + _uuid.uuid4().hex[:6]
    db_path = Path(cfg.metrics_db_path)
    save_run(em, db_path, run_id)
    log.info("Metrics DB: %s  run_id=%s", db_path, run_id)

    # MLflow (optional)
    log_mlflow(em, cfg.mlflow_tracking_uri, run_id,
               artifacts=[report, cases_path, csv_path])

    if baseline and baseline.exists():
        with baseline.open() as f:
            prev = json.load(f)
        prev_score = float(prev.get("composite_score", 0))
        drop = prev_score - em.composite_score
        cmp_path = report.with_name("comparison.json")
        if "composite_score" in prev:
            from .metrics import EvalMetrics
            # build minimal stub for compare_runs
            delta = {f"delta_{k}": getattr(em, k, float("nan")) - prev.get(k, float("nan"))
                     for k in SCALAR_METRIC_FIELDS}
            save_comparison_json(delta, cmp_path)
            log.info("Comparison saved: %s", cmp_path)
        if drop > epsilon:
            log.error("REGRESSION: composite_score dropped %.4f > epsilon=%.4f", drop, epsilon)
            sys.exit(1)

    click.echo(f"\ncomposite_score : {em.composite_score:.4f}")
    click.echo(f"pass_rate       : {em.pass_rate:.4f}")
    click.echo(f"mean_correctness: {em.mean_correctness:.4f}")
    click.echo(f"mean_faithfulness: {em.mean_faithfulness:.4f}")
    click.echo(f"mean_relevance  : {em.mean_relevance:.4f}")
    click.echo(f"evidence_recall : {em.evidence_recall:.4f}")
    click.echo(f"hallucination   : {em.hallucination_rate:.4f}")
    click.echo(f"detected_refusal: {em.detected_refusal_rate:.4f}  (structural; judge-independent)")
    if em.cohens_kappa_correctness is not None:
        click.echo(f"kappa_correctness: {em.cohens_kappa_correctness:.4f}")


# ── compare ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--baseline",  required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--candidate", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", default=None, type=click.Path(path_type=Path))
@click.option("--epsilon", default=0.02, show_default=True)
def compare(baseline, candidate, output, epsilon):
    """Print delta table between two runs; exit 1 if composite_score regressed."""
    from .reporter import save_comparison_json
    from .metrics import SCALAR_METRIC_FIELDS

    with baseline.open() as f:
        b = json.load(f)
    with candidate.open() as f:
        c = json.load(f)

    fields = SCALAR_METRIC_FIELDS

    delta = {}
    click.echo(f"\n{'Metric':<30} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}")
    click.echo("-" * 65)
    for field in fields:
        bv = float(b.get(field, float("nan")))
        cv = float(c.get(field, float("nan")))
        d  = cv - bv
        delta[f"delta_{field}"] = d
        sign = "▲" if d > 0 else ("▼" if d < 0 else "=")
        click.echo(f"{field:<30} {bv:>10.4f} {cv:>10.4f} {sign}{abs(d):>9.4f}")

    if output:
        save_comparison_json(delta, output)
        log.info("Delta saved: %s", output)

    drop = -delta.get("delta_composite_score", 0)
    if drop > epsilon:
        click.echo(f"\nREGRESSION: composite_score dropped {drop:.4f} > epsilon={epsilon:.4f}", err=True)
        sys.exit(1)


# ── calibrate ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--annotations", required=True, type=click.Path(exists=True, path_type=Path),
              help="JSON array: [{case_id, judge_score, human_label}, ...]")
@click.option("--output", default=None, type=click.Path(path_type=Path))
def calibrate(annotations, output):
    """Calibrate correctness threshold tau from manual annotations."""
    from .calibration import calibrate_threshold

    with annotations.open() as f:
        data = json.load(f)

    result = calibrate_threshold(data)
    click.echo(f"Best tau = {result.best_threshold}  F1 = {result.best_f1:.4f}")
    click.echo(f"\n{'τ':>6}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}")
    for t, f, p, r in zip(result.thresholds, result.f1_scores,
                           result.precision_scores, result.recall_scores):
        click.echo(f"{t:>6.2f}  {f:>6.4f}  {p:>6.4f}  {r:>6.4f}")

    if output:
        out = {
            "best_threshold": result.best_threshold,
            "best_f1": result.best_f1,
            "detail": [
                {"threshold": t, "f1": f, "precision": p, "recall": r}
                for t, f, p, r in zip(result.thresholds, result.f1_scores,
                                       result.precision_scores, result.recall_scores)
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2))
        log.info("Calibration saved: %s", output)


if __name__ == "__main__":
    cli()
