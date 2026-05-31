from __future__ import annotations

import math
import sqlite3
from datetime import datetime as _dt
from pathlib import Path

from .metrics import EvalMetrics


def _ts(raw: str | None) -> str | None:
    """Convert ISO timestamp to SQLite-compatible 'YYYY-MM-DD HH:MM:SS'.
    SQLite strftime() does not handle timezone offsets like +00:00."""
    if not raw:
        return None
    try:
        return _dt.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    timestamp   TEXT,
    model_name  TEXT,
    top_k       INTEGER,
    dry_run     INTEGER,
    winmerge_commit TEXT,
    composite_score      REAL,
    pass_rate            REAL,
    mean_correctness     REAL,
    mean_faithfulness    REAL,
    evidence_recall      REAL,
    hallucination_rate   REAL,
    false_refusal_rate   REAL,
    false_answer_rate    REAL,
    forbidden_claim_rate REAL,
    detected_refusal_rate REAL,
    cohens_kappa_correctness REAL,
    cohens_kappa_faithfulness REAL,
    n_cases     INTEGER
);

CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    case_id     TEXT,
    language    TEXT,
    difficulty  TEXT,
    should_have_answer  INTEGER,
    correctness         REAL,
    faithfulness        REAL,
    relevance           REAL,
    passed              INTEGER,
    hallucination       INTEGER,
    false_refusal       INTEGER,
    false_answer        INTEGER,
    evidence_hit        INTEGER,
    forbidden_claim_hit INTEGER,
    is_refusal          INTEGER,
    secondary_correctness  REAL,
    secondary_faithfulness REAL
);
"""


def _n(v: float | None) -> float | None:
    if v is None:
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _ensure_columns(con: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    """Add any missing columns to an existing table (idempotent migration).

    Guards against an older metrics.db on disk (or a db-seed that created the
    table with an earlier schema) whose column set predates newer metrics.
    CREATE TABLE IF NOT EXISTS never alters an existing table, so we patch here.
    """
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    for name, decl in cols.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def save_run(em: EvalMetrics, db_path: Path, run_id: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_DDL)
        _ensure_columns(con, "runs", {"detected_refusal_rate": "REAL"})
        _ensure_columns(con, "cases", {"is_refusal": "INTEGER"})
        con.execute(
            "INSERT OR REPLACE INTO runs"
            " (run_id, timestamp, model_name, top_k, dry_run, winmerge_commit,"
            " composite_score, pass_rate, mean_correctness, mean_faithfulness,"
            " evidence_recall, hallucination_rate, false_refusal_rate,"
            " false_answer_rate, forbidden_claim_rate, detected_refusal_rate,"
            " cohens_kappa_correctness, cohens_kappa_faithfulness, n_cases)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                _ts(em.metadata.get("timestamp")),
                em.metadata.get("model_name"),
                em.metadata.get("top_k"),
                int(bool(em.metadata.get("dry_run"))),
                em.metadata.get("winmerge_commit"),
                _n(em.composite_score),
                _n(em.pass_rate),
                _n(em.mean_correctness),
                _n(em.mean_faithfulness),
                _n(em.evidence_recall),
                _n(em.hallucination_rate),
                _n(em.false_refusal_rate),
                _n(em.false_answer_rate),
                _n(em.forbidden_claim_rate),
                _n(em.detected_refusal_rate),
                _n(em.cohens_kappa_correctness),
                _n(em.cohens_kappa_faithfulness),
                em.n_cases,
            ),
        )
        con.executemany(
            "INSERT INTO cases"
            " (run_id, case_id, language, difficulty, should_have_answer,"
            " correctness, faithfulness, relevance, passed, hallucination,"
            " false_refusal, false_answer, evidence_hit, forbidden_claim_hit,"
            " is_refusal, secondary_correctness, secondary_faithfulness)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    c.case_id, c.language, c.difficulty,
                    int(c.should_have_answer),
                    _n(c.correctness), _n(c.faithfulness), _n(c.relevance),
                    int(c.passed), int(c.hallucination),
                    int(c.false_refusal), int(c.false_answer),
                    int(c.evidence_hit) if c.evidence_hit is not None else None,
                    int(c.forbidden_claim_hit),
                    int(c.is_refusal),
                    _n(c.secondary_correctness),
                    _n(c.secondary_faithfulness),
                )
                for c in em.per_case
            ],
        )
        con.commit()
    finally:
        con.close()
