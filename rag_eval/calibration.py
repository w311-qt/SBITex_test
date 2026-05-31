"""
Threshold τ calibration from a small set of manual annotations.

Usage:
    annotations = [
        {"case_id": "wm_en_01", "judge_score": 0.75, "human_label": 1},  # 1=correct, 0=incorrect
        ...
    ]
    result = calibrate_threshold(annotations)
    print(result.best_threshold, result.best_f1)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationResult:
    best_threshold: float
    best_f1: float
    thresholds: list[float]
    f1_scores: list[float]
    precision_scores: list[float]
    recall_scores: list[float]


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def calibrate_threshold(
    annotations: list[dict],
    candidates: list[float] | None = None,
) -> CalibrationResult:
    """
    annotations: list of dicts with keys:
        judge_score (float)   — primary judge's correctness score
        human_label (int)     — 1 if human considers the answer correct, else 0

    candidates: τ values to try; default 0.3 … 0.95 in steps of 0.05
    """
    if candidates is None:
        candidates = [round(t, 2) for t in (i * 0.05 for i in range(6, 20))]  # 0.30 … 0.95

    scores = [a["judge_score"] for a in annotations]
    labels = [int(a["human_label"]) for a in annotations]

    best_t = candidates[0]
    best_f1 = -1.0
    f1s, precs, recs = [], [], []

    for t in candidates:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        f = _f1(tp, fp, fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(f)
        precs.append(prec)
        recs.append(rec)
        if f > best_f1:
            best_f1 = f
            best_t = t

    return CalibrationResult(
        best_threshold=best_t,
        best_f1=best_f1,
        thresholds=candidates,
        f1_scores=f1s,
        precision_scores=precs,
        recall_scores=recs,
    )
