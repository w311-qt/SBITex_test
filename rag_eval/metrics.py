from __future__ import annotations

import concurrent.futures
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from .dataset import EvalCase, load_dataset
from .judge.judge import Judge
from .judge.agreement import AgreementAggregator, AgreementResult, DualJudge
from .rag.pipeline import RagAnswer

log = logging.getLogger(__name__)

AnswerFn = Callable[[str], RagAnswer]

# Technical symbol pattern: CamelCase identifiers, ::scoped names, file paths.
# Language-agnostic — works for both RU and EN questions.
_TECH_SYM_RE = re.compile(
    r'[A-Z][a-zA-Z]*(?:::[A-Za-z_][a-zA-Z0-9_]*)+|'  # Scope::Method
    r'[A-Z][a-z]+(?:[A-Z][a-z]+){2,}|'                # ThreePart CamelCase
    r'[A-Z]{2,}[a-z]+[A-Z][a-zA-Z]+|'                 # CDiffWrapper, BM25Index
    r'[\w./\\]+\.(?:cpp|h|hpp|cc|c)\b'                 # file.cpp / path/to.h
)


def _tech_symbols(text: str) -> frozenset[str]:
    return frozenset(m.group().lower() for m in _TECH_SYM_RE.finditer(text))


def _is_structural_refusal(question: str, answer: str) -> bool:
    """Detect pure refusals without keyword lists or language-specific examples.

    Two conditions must BOTH hold:
      1. Short answer (< 40 words): a substantive answer describing code
         behaviour typically requires at least a few sentences.
      2. No NEW technical symbols: the answer doesn't introduce class names,
         function names, or file paths beyond those already in the question.

    This combination is language-agnostic (works for RU and EN) and avoids
    hardcoded refusal phrases — it relies purely on structural properties.

    Counter-examples handled correctly:
    - 7-word RU refusal ("method X not in context")  → both conditions met → True
    - 25-word EN refusal ("function Y not present…") → both conditions met → True
    - 61-word EN answer about CDirDoc                → fails condition 1    → False
    - Answer that invents CMergeEditSplitterView      → fails condition 2    → False
    """
    if len(answer.split()) >= 40:
        return False
    q_syms = _tech_symbols(question)
    a_syms = _tech_symbols(answer)
    return len(a_syms - q_syms) == 0


@dataclass(frozen=True)
class ThresholdConfig:
    correctness_pass: float = 0.7   # correctness >= this → passed
    hallucination: float = 0.5      # faithfulness < this (pos cases) → hallucination
    false_refusal: float = 0.3      # correctness < this (pos cases) → false refusal
    # For negative cases with filled reference_answer, low correctness means the
    # candidate *disagrees* with the correct denial → hallucinated a positive answer.
    # Original spec formula was >= 0.5, designed for empty-reference datasets.
    false_answer: float = 0.5       # correctness < this (neg cases) → false answer


@dataclass(frozen=True)
class CaseMetrics:
    """Per-case evaluation result. Immutable; all fields set at construction."""
    case_id: str
    question: str
    language: str
    difficulty: str
    should_have_answer: bool

    answer: str
    evidence_files: list[str]

    correctness: float
    faithfulness: float
    relevance: Optional[float]

    passed: bool
    hallucination: bool
    false_refusal: bool
    false_answer: bool

    # Structural refusal: answer introduces no new technical symbols beyond the question.
    # Detected deterministically (no LLM, no keyword lists) — see _is_structural_refusal().
    # When True: faithfulness is forced to 1.0; hallucination forced to False.
    is_refusal: bool

    # evidence_hit is None when required_evidence_any was not set for this case
    # (excluded from evidence_recall denominator per spec §6)
    evidence_hit: Optional[bool]
    # forbidden_claim_hit: True if any forbidden_claims_any substring found in answer
    # deterministic post-hoc check, no LLM call
    forbidden_claim_hit: bool
    forbidden_claims_any_set: bool  # True when the dataset case had forbidden_claims_any

    judge_correctness_reason: str
    judge_faithfulness_reason: str

    # dual-judge (None when dual judge not used)
    secondary_correctness: Optional[float]
    secondary_faithfulness: Optional[float]
    correctness_score_delta: Optional[float]
    faithfulness_score_delta: Optional[float]
    correctness_label_agree: Optional[bool]
    faithfulness_label_agree: Optional[bool]


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate over a full evaluation run. Immutable."""
    n_cases: int
    pass_rate: float
    mean_correctness: float
    mean_faithfulness: float
    hallucination_rate: float
    false_refusal_rate: float
    false_answer_rate: float
    evidence_recall: float          # denominator = cases where required_evidence_any is set
    forbidden_claim_rate: float     # fraction where forbidden claim found in answer
    composite_score: float          # 0.45·corr + 0.35·faith + 0.20·evidence_recall

    # Diagnostic (non-spec): structural refusal detection independent of judge calibration.
    # Fraction of positive cases where model introduced no new technical symbols.
    # Reveals retrieval failures that the LLM judge obscures by giving high correctness scores.
    detected_refusal_rate: float

    per_case: tuple[CaseMetrics, ...]
    metadata: dict[str, Any]

    # breakdowns by language / difficulty
    by_language: dict[str, dict[str, float]]
    by_difficulty: dict[str, dict[str, float]]

    # dual-judge agreement (None when dual judge not used)
    cohens_kappa_correctness: Optional[float]
    cohens_kappa_faithfulness: Optional[float]
    mean_score_delta_correctness: Optional[float]
    mean_score_delta_faithfulness: Optional[float]


def _safe_mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _aggregate(cases: list[CaseMetrics]) -> dict[str, float]:
    pos = [c for c in cases if c.should_have_answer]
    neg = [c for c in cases if not c.should_have_answer]
    # evidence_recall: only cases where required_evidence_any was set (evidence_hit is not None)
    ev_cases = [c for c in cases if c.evidence_hit is not None]
    fb_cases = [c for c in cases if c.forbidden_claims_any_set]

    mean_corr = _safe_mean([c.correctness for c in cases])
    mean_faith = _safe_mean([c.faithfulness for c in cases])
    ev_recall = _safe_mean([float(c.evidence_hit) for c in ev_cases]) if ev_cases else float("nan")
    composite = (
        0.45 * mean_corr
        + 0.35 * mean_faith
        + 0.20 * (ev_recall if not _is_nan(ev_recall) else 0.0)
    )

    return {
        "pass_rate": _safe_mean([float(c.passed) for c in cases]),
        "mean_correctness": mean_corr,
        "mean_faithfulness": mean_faith,
        "hallucination_rate": _safe_mean([float(c.hallucination) for c in pos]) if pos else float("nan"),
        "false_refusal_rate": _safe_mean([float(c.false_refusal) for c in pos]) if pos else float("nan"),
        "false_answer_rate": _safe_mean([float(c.false_answer) for c in neg]) if neg else float("nan"),
        "evidence_recall": ev_recall,
        "forbidden_claim_rate": _safe_mean([float(c.forbidden_claim_hit) for c in fb_cases]) if fb_cases else float("nan"),
        # Non-spec diagnostic: fraction of positive cases where structural refusal was detected.
        # Measures retrieval failure rate independent of judge calibration.
        "detected_refusal_rate": (
            _safe_mean([float(c.is_refusal) for c in pos]) if pos else float("nan")
        ),
        "composite_score": composite,
    }


def _breakdown(cases: list[CaseMetrics], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[CaseMetrics]] = {}
    for c in cases:
        groups.setdefault(getattr(c, key), []).append(c)
    return {k: _aggregate(v) for k, v in groups.items()}


def _process_case(
    case: EvalCase,
    answer_fn: AnswerFn,
    judge: Judge,
    dual_judge: DualJudge | None,
    thresholds: ThresholdConfig,
) -> tuple[CaseMetrics, AgreementResult | None, AgreementResult | None]:
    rag = answer_fn(case.question)

    # Structural refusal detection — deterministic, no LLM, no keyword lists.
    # Must run before judge calls: if refusal, we skip faithfulness judge entirely.
    is_refusal = _is_structural_refusal(case.question, rag.answer)

    corr_res = judge.correctness(case.question, rag.answer, case.reference_answer, case.id)

    if is_refusal:
        from .judge.judge import JudgeResult
        faith_res = JudgeResult(
            score=1.0, label="refusal",
            reason="structural_refusal: no new technical symbols introduced",
            raw={"score": 1.0, "label": "refusal", "reason": "structural_refusal"},
        )
    else:
        faith_res = judge.faithfulness(case.question, rag.answer, rag.context, case.id)

    rel_res = judge.relevance(case.question, rag.answer, case.id)

    # evidence_hit: None when field not set (excluded from evidence_recall denominator)
    if case.required_evidence_any:
        evidence_hit: Optional[bool] = any(ev in rag.evidence_files for ev in case.required_evidence_any)
    else:
        evidence_hit = None

    # forbidden_claim_hit: deterministic post-hoc check, no LLM needed
    answer_lower = rag.answer.lower()
    forbidden_claim_hit = any(claim.lower() in answer_lower for claim in case.forbidden_claims_any)

    # --- Metric formulas per spec §6 ---
    #
    # pass_rate: correctness >= τ  (spec formula, no overrides)
    passed = corr_res.score >= thresholds.correctness_pass

    # hallucination_rate: faithfulness < 0.5 for positive cases.
    # faith=1.0 is forced for structural refusals → they can never trigger hallucination.
    # Net effect is equivalent to spec formula.
    hallucination = case.should_have_answer and faith_res.score < thresholds.hallucination

    # false_refusal_rate: spec requires BOTH — structural refusal AND correctness < 0.3.
    # Note: a 7B judge often gives high correctness to refusals (known calibration issue),
    # so this rate will be 0 on our data. See detected_refusal_rate for the structural count.
    false_refusal = (
        case.should_have_answer
        and is_refusal
        and corr_res.score < thresholds.false_refusal
    )

    # false_answer_rate: structural detection, independent of judge calibration.
    # A negative case is a false_answer if the model gave a SUBSTANTIVE answer
    # (introduced new technical symbols) instead of refusing/denying.
    # - is_refusal=True  → model said "X not in context" or similar → NOT false_answer
    # - is_refusal=False → model introduced new technical claims about something that
    #   doesn't exist (e.g., "CUDA is in Src/X.cpp") → false_answer
    # This works regardless of the correctness judge's calibration, because refusal
    # detection is purely structural (no LLM required).
    false_answer = not case.should_have_answer and not is_refusal

    # collect dual-judge results before constructing frozen CaseMetrics
    sec_corr = sec_faith = corr_delta = faith_delta = None
    corr_agree = faith_agree = None
    ag_corr = ag_faith = None

    if dual_judge is not None:
        ag_corr = dual_judge.correctness(case.question, rag.answer, case.reference_answer, case.id,
                                         primary_result=corr_res)
        if not is_refusal:
            ag_faith = dual_judge.faithfulness(case.question, rag.answer, rag.context, case.id,
                                               primary_result=faith_res)
        sec_corr = ag_corr.secondary.score
        sec_faith = ag_faith.secondary.score if ag_faith else None
        corr_delta = ag_corr.score_delta
        faith_delta = ag_faith.score_delta if ag_faith else None
        corr_agree = ag_corr.label_agree
        faith_agree = ag_faith.label_agree if ag_faith else None

    cm = CaseMetrics(
        case_id=case.id,
        question=case.question,
        language=case.language,
        difficulty=case.difficulty,
        should_have_answer=case.should_have_answer,
        answer=rag.answer,
        evidence_files=rag.evidence_files,
        correctness=corr_res.score,
        faithfulness=faith_res.score,
        relevance=rel_res.score if rel_res else None,
        passed=passed,
        hallucination=hallucination,
        false_refusal=false_refusal,
        false_answer=false_answer,
        is_refusal=is_refusal,
        evidence_hit=evidence_hit,
        forbidden_claim_hit=forbidden_claim_hit,
        forbidden_claims_any_set=bool(case.forbidden_claims_any),
        judge_correctness_reason=corr_res.reason,
        judge_faithfulness_reason=faith_res.reason,
        secondary_correctness=sec_corr,
        secondary_faithfulness=sec_faith,
        correctness_score_delta=corr_delta,
        faithfulness_score_delta=faith_delta,
        correctness_label_agree=corr_agree,
        faithfulness_label_agree=faith_agree,
    )

    return cm, ag_corr, ag_faith


def evaluate_rag_run(
    dataset_path: Path,
    answer_fn: AnswerFn,
    *,
    judge: Judge,
    dual_judge: DualJudge | None = None,
    correctness_threshold: float = 0.7,
    thresholds: ThresholdConfig | None = None,
    max_workers: int = 1,
    metadata: dict[str, Any] | None = None,
) -> EvalMetrics:
    """
    End-to-end: for each JSONL row call answer_fn, then LLM-judge,
    collect CaseMetrics and aggregate EvalMetrics.

    Signature note: the spec template uses (judge_model, judge_base_url) strings.
    We accept a pre-built Judge object for better testability and reuse.
    A thin wrapper make_judge(model, base_url, api_key) is available in judge.judge.
    """
    thr = thresholds or ThresholdConfig(correctness_pass=correctness_threshold)

    cases = load_dataset(dataset_path)
    case_metrics: list[CaseMetrics] = [None] * len(cases)  # type: ignore[list-item]
    ag_corr_list: list[AgreementResult | None] = [None] * len(cases)
    ag_faith_list: list[AgreementResult | None] = [None] * len(cases)

    def _run(idx_case: tuple[int, EvalCase]):
        idx, case = idx_case
        try:
            cm, ag_c, ag_f = _process_case(case, answer_fn, judge, dual_judge, thr)
            return idx, cm, ag_c, ag_f
        except Exception as exc:
            log.error("case=%s failed: %s", case.id, exc, exc_info=True)
            raise

    bar = tqdm(total=len(cases), unit="case", desc="Evaluating")
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run, (i, c)): i for i, c in enumerate(cases)}
            for fut in concurrent.futures.as_completed(futs):
                idx, cm, ag_c, ag_f = fut.result()
                case_metrics[idx] = cm
                ag_corr_list[idx] = ag_c
                ag_faith_list[idx] = ag_f
                bar.update(1)
    else:
        for i, case in enumerate(cases):
            _, cm, ag_c, ag_f = _run((i, case))
            case_metrics[i] = cm
            ag_corr_list[i] = ag_c
            ag_faith_list[i] = ag_f
            bar.update(1)
    bar.close()

    agg = _aggregate(case_metrics)
    comp = agg["composite_score"]

    # compute dual-judge aggregates before constructing frozen EvalMetrics
    kappa_c = kappa_f = delta_c = delta_f = None
    if dual_judge is not None:
        valid_c = [r for r in ag_corr_list if r is not None]
        valid_f = [r for r in ag_faith_list if r is not None]
        agg_c = AgreementAggregator()
        agg_f = AgreementAggregator()
        for r in valid_c:
            agg_c.add(r)
        for r in valid_f:
            agg_f.add(r)
        kappa_c = agg_c.cohens_kappa()
        kappa_f = agg_f.cohens_kappa()
        delta_c = agg_c.mean_score_delta(valid_c)
        delta_f = agg_f.mean_score_delta(valid_f)

    return EvalMetrics(
        n_cases=len(cases),
        pass_rate=agg["pass_rate"],
        mean_correctness=agg["mean_correctness"],
        mean_faithfulness=agg["mean_faithfulness"],
        hallucination_rate=agg["hallucination_rate"],
        false_refusal_rate=agg["false_refusal_rate"],
        false_answer_rate=agg["false_answer_rate"],
        evidence_recall=agg["evidence_recall"],
        forbidden_claim_rate=agg["forbidden_claim_rate"],
        composite_score=comp,
        detected_refusal_rate=agg["detected_refusal_rate"],
        per_case=tuple(case_metrics),
        metadata=metadata or {},
        by_language=_breakdown(case_metrics, "language"),
        by_difficulty=_breakdown(case_metrics, "difficulty"),
        cohens_kappa_correctness=kappa_c,
        cohens_kappa_faithfulness=kappa_f,
        mean_score_delta_correctness=delta_c,
        mean_score_delta_faithfulness=delta_f,
    )


def _is_nan(v: float) -> bool:
    return v != v  # NaN is the only float not equal to itself


def compare_runs(baseline: EvalMetrics, candidate: EvalMetrics) -> dict[str, float]:
    """Return delta_* for all scalar metrics. Positive = improvement."""
    scalar_fields = [
        "pass_rate", "mean_correctness", "mean_faithfulness",
        "hallucination_rate", "false_refusal_rate", "false_answer_rate",
        "evidence_recall", "forbidden_claim_rate", "composite_score",
        "detected_refusal_rate",
    ]
    return {
        f"delta_{f}": getattr(candidate, f) - getattr(baseline, f)
        for f in scalar_fields
    }
