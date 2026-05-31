"""
Dual-judge agreement: primary (Qwen/local) + secondary (llama3.2:3b/local).
Computes Cohen's kappa on binarised labels and mean absolute score delta.
Primary result is passed in to avoid double-calling the same model.
"""
import logging
from dataclasses import dataclass

from .judge import Judge, JudgeResult

log = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    primary: JudgeResult
    secondary: JudgeResult
    score_delta: float    # |primary.score - secondary.score|
    label_agree: bool     # both binarised at 0.5 agree


def _discretise(score: float, threshold: float = 0.5) -> int:
    return 1 if score >= threshold else 0


class DualJudge:
    def __init__(
        self,
        primary: Judge,
        secondary_base_url: str,
        secondary_api_key: str,
        secondary_model: str,
        timeout: int = 120,
        dry_run: bool = False,
        max_tokens: int = 200,
        num_ctx: int = 4096,
    ) -> None:
        self._primary = primary
        self._secondary = Judge(
            base_url=secondary_base_url,
            api_key=secondary_api_key,
            model=secondary_model,
            timeout=timeout,
            dry_run=dry_run,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
        )

    def correctness(
        self,
        question: str,
        candidate: str,
        reference: str,
        case_id: str,
        *,
        primary_result: JudgeResult | None = None,
    ) -> AgreementResult:
        p = primary_result if primary_result is not None else self._primary.correctness(question, candidate, reference, case_id)
        s = self._secondary.correctness(question, candidate, reference, case_id)
        return AgreementResult(
            primary=p,
            secondary=s,
            score_delta=abs(p.score - s.score),
            label_agree=_discretise(p.score) == _discretise(s.score),
        )

    def faithfulness(
        self,
        question: str,
        candidate: str,
        context: str,
        case_id: str,
        *,
        primary_result: JudgeResult | None = None,
    ) -> AgreementResult:
        p = primary_result if primary_result is not None else self._primary.faithfulness(question, candidate, context, case_id)
        s = self._secondary.faithfulness(question, candidate, context, case_id)
        return AgreementResult(
            primary=p,
            secondary=s,
            score_delta=abs(p.score - s.score),
            label_agree=_discretise(p.score) == _discretise(s.score),
        )


class AgreementAggregator:
    """Collect per-case agreement and compute dataset-level Cohen's kappa."""

    def __init__(self) -> None:
        self._primary_labels: list[int] = []
        self._secondary_labels: list[int] = []

    def add(self, result: AgreementResult) -> None:
        self._primary_labels.append(_discretise(result.primary.score))
        self._secondary_labels.append(_discretise(result.secondary.score))

    def cohens_kappa(self) -> float:
        if len(self._primary_labels) < 2:
            return float("nan")
        # Kappa undefined when all labels are identical (denominator = 0)
        if len(set(self._primary_labels) | set(self._secondary_labels)) < 2:
            return float("nan")
        try:
            from sklearn.metrics import cohen_kappa_score
            return float(cohen_kappa_score(
                self._primary_labels, self._secondary_labels, labels=[0, 1]
            ))
        except Exception as exc:
            log.warning("cohen_kappa_score failed: %s", exc)
            return float("nan")

    def mean_score_delta(self, results: list[AgreementResult]) -> float:
        if not results:
            return float("nan")
        return sum(r.score_delta for r in results) / len(results)
