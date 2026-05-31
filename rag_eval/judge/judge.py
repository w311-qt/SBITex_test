import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .prompts import (
    CORRECTNESS_SYSTEM, CORRECTNESS_USER,
    FAITHFULNESS_SYSTEM, FAITHFULNESS_USER,
    RELEVANCE_SYSTEM, RELEVANCE_USER,
)

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract JSON from LLM output that may include markdown fences or prose."""
    # Strip markdown fences
    cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first {...} block in the text
    m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to fix common issues: single quotes → double, trailing commas
    fixed = re.sub(r"'([^']*)':", r'"\1":', cleaned)         # single-quoted keys
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)       # single-quoted values
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)              # trailing commas
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    raise json.JSONDecodeError("cannot extract JSON from response", text, 0)


_MOCK_CORRECTNESS = {"score": 0.75, "label": "mostly_correct", "reason": "[DRY-RUN] mock correctness."}
_MOCK_FAITHFULNESS = {"score": 0.80, "label": "mostly_faithful", "reason": "[DRY-RUN] mock faithfulness."}
_MOCK_RELEVANCE = {"score": 0.90, "label": "relevant", "reason": "[DRY-RUN] mock relevance."}


@dataclass
class JudgeResult:
    score: float
    label: str
    reason: str
    raw: dict[str, Any]


def _call_judge(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    timeout: int,
    case_id: str,
    metric: str,
    max_retries: int = 2,
    max_tokens: int = 200,
    num_ctx: int = 4096,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"options": {"num_ctx": num_ctx}},
            )
            text = (resp.choices[0].message.content or "").strip()
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("case=%s metric=%s attempt=%d parse error: %s", case_id, metric, attempt, exc)
            if attempt == max_retries:
                log.error("case=%s metric=%s: all retries exhausted, returning fallback", case_id, metric)
                return {"score": 0.0, "label": "parse_error", "reason": str(exc)}
        except Exception as exc:
            log.error("case=%s metric=%s attempt=%d API error: %s", case_id, metric, attempt, exc)
            # do not retry on client errors (4xx) except 429 rate-limit
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                return {"score": 0.0, "label": "api_error", "reason": str(exc)}
            if attempt == max_retries:
                return {"score": 0.0, "label": "api_error", "reason": str(exc)}
    return {"score": 0.0, "label": "unknown_error", "reason": "unreachable"}


def _parse(raw: dict[str, Any]) -> JudgeResult:
    return JudgeResult(
        score=float(raw.get("score", 0.0)),
        label=str(raw.get("label", "")),
        reason=str(raw.get("reason", "")),
        raw=raw,
    )


class Judge:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
        dry_run: bool = False,
        max_tokens: int = 200,
        num_ctx: int = 4096,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key or "dry-run-placeholder")
        self._model = model
        self._timeout = timeout
        self._dry_run = dry_run
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx

    def _call(self, system: str, user: str, case_id: str, metric: str) -> dict:
        return _call_judge(
            self._client, self._model, system, user,
            self._timeout, case_id, metric,
            max_tokens=self._max_tokens, num_ctx=self._num_ctx,
        )

    def correctness(
        self, question: str, candidate: str, reference: str, case_id: str
    ) -> JudgeResult:
        if self._dry_run:
            return _parse(_MOCK_CORRECTNESS)
        return _parse(self._call(
            CORRECTNESS_SYSTEM,
            CORRECTNESS_USER.format(question=question, reference_answer=reference, candidate_answer=candidate),
            case_id, "correctness",
        ))

    def faithfulness(
        self, question: str, candidate: str, context: str, case_id: str
    ) -> JudgeResult:
        if self._dry_run:
            return _parse(_MOCK_FAITHFULNESS)
        return _parse(self._call(
            FAITHFULNESS_SYSTEM,
            FAITHFULNESS_USER.format(question=question, context=context, candidate_answer=candidate),
            case_id, "faithfulness",
        ))

    def relevance(
        self, question: str, candidate: str, case_id: str
    ) -> JudgeResult | None:
        if self._dry_run:
            return _parse(_MOCK_RELEVANCE)
        return _parse(self._call(
            RELEVANCE_SYSTEM,
            RELEVANCE_USER.format(question=question, candidate_answer=candidate),
            case_id, "relevance",
        ))


def make_judge(
    model: str,
    base_url: str,
    api_key: str = "none",
    *,
    timeout: int = 120,
    dry_run: bool = False,
    max_tokens: int = 200,
    num_ctx: int = 4096,
) -> Judge:
    """Factory matching the spec §7 (judge_model, judge_base_url) string-based contract.

    Bridges the spec's string parameters to our reusable Judge object.
    """
    return Judge(
        base_url=base_url, api_key=api_key, model=model,
        timeout=timeout, dry_run=dry_run, max_tokens=max_tokens, num_ctx=num_ctx,
    )
