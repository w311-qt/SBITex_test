import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator


class EvalCase(BaseModel):
    id: str
    question: str
    language: str  # "en" | "ru"
    should_have_answer: bool = True
    reference_answer: str = ""
    required_evidence_any: list[str] = []
    forbidden_claims_any: list[str] = []
    tags: list[str] = []
    difficulty: str = "medium"  # "easy" | "medium" | "hard"

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        if v not in ("en", "ru"):
            raise ValueError("language must be 'en' or 'ru'")
        return v

    @field_validator("difficulty")
    @classmethod
    def _diff(cls, v: str) -> str:
        if v not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy/medium/hard")
        return v


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Line {lineno} in {path}: {exc}") from exc
    return cases
