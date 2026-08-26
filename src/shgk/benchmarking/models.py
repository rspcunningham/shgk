from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..translation import TranslationInput


class BenchmarkCase(BaseModel):
    case_id: str
    split: Literal["train", "heldout"] | None = None
    source: str = "benchmark"
    source_question_id: str = ""
    source_content_hash: str = ""
    question: str
    answer: str
    explanation: str = ""
    acceptance_criteria: str = ""
    handout_text: str = ""
    package_title: str = ""
    expected_status: Literal["translated", "adapted", "untranslatable"] | None = None

    def translation_input(self) -> TranslationInput:
        return TranslationInput(
            source=self.source,
            source_question_id=self.source_question_id or self.case_id,
            source_content_hash=self.source_content_hash,
            question=self.question,
            answer=self.answer,
            explanation=self.explanation,
            acceptance_criteria=self.acceptance_criteria,
            handout_text=self.handout_text,
            package_title=self.package_title,
        )


class CategorySpec(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    description: str
    minimum: float = 0
    maximum: float = 4
    weight: float = 1
    higher_is_better: bool = True


class RubricConfig(BaseModel):
    name: str
    version: str
    categories: list[CategorySpec]
    hard_failures: list[str] = Field(default_factory=list)


class ScoringInput(BaseModel):
    case_id: str
    initial_question: str
    initial_answer: str
    initial_explanation: str = ""
    initial_acceptance_criteria: str = ""
    initial_handout_text: str = ""
    translated_question: str = ""
    translated_answer: str = ""
    translated_explanation: str = ""
    translated_acceptance_criteria: str = ""
    translated_handout_text: str = ""
    expected_status: str | None = None
    actual_status: str = "error"
    untranslatable_reason: str = ""
    editor_status: str = ""
    generation_error: str = ""


class ScoringResult(BaseModel):
    scorer: str
    version: str
    scores: dict[str, float]
    rationales: dict[str, str] = Field(default_factory=dict)
    hard_failures: list[str] = Field(default_factory=list)
    category_specs: list[CategorySpec] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def load_cases(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases = [BenchmarkCase.model_validate(record) for record in load_jsonl(path)]
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate case_id in {path}")
    return cases[:limit] if limit is not None else cases


def load_rubric(path: str | Path) -> RubricConfig:
    with Path(path).open(encoding="utf-8") as stream:
        rubric = RubricConfig.model_validate(json.load(stream))
    names = [category.name for category in rubric.categories]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate rubric category in {path}")
    for category in rubric.categories:
        if category.maximum <= category.minimum:
            raise ValueError(f"Invalid range for rubric category {category.name}")
        if category.weight < 0:
            raise ValueError(f"Negative weight for rubric category {category.name}")
    return rubric


def scoring_input_from_raw(record: dict[str, object]) -> ScoringInput:
    case = BenchmarkCase.model_validate(record["case"])
    translation = record.get("translation") or {}
    if not isinstance(translation, dict):
        translation = {}
    output = translation.get("output") or {}
    workflow = translation.get("workflow") or {}
    if not isinstance(output, dict):
        output = {}
    if not isinstance(workflow, dict):
        workflow = {}
    return ScoringInput(
        case_id=case.case_id,
        initial_question=case.question,
        initial_answer=case.answer,
        initial_explanation=case.explanation,
        initial_acceptance_criteria=case.acceptance_criteria,
        initial_handout_text=case.handout_text,
        translated_question=str(output.get("question_en") or ""),
        translated_answer=str(output.get("answer_en") or ""),
        translated_explanation=str(output.get("explanation_en") or ""),
        translated_acceptance_criteria=str(
            output.get("acceptance_criteria_en") or ""
        ),
        translated_handout_text=str(output.get("handout_text_en") or ""),
        expected_status=case.expected_status,
        actual_status=str(output.get("status") or "error"),
        untranslatable_reason=str(output.get("untranslatable_reason") or ""),
        editor_status=str(workflow.get("editor_status") or ""),
        generation_error=str(record.get("error") or ""),
    )
