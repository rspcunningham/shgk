"""Structured results the three agents exchange, and the values they carry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

# The English payload of a translation. Named once because the workflow copies,
# compares and validates these fields in several places.
ENGLISH_FIELDS = (
    "question_en",
    "answer_en",
    "explanation_en",
    "acceptance_criteria_en",
    "handout_text_en",
)

# A translation missing any of these is not usable, whatever the agents claim.
REQUIRED_ENGLISH_FIELDS = ("question_en", "answer_en", "explanation_en")


class TranslationCandidate(BaseModel):
    status: Literal["translated", "adapted", "untranslatable"]
    question_en: str = Field(description="English question, or empty if untranslatable")
    answer_en: str = Field(description="English answer, or empty if untranslatable")
    explanation_en: str = Field(
        description="English explanation/comment, or empty if untranslatable"
    )
    acceptance_criteria_en: str = Field(
        description="English accepted-answer criteria; empty if absent"
    )
    handout_text_en: str = Field(description="English textual handout; empty if absent")
    changes_description: str = Field(
        description="Concise account of adaptations, or that none were needed"
    )
    untranslatable_reason: str = Field(
        description="Concrete reason if untranslatable; otherwise empty"
    )


class TranslationCritique(BaseModel):
    decision: Literal["accept", "revise"]
    accepted_status: Literal["translated", "adapted", "untranslatable"]
    summary: str
    issues: list[str]
    revision_instructions: str


class EnglishEdit(BaseModel):
    decision: Literal["unchanged", "edited", "needs_rework"]
    question_en: str
    answer_en: str
    explanation_en: str
    acceptance_criteria_en: str
    handout_text_en: str
    edit_summary: str = Field(
        description="Concise description of copy edits, or why none were needed"
    )
    needs_rework_reason: str = Field(
        description="Why safe copy editing is impossible; otherwise empty"
    )


@dataclass(frozen=True, slots=True)
class TranslationInput:
    question_id: int
    content_hash: str
    question: str
    answer: str
    explanation: str
    acceptance_criteria: str
    handout_text: str
    package_title: str

    def prompt_dict(self) -> dict[str, str]:
        return {
            "question_ru": self.question,
            "answer_ru": self.answer,
            "explanation_ru": self.explanation,
            "acceptance_criteria_ru": self.acceptance_criteria,
            "handout_text_ru": self.handout_text,
            "package_title": self.package_title,
        }


@dataclass(slots=True)
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def add(self, other: UsageTotals) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_write_input_tokens += other.cache_write_input_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens


@dataclass(slots=True)
class AgentCall:
    output: TranslationCandidate | TranslationCritique | EnglishEdit
    usage: UsageTotals


class TranslationClient(Protocol):
    translator_model: str
    critic_model: str
    editor_model: str
    reasoning_effort: str

    async def propose(
        self,
        source: TranslationInput,
        *,
        previous: TranslationCandidate | None = None,
        feedback: TranslationCritique | None = None,
    ) -> AgentCall: ...

    async def critique(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall: ...

    async def edit(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall: ...
