from __future__ import annotations

import asyncio
import sqlite3

from shgk.database import QuestionDatabase
from shgk.models import QuestionRecord
from shgk.pipeline import BasicFilterPipeline
from shgk.translation import (
    AgentCall,
    EnglishEdit,
    TranslationCandidate,
    TranslationCritique,
    TranslationInput,
    TranslationPipeline,
    UsageTotals,
    is_transient_error,
    run_translation_workflow,
)


def _record(identifier: str, *, explanation: str = "Объяснение") -> QuestionRecord:
    return QuestionRecord(
        source="test",
        source_question_id=identifier,
        source_url=f"https://example.test/{identifier}",
        game_kind="sport_chgk",
        question="Вопрос",
        answer="Ответ",
        explanation=explanation,
        fetched_at="2025-01-01T00:00:00+00:00",
    )


def _candidate(
    *, status: str = "translated", question: str = "Question"
) -> TranslationCandidate:
    return TranslationCandidate(
        status=status,
        question_en=question,
        answer_en="Answer",
        explanation_en="Explanation",
        acceptance_criteria_en="",
        handout_text_en="",
        changes_description="No adaptation beyond ordinary translation.",
        untranslatable_reason="",
    )


def _critique(
    *, decision: str = "accept", status: str = "translated"
) -> TranslationCritique:
    return TranslationCritique(
        decision=decision,
        accepted_status=status,
        summary="The clue path is preserved.",
        issues=[] if decision == "accept" else ["The word count is wrong."],
        revision_instructions="" if decision == "accept" else "Fix the word count.",
    )


def _edit(
    *,
    decision: str = "unchanged",
    question: str = "Question",
    reason: str = "",
) -> EnglishEdit:
    return EnglishEdit(
        decision=decision,
        question_en=question,
        answer_en="Answer",
        explanation_en="Explanation",
        acceptance_criteria_en="",
        handout_text_en="",
        edit_summary="Polished the wording."
        if decision == "edited"
        else "No edit needed.",
        needs_rework_reason=reason,
    )


class FakeClient:
    translator_model = "fake-translator"
    critic_model = "fake-critic"
    editor_model = "fake-editor"
    reasoning_effort = "low"

    def __init__(
        self,
        candidates: list[TranslationCandidate],
        critiques: list[TranslationCritique],
        edits: list[EnglishEdit] | None = None,
    ):
        self.candidates = iter(candidates)
        self.critiques = iter(critiques)
        self.edits = iter(edits or [])
        self.feedback_seen: list[TranslationCritique | None] = []
        self.editor_inputs: list[TranslationCandidate] = []

    async def propose(self, source, *, previous=None, feedback=None) -> AgentCall:
        self.feedback_seen.append(feedback)
        return AgentCall(next(self.candidates), UsageTotals(1, 100, 40))

    async def critique(self, source, candidate) -> AgentCall:
        return AgentCall(next(self.critiques), UsageTotals(1, 80, 20))

    async def edit(self, source, candidate) -> AgentCall:
        self.editor_inputs.append(candidate)
        try:
            edit = next(self.edits)
        except StopIteration:
            fields = {
                name: getattr(candidate, name)
                for name in (
                    "question_en",
                    "answer_en",
                    "explanation_en",
                    "acceptance_criteria_en",
                    "handout_text_en",
                )
            }
            edit = EnglishEdit(
                decision="unchanged",
                **fields,
                edit_summary="No edit needed.",
                needs_rework_reason="",
            )
        return AgentCall(edit, UsageTotals(1, 60, 10))


def _input() -> TranslationInput:
    return TranslationInput(
        source="test",
        source_question_id="one",
        source_content_hash="hash",
        question="Вопрос",
        answer="Ответ",
        explanation="Объяснение",
        acceptance_criteria="",
        handout_text="",
        package_title="Пакет",
    )


def test_nvidia_overload_payload_is_transient() -> None:
    assert is_transient_error(
        "ChatCompletion response has no choices (possible provider error payload): "
        "{'message': 'Upstream error from Nvidia: Service temporarily overloaded', "
        "'code': 502}"
    )
    assert is_transient_error(
        "APIStatusError: Error code: 402 - This request requires more credits"
    )
    assert not is_transient_error("ValueError: model did not return a JSON object")
    assert not is_transient_error("ValidationError: 1 validation error for TranslationCandidate")


def test_workflow_passes_critic_feedback_to_one_revision() -> None:
    client = FakeClient(
        [_candidate(question="Bad question"), _candidate(status="adapted")],
        [_critique(decision="revise"), _critique(status="adapted")],
    )

    result = asyncio.run(run_translation_workflow(client, _input(), max_revisions=2))

    assert result.candidate.status == "adapted"
    assert result.translation_attempts == 2
    assert result.critic_attempts == 2
    assert result.usage == UsageTotals(requests=5, input_tokens=420, output_tokens=130)
    assert client.feedback_seen[0] is None
    assert client.feedback_seen[1].revision_instructions == "Fix the word count."


def test_workflow_becomes_untranslatable_when_revision_limit_is_exhausted() -> None:
    client = FakeClient(
        [_candidate(), _candidate()],
        [
            _critique(decision="revise", status="untranslatable"),
            _critique(decision="revise", status="untranslatable"),
        ],
    )

    result = asyncio.run(run_translation_workflow(client, _input(), max_revisions=1))

    assert result.candidate.status == "untranslatable"
    assert result.candidate.question_en == ""
    assert "revision limit" in result.candidate.changes_description
    assert len(result.history) == 2
    assert result.editor_status == "skipped"


def test_workflow_keeps_playable_candidate_when_only_polish_hits_limit() -> None:
    client = FakeClient(
        [_candidate(), _candidate(question="Polished question")],
        [
            _critique(decision="revise", status="adapted"),
            _critique(decision="revise", status="adapted"),
        ],
    )

    result = asyncio.run(run_translation_workflow(client, _input(), max_revisions=1))

    assert result.candidate.status == "adapted"
    assert result.candidate.question_en == "Polished question"
    assert result.translation_attempts == 2


def test_workflow_applies_final_english_edit_and_keeps_pre_edit_candidate() -> None:
    client = FakeClient(
        [_candidate(question="The population grows only through unnatural means.")],
        [_critique(status="adapted")],
        [
            _edit(
                decision="edited",
                question="Where can the citizenry grow only through administrative means?",
            )
        ],
    )

    result = asyncio.run(run_translation_workflow(client, _input()))

    assert result.candidate.question_en.startswith("Where can the citizenry")
    assert result.pre_editor_candidate is not None
    assert result.pre_editor_candidate.question_en.startswith("The population")
    assert result.editor_status == "edited"
    assert result.editor_attempts == 1
    assert result.history[-1]["editor"]["decision"] == "edited"


def test_workflow_keeps_candidate_when_editor_flags_needs_rework() -> None:
    original = _candidate(question="Awkward but playable question")
    client = FakeClient(
        [original],
        [_critique(status="adapted")],
        [
            _edit(
                decision="needs_rework",
                question="Changed question that must be ignored",
                reason="Natural wording would reveal the clue.",
            )
        ],
    )

    result = asyncio.run(run_translation_workflow(client, _input()))

    assert result.candidate.question_en == original.question_en
    assert result.editor_status == "needs_rework"
    assert result.editor_result is not None
    assert result.editor_result.question_en == original.question_en


def test_pipeline_translates_only_current_eligible_rows_and_resumes(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert(
        [_record("eligible"), _record("ineligible", explanation="")]
    )
    BasicFilterPipeline(source_path, pipeline_path).run()
    pipeline = TranslationPipeline(source_path, pipeline_path)
    client = FakeClient([_candidate()], [_critique()])

    first = asyncio.run(pipeline.run(client, limit=10, max_revisions=2))
    second = asyncio.run(pipeline.run(client, limit=10, max_revisions=2))

    assert first == {"selected": 1, "completed": 1, "errors": 0}
    assert second == {"selected": 0, "completed": 0, "errors": 0}
    with sqlite3.connect(pipeline_path) as connection:
        row = connection.execute(
            """
            SELECT source_question_id, status, question_en, answer_en,
                   reasoning_effort, editor_model, editor_status, editor_attempts,
                   api_requests, input_tokens, output_tokens
            FROM translations
            """
        ).fetchone()
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    assert row == (
        "eligible",
        "translated",
        "Question",
        "Answer",
        "low",
        "fake-editor",
        "unchanged",
        1,
        3,
        240,
        70,
    )
    assert tables == [("basic_filter_results",), ("translations",)]


def test_pipeline_offset_creates_a_deterministic_holdout_slice(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert([_record("a"), _record("b"), _record("c")])
    BasicFilterPipeline(source_path, pipeline_path).run()
    pipeline = TranslationPipeline(source_path, pipeline_path)
    client = FakeClient([_candidate()], [_critique()])

    result = asyncio.run(pipeline.run(client, limit=1, offset=1))

    assert result == {"selected": 1, "completed": 1, "errors": 0}
    with sqlite3.connect(pipeline_path) as connection:
        identifier = connection.execute(
            "SELECT source_question_id FROM translations"
        ).fetchone()[0]
    assert identifier == "b"


def test_pipeline_sample_is_reproducible_and_no_commit_is_read_only(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert([_record(str(index)) for index in range(10)])
    BasicFilterPipeline(source_path, pipeline_path).run()

    async def select_once():
        identifiers: list[str] = []
        client = FakeClient([_candidate() for _ in range(3)], [_critique() for _ in range(3)])
        result = await TranslationPipeline(source_path, pipeline_path).run(
            client,
            sample_size=3,
            seed=42,
            commit=False,
            on_result=lambda source, _: identifiers.append(source.source_question_id),
        )
        return result, identifiers

    first, first_ids = asyncio.run(select_once())
    second, second_ids = asyncio.run(select_once())

    assert first == second == {"selected": 3, "completed": 3, "errors": 0}
    assert first_ids == second_ids
    with sqlite3.connect(pipeline_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    assert tables == [("basic_filter_results",)]


class ConcurrencyProbeClient:
    """Fresh outputs per call, recording how many workflows overlap."""

    translator_model = "fake-translator"
    critic_model = "fake-critic"
    editor_model = "fake-editor"
    reasoning_effort = "low"

    def __init__(self):
        self.active = 0
        self.peak = 0

    async def _tick(self):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1

    async def propose(self, source, *, previous=None, feedback=None) -> AgentCall:
        await self._tick()
        return AgentCall(_candidate(), UsageTotals(1, 100, 40))

    async def critique(self, source, candidate) -> AgentCall:
        await self._tick()
        return AgentCall(_critique(), UsageTotals(1, 80, 20))

    async def edit(self, source, candidate) -> AgentCall:
        await self._tick()
        return AgentCall(
            _edit(question=candidate.question_en), UsageTotals(1, 60, 10)
        )


def test_pipeline_workers_translate_concurrently_and_commit_all(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert([_record(str(index)) for index in range(6)])
    BasicFilterPipeline(source_path, pipeline_path).run()
    pipeline = TranslationPipeline(source_path, pipeline_path)
    client = ConcurrencyProbeClient()

    result = asyncio.run(pipeline.run(client, limit=10, workers=4))

    assert result == {"selected": 6, "completed": 6, "errors": 0}
    assert client.peak >= 3
    with sqlite3.connect(pipeline_path) as connection:
        committed = connection.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    assert committed == 6


def test_committed_sample_rerun_does_not_advance_to_a_different_sample(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert([_record(str(index)) for index in range(10)])
    BasicFilterPipeline(source_path, pipeline_path).run()
    pipeline = TranslationPipeline(source_path, pipeline_path)
    client = FakeClient([_candidate() for _ in range(3)], [_critique() for _ in range(3)])

    first = asyncio.run(pipeline.run(client, sample_size=3, seed=42))
    second = asyncio.run(pipeline.run(client, sample_size=3, seed=42))

    assert first == {"selected": 3, "completed": 3, "errors": 0}
    assert second == {"selected": 0, "completed": 0, "errors": 0}
