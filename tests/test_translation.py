from __future__ import annotations

import asyncio
import time
from typing import Literal

from shgk import db
from shgk.curation import content_hash, rebuild_canonical, rebuild_exclusions
from shgk.translation import (
    CRITIC_MODEL,
    EDITOR_MODEL,
    REASONING_EFFORT,
    TRANSLATOR_MODEL,
    AgentCall,
    AgentsTranslationClient,
    EnglishEdit,
    TranslationCandidate,
    TranslationCritique,
    TranslationInput,
    TranslationPipeline,
    UsageTotals,
    run_translation_workflow,
)
from shgk.translation.pipeline import prune_translations

QUESTION = "Вопрос, достаточно длинный, чтобы пройти проверку на длину."


def _seed(tmp_path, count: int = 1, *, question: str = QUESTION):
    """Build a database holding `count` clean, distinct, canonical questions."""
    path = tmp_path / "shgk.sqlite3"
    db.initialize(path)
    with db.connect(path) as connection:
        connection.execute(
            "INSERT INTO packages (id,title,url,status,first_seen_at,fetched_at) "
            "VALUES (1,'Pack','u','ok','t','t')"
        )
        for index in range(count):
            text = f"{question} {index}"
            connection.execute(
                """INSERT INTO questions (id,package_id,question,answer,explanation,
                     content_hash)
                   VALUES (?,1,?,'Ответ','Объяснение',?)""",
                (index + 1, text, content_hash(text, "Ответ", "Объяснение", "", "")),
            )
        rebuild_exclusions(connection)
        rebuild_canonical(connection)
        connection.commit()
    return path


def _translated_ids(path) -> list[int]:
    with db.connect(path, read_only=True) as connection:
        return [row[0] for row in connection.execute(
            "SELECT question_id FROM translations ORDER BY question_id")]


def _candidate(
    *,
    status: Literal["translated", "adapted", "untranslatable"] = "translated",
    question: str = "Question",
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
    *,
    decision: Literal["accept", "revise"] = "accept",
    status: Literal["translated", "adapted", "untranslatable"] = "translated",
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
    decision: Literal["unchanged", "edited", "needs_rework"] = "unchanged",
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
        question_id=1,
        content_hash="hash",
        question="Вопрос",
        answer="Ответ",
        explanation="Объяснение",
        acceptance_criteria="",
        handout_text="",
        package_title="Пакет",
    )



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
    feedback = client.feedback_seen[1]
    assert feedback is not None
    assert feedback.revision_instructions == "Fix the word count."


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


def test_pipeline_translates_pending_rows_and_resumes(tmp_path) -> None:
    path = _seed(tmp_path, 3)
    client = FakeClient([_candidate()] * 3, [_critique()] * 3)
    pipeline = TranslationPipeline(path)

    first = asyncio.run(pipeline.run(client, limit=2))
    assert first.selected == 2 and first.completed == 2
    assert sorted(first.translated_ids) == _translated_ids(path)

    # Whichever two were drawn, a rerun must pick up only the one left over.
    second = asyncio.run(pipeline.run(client, limit=2))
    assert second.selected == 1
    assert _translated_ids(path) == [1, 2, 3]

    third = asyncio.run(pipeline.run(client, limit=2))
    assert third.selected == 0 and third.completed == 0


def test_changed_canonical_text_drops_the_translation(tmp_path) -> None:
    """A translation lives exactly as long as the record it was made from."""
    path = _seed(tmp_path, 2)
    client = FakeClient([_candidate()] * 2, [_critique()] * 2)
    asyncio.run(TranslationPipeline(path).run(client, limit=10))
    assert _translated_ids(path) == [1, 2]

    with db.connect(path) as connection:
        # A reprint of question 1 arrives carrying an explanation the original
        # lacked; the merged record, and so its content_hash, changes.
        connection.execute(
            "INSERT INTO questions (id,package_id,question,answer,explanation,"
            "content_hash) VALUES (3,1,?,'Ответ','Куда более длинное объяснение.','h3')",
            (QUESTION + " 0",),
        )
        rebuild_exclusions(connection)
        rebuild_canonical(connection)
        assert prune_translations(connection) == 1
        connection.commit()

    assert _translated_ids(path) == [2]
    pending = TranslationPipeline(path)._pending_inputs(limit=10, refresh=False)
    assert [item.question_id for item in pending] == [1]
    assert pending[0].explanation == "Куда более длинное объяснение."


def test_prune_leaves_unchanged_translations_alone(tmp_path) -> None:
    path = _seed(tmp_path, 1)
    asyncio.run(TranslationPipeline(path).run(FakeClient([_candidate()], [_critique()])))
    with db.connect(path) as connection:
        rebuild_exclusions(connection)
        rebuild_canonical(connection)
        assert prune_translations(connection) == 0
        connection.commit()
    assert _translated_ids(path) == [1]


def test_refresh_retranslates_current_rows(tmp_path) -> None:
    path = _seed(tmp_path, 2)
    client = FakeClient([_candidate()] * 4, [_critique()] * 4)
    pipeline = TranslationPipeline(path)
    asyncio.run(pipeline.run(client, limit=10))

    assert asyncio.run(pipeline.run(client, limit=10)).selected == 0
    assert asyncio.run(pipeline.run(client, limit=10, refresh=True)).selected == 2


def test_excluded_and_duplicate_questions_are_never_translated(tmp_path) -> None:
    """Stage 4 draws from questions_canonical, so stages 2 and 3 gate it."""
    path = _seed(tmp_path, 1)
    with db.connect(path) as connection:
        connection.execute(
            "INSERT INTO questions (id,package_id,question,answer,content_hash) "
            "VALUES (2,1,'$1a','Ответ','h2')"
        )
        connection.execute(
            "INSERT INTO questions (id,package_id,question,answer,content_hash) "
            "VALUES (3,1,?,'Ответ','h3')", (QUESTION + " 0",)
        )
        rebuild_exclusions(connection)
        rebuild_canonical(connection)
        connection.commit()

    pending = TranslationPipeline(path)._pending_inputs(limit=10, refresh=False)
    assert [item.question_id for item in pending] == [1]



def test_selection_is_a_random_sample_not_the_lowest_ids(tmp_path) -> None:
    """Ids follow packages, so taking them in order translates oldest-first."""
    path = _seed(tmp_path, 40)
    pipeline = TranslationPipeline(path)
    draws = [
        tuple(item.question_id for item in pipeline._pending_inputs(limit=5, refresh=False))
        for _ in range(8)
    ]
    assert len(set(draws)) > 1, "every draw returned the same questions"
    assert set(draws[0]) != {1, 2, 3, 4, 5}


def test_translating_in_batches_covers_each_question_exactly_once(tmp_path) -> None:
    """Random order must not cause a question to be redrawn or missed."""
    path = _seed(tmp_path, 25)
    pipeline = TranslationPipeline(path)
    client = FakeClient([_candidate()] * 40, [_critique()] * 40)

    drawn: list[int] = []
    while True:
        result = asyncio.run(pipeline.run(client, limit=4))
        if not result.selected:
            break
        drawn.extend(result.translated_ids)

    assert sorted(drawn) == list(range(1, 26))
    assert len(drawn) == len(set(drawn)), "a question was translated twice"
    assert _translated_ids(path) == list(range(1, 26))


def test_a_draw_never_offers_the_same_question_twice(tmp_path) -> None:
    path = _seed(tmp_path, 30)
    ids = [
        item.question_id
        for item in TranslationPipeline(path)._pending_inputs(limit=30, refresh=False)
    ]
    assert len(ids) == len(set(ids)) == 30


def test_refresh_offers_questions_that_are_already_current(tmp_path) -> None:
    path = _seed(tmp_path, 5)
    pipeline = TranslationPipeline(path)
    asyncio.run(pipeline.run(FakeClient([_candidate()] * 5, [_critique()] * 5), limit=5))

    assert pipeline._pending_inputs(limit=10, refresh=False) == []
    assert len(pipeline._pending_inputs(limit=10, refresh=True)) == 5


def test_every_selected_question_is_saved(tmp_path) -> None:
    path = _seed(tmp_path, 6)
    client = FakeClient([_candidate()] * 6, [_critique()] * 6)
    result = asyncio.run(TranslationPipeline(path).run(client, limit=6, concurrency=4))
    assert result.selected == 6 and result.completed == 6 and result.errors == 0
    assert _translated_ids(path) == [1, 2, 3, 4, 5, 6]


def test_client_constructs_its_three_agents(monkeypatch) -> None:
    """The unit tests all use a fake client, so nothing else builds the real one."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-for-any-request")
    client = AgentsTranslationClient()
    assert client.translator.model == TRANSLATOR_MODEL
    assert client.critic.model == CRITIC_MODEL
    assert client.editor.model == EDITOR_MODEL
    assert client.reasoning_effort == REASONING_EFFORT
    for agent in (client.translator, client.critic, client.editor):
        assert (agent.model_settings.max_tokens or 0) > 0
        assert agent.output_type is not None


class SlowClient:
    """A client that takes measurable time and records how much overlapped."""

    translator_model = critic_model = editor_model = "fake"
    reasoning_effort = "low"

    def __init__(self, latency: float = 0.05):
        self.latency = latency
        self.in_flight = 0
        self.peak_in_flight = 0

    async def _call(self, output) -> AgentCall:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(self.latency)
        self.in_flight -= 1
        return AgentCall(output, UsageTotals(1, 10, 5))

    async def propose(self, source, *, previous=None, feedback=None) -> AgentCall:
        return await self._call(_candidate())

    async def critique(self, source, candidate) -> AgentCall:
        return await self._call(_critique())

    async def edit(self, source, candidate) -> AgentCall:
        return await self._call(_edit())


def test_concurrency_bounds_requests_in_flight(tmp_path) -> None:
    """Translation runs on one event loop, so the limit is the only throttle."""
    path = _seed(tmp_path, 12)
    client = SlowClient()
    asyncio.run(TranslationPipeline(path).run(client, limit=12, concurrency=4))
    assert client.peak_in_flight == 4


def test_raising_concurrency_overlaps_more_work(tmp_path) -> None:
    path = _seed(tmp_path, 12)
    client = SlowClient()
    asyncio.run(TranslationPipeline(path).run(client, limit=12, concurrency=12))
    assert client.peak_in_flight == 12


def test_concurrency_of_one_serialises(tmp_path) -> None:
    path = _seed(tmp_path, 4)
    client = SlowClient()
    asyncio.run(TranslationPipeline(path).run(client, limit=4, concurrency=1))
    assert client.peak_in_flight == 1


def test_saving_does_not_stall_the_event_loop(tmp_path) -> None:
    """The write is synchronous SQLite; it must stay far below call latency."""
    path = _seed(tmp_path, 20)
    latency = 0.05
    client = SlowClient(latency=latency)
    started = time.perf_counter()
    asyncio.run(TranslationPipeline(path).run(client, limit=20, concurrency=20))
    elapsed = time.perf_counter() - started
    # Three calls deep, all twenty overlapping: anything beyond a small margin
    # over one round means saving is blocking the loop.
    assert elapsed < latency * 3 * 2


