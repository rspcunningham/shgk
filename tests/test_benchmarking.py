from __future__ import annotations

import asyncio
import json

import pytest

from shgk.benchmarking.models import (
    BenchmarkCase,
    CategorySpec,
    RubricConfig,
    ScoringInput,
    ScoringResult,
    scoring_input_from_raw,
)
from shgk.benchmarking.report import render_report
from shgk.benchmarking.runner import run_benchmark
from shgk.benchmarking.scoring import (
    DeterministicScorer,
    PanelRubricScorer,
    score_raw_file,
)
from shgk.providers import parse_model_spec
from shgk.translation import (
    TranslationCandidate,
    UsageTotals,
    WorkflowResult,
)


def _raw_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "one",
        "provider": "openai",
        "model": "test-model",
        "status": "completed",
        "error": "",
        "case": {
            "case_id": "one",
            "question": "Вопрос",
            "answer": "Ответ",
            "explanation": "Объяснение",
        },
        "translation": {
            "output": {
                "status": "translated",
                "question_en": "Question",
                "answer_en": "Answer",
                "explanation_en": "Explanation",
                "acceptance_criteria_en": "",
                "handout_text_en": "",
                "changes_description": "Ordinary translation.",
                "untranslatable_reason": "",
            },
            "workflow": {"editor_status": "unchanged"},
        },
    }


def test_model_spec_preserves_openrouter_free_suffix() -> None:
    assert parse_model_spec("openrouter:openai/gpt-oss-20b:free") == (
        "openrouter",
        "openai/gpt-oss-20b:free",
    )


def test_deterministic_scoring_and_dynamic_report(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    scored_path = tmp_path / "scored.jsonl"
    report_dir = tmp_path / "report"
    raw_path.write_text(json.dumps(_raw_record()) + "\n", encoding="utf-8")

    result = asyncio.run(
        score_raw_file(
            raw_path,
            scored_path,
            scorers=[DeterministicScorer()],
            progress=None,
        )
    )
    resumed = asyncio.run(
        score_raw_file(
            raw_path,
            scored_path,
            scorers=[DeterministicScorer()],
            resume=True,
            progress=None,
        )
    )
    summary = render_report([scored_path], report_dir)

    assert result == {"records": 1, "scored": 1, "scoring_errors": 0}
    assert resumed == result
    assert len(scored_path.read_text(encoding="utf-8").splitlines()) == 1
    assert scoring_input_from_raw(_raw_record()).translated_answer == "Answer"
    assert [item["name"] for item in summary["categories"]] == [
        "workflow_complete",
        "output_shape_valid",
        "status_consistent",
    ]
    assert (report_dir / "summary.json").is_file()
    assert (report_dir / "summary.csv").is_file()
    assert (report_dir / "summary.md").is_file()


def test_run_benchmark_overlaps_cases(tmp_path, monkeypatch) -> None:
    in_flight = 0
    max_in_flight = 0

    class DummyClient:
        translator_model = "test-model"
        critic_model = "test-model"
        editor_model = "test-model"
        reasoning_effort = "low"

    async def fake_workflow(client, source, *, max_revisions=2):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        candidate = TranslationCandidate(
            status="translated",
            question_en="Question",
            answer_en="Answer",
            explanation_en="Explanation",
            acceptance_criteria_en="",
            handout_text_en="",
            changes_description="Ordinary translation.",
            untranslatable_reason="",
        )
        return WorkflowResult(
            candidate=candidate,
            translation_attempts=1,
            critic_attempts=1,
            editor_attempts=0,
            usage=UsageTotals(),
            history=[],
            pre_editor_candidate=None,
            editor_result=None,
            editor_usage=UsageTotals(),
            editor_status="skipped",
        )

    dummy_kwargs: dict[str, object] = {}

    def capture_client(**kwargs):
        dummy_kwargs.update(kwargs)
        return DummyClient()

    monkeypatch.setattr(
        "shgk.benchmarking.runner.build_translation_client",
        capture_client,
    )
    monkeypatch.setattr(
        "shgk.benchmarking.runner.run_translation_workflow", fake_workflow
    )
    cases = [
        BenchmarkCase(
            case_id=f"case-{index}",
            question="Вопрос",
            answer="Ответ",
            explanation="Объяснение",
        )
        for index in range(5)
    ]
    output = tmp_path / "raw.jsonl"
    result = asyncio.run(
        run_benchmark(
            cases,
            provider="openai",
            model="test-model",
            output=output,
            concurrency=3,
            progress=None,
        )
    )
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert dummy_kwargs.get("transient_retries") == 0
    assert result == {"cases": 5, "completed": 5, "errors": 0}
    assert {record["case_id"] for record in records} == {case.case_id for case in cases}
    assert max_in_flight >= 2
    assert max_in_flight <= 3


def test_run_benchmark_does_not_abort_siblings_on_transient(tmp_path, monkeypatch) -> None:
    class DummyClient:
        translator_model = "test-model"
        critic_model = "test-model"
        editor_model = "test-model"
        reasoning_effort = "low"

    async def fake_workflow(client, source, *, max_revisions=2):
        if source.source_question_id == "case-0":
            raise RuntimeError(
                "ChatCompletion response has no choices "
                "(possible provider error payload): "
                "{'message': 'Upstream error from Nvidia: Service temporarily "
                "overloaded', 'code': 502}"
            )
        candidate = TranslationCandidate(
            status="translated",
            question_en="Question",
            answer_en="Answer",
            explanation_en="Explanation",
            acceptance_criteria_en="",
            handout_text_en="",
            changes_description="Ordinary translation.",
            untranslatable_reason="",
        )
        return WorkflowResult(
            candidate=candidate,
            translation_attempts=1,
            critic_attempts=1,
            editor_attempts=0,
            usage=UsageTotals(),
            history=[],
            pre_editor_candidate=None,
            editor_result=None,
            editor_usage=UsageTotals(),
            editor_status="skipped",
        )

    monkeypatch.setattr(
        "shgk.benchmarking.runner.build_translation_client",
        lambda **kwargs: DummyClient(),
    )
    monkeypatch.setattr(
        "shgk.benchmarking.runner.run_translation_workflow", fake_workflow
    )
    cases = [
        BenchmarkCase(
            case_id=f"case-{index}",
            question="Вопрос",
            answer="Ответ",
            explanation="Объяснение",
        )
        for index in range(3)
    ]
    output = tmp_path / "raw.jsonl"
    result = asyncio.run(
        run_benchmark(
            cases,
            provider="openai",
            model="test-model",
            output=output,
            concurrency=3,
            progress=None,
        )
    )
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert result["cases"] == 3
    assert result["completed"] == 2
    assert result["errors"] == 0
    assert {record["case_id"] for record in records} == {"case-1", "case-2"}


def test_run_benchmark_is_idempotent_across_invocations(tmp_path, monkeypatch) -> None:
    calls = 0

    class DummyClient:
        translator_model = "test-model"
        critic_model = "test-model"
        editor_model = "test-model"
        reasoning_effort = "low"

    async def fake_workflow(client, source, *, max_revisions=2):
        nonlocal calls
        calls += 1
        candidate = TranslationCandidate(
            status="translated",
            question_en="Question",
            answer_en="Answer",
            explanation_en="Explanation",
            acceptance_criteria_en="",
            handout_text_en="",
            changes_description="Ordinary translation.",
            untranslatable_reason="",
        )
        return WorkflowResult(
            candidate=candidate,
            translation_attempts=1,
            critic_attempts=1,
            editor_attempts=0,
            usage=UsageTotals(),
            history=[],
            pre_editor_candidate=None,
            editor_result=None,
            editor_usage=UsageTotals(),
            editor_status="skipped",
        )

    monkeypatch.setattr(
        "shgk.benchmarking.runner.build_translation_client",
        lambda **kwargs: DummyClient(),
    )
    monkeypatch.setattr(
        "shgk.benchmarking.runner.run_translation_workflow", fake_workflow
    )
    cases = [
        BenchmarkCase(
            case_id="case-0",
            question="Вопрос",
            answer="Ответ",
            explanation="Объяснение",
        )
    ]
    output = tmp_path / "raw.jsonl"
    first = asyncio.run(
        run_benchmark(
            cases,
            provider="openai",
            model="test-model",
            output=output,
            progress=None,
        )
    )
    second = asyncio.run(
        run_benchmark(
            cases,
            provider="openai",
            model="test-model",
            output=output,
            progress=None,
        )
    )
    assert first == {"cases": 1, "completed": 1, "errors": 0}
    assert second == first
    assert calls == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_parallel_suite_overlaps_models(tmp_path, monkeypatch) -> None:
    in_flight = 0
    max_in_flight = 0

    async def fake_benchmark(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return {"cases": 0, "completed": 0, "errors": 0}

    async def fake_score(*args, **kwargs):
        return {"records": 0, "scored": 0, "scoring_errors": 0}

    monkeypatch.setattr("shgk.benchmarking.runner.run_benchmark", fake_benchmark)
    monkeypatch.setattr("shgk.benchmarking.runner.score_raw_file", fake_score)
    from shgk.benchmarking.runner import run_parallel_suite

    cases = [
        BenchmarkCase(
            case_id="case-0",
            question="Вопрос",
            answer="Ответ",
            explanation="Объяснение",
        )
    ]
    planned = [
        ("openai", "one", tmp_path / "one.raw.jsonl", tmp_path / "one.scored.jsonl"),
        ("openai", "two", tmp_path / "two.raw.jsonl", tmp_path / "two.scored.jsonl"),
    ]
    results, scored = asyncio.run(
        run_parallel_suite(
            cases,
            planned,
            make_scorers=lambda: [],
            progress=None,
        )
    )
    assert max_in_flight == 2
    assert [item["model"] for item in results] == ["one", "two"]
    assert scored == [tmp_path / "one.scored.jsonl", tmp_path / "two.scored.jsonl"]


class _StubJudge:
    """Rubric judge stub returning canned judgements, one per call."""

    def __init__(self, name, judgements):
        self.name = "translation_rubric"
        self.version = "1"
        self.provider = "stub"
        self.model = name
        self._judgements = list(judgements)
        self.calls = 0

    async def score(self, item):
        payload = self._judgements[self.calls % len(self._judgements)]
        self.calls += 1
        if isinstance(payload, Exception):
            raise payload
        scores, failures = payload
        return ScoringResult(
            scorer=self.name,
            version=self.version,
            scores=scores,
            rationales={name: f"{self.model} says {name}" for name in scores},
            hard_failures=failures,
            category_specs=[],
            metadata={"model": self.model},
        )


def _panel_rubric():
    return RubricConfig(
        name="translation_rubric",
        version="1",
        categories=[
            CategorySpec(
                name="clue_preservation",
                label="Clues",
                description="clues",
                minimum=0,
                maximum=4,
                weight=1.0,
            )
        ],
        hard_failures=["new_clue_added", "essential_clue_removed"],
    )


def _panel_item():
    return ScoringInput(
        case_id="case-1",
        initial_question="Вопрос",
        initial_answer="Ответ",
        translated_question="Question",
        translated_answer="Answer",
        actual_status="translated",
    )


def test_panel_takes_per_category_median_across_judges_and_passes() -> None:
    sol = _StubJudge("sol", [({"clue_preservation": 4.0}, []), ({"clue_preservation": 3.0}, [])])
    sonnet = _StubJudge("sonnet", [({"clue_preservation": 2.0}, []), ({"clue_preservation": 1.0}, [])])
    panel = PanelRubricScorer(_panel_rubric(), members=[sol, sonnet], passes=2)

    result = asyncio.run(panel.score(_panel_item()))

    # four judgements: 4, 3, 2, 1 -> median 2.5
    assert result.scores["clue_preservation"] == 2.5
    assert result.metadata["judgements"] == 4
    assert sol.calls == 2 and sonnet.calls == 2
    assert len(result.metadata["members"]) == 4


def test_panel_requires_majority_vote_for_hard_failures() -> None:
    sol = _StubJudge("sol", [({"clue_preservation": 3.0}, ["new_clue_added"])])
    sonnet = _StubJudge("sonnet", [({"clue_preservation": 3.0}, [])])
    panel = PanelRubricScorer(_panel_rubric(), members=[sol, sonnet], passes=1)

    split = asyncio.run(panel.score(_panel_item()))
    assert split.hard_failures == ["new_clue_added"]  # 2 of 2 needed... 1 of 2 is a tie

    agreeing = PanelRubricScorer(
        _panel_rubric(),
        members=[
            _StubJudge("sol", [({"clue_preservation": 3.0}, ["new_clue_added"])]),
            _StubJudge("sonnet", [({"clue_preservation": 3.0}, ["new_clue_added"])]),
            _StubJudge("third", [({"clue_preservation": 3.0}, [])]),
        ],
        passes=1,
    )
    voted = asyncio.run(agreeing.score(_panel_item()))
    assert voted.hard_failures == ["new_clue_added"]


def test_panel_tolerates_one_flaky_pass_while_the_member_still_answers() -> None:
    """A single failed pass is survivable as long as that judge still reports."""
    good = _StubJudge("sol", [({"clue_preservation": 3.0}, [])])
    flaky = _StubJudge(
        "sonnet", [ValueError("judge exploded"), ({"clue_preservation": 1.0}, [])]
    )
    panel = PanelRubricScorer(
        _panel_rubric(), members=[good, flaky], passes=2, transient_retries=0
    )

    result = asyncio.run(panel.score(_panel_item()))
    assert result.metadata["failed_judgements"] == 1
    assert result.metadata["judgements"] == 3
    assert result.scores["clue_preservation"] == 3.0  # median of 3.0, 3.0, 1.0


def test_panel_raises_when_quorum_is_lost() -> None:
    broken_a = _StubJudge("sol", [ValueError("boom")])
    broken_b = _StubJudge("sonnet", [ValueError("boom")])
    panel = PanelRubricScorer(
        _panel_rubric(), members=[broken_a, broken_b], passes=1, transient_retries=0
    )

    with pytest.raises(RuntimeError, match="quorum"):
        asyncio.run(panel.score(_panel_item()))


def test_panel_raises_when_a_member_never_answers() -> None:
    """A dead provider must fail loudly instead of silently halving the panel."""
    working = _StubJudge("sol", [({"clue_preservation": 3.0}, [])])
    dead = _StubJudge("sonnet", [RuntimeError("Error code: 402 - Insufficient credits")])
    panel = PanelRubricScorer(
        _panel_rubric(), members=[working, dead], passes=2, transient_retries=0
    )

    with pytest.raises(RuntimeError, match="returned nothing"):
        asyncio.run(panel.score(_panel_item()))


def test_panel_times_out_a_hung_judge_instead_of_stalling() -> None:
    """One never-returning judge must not block the whole case."""

    class _HangingJudge:
        name, version, provider, model = "translation_rubric", "1", "stub", "hangs"

        async def score(self, item):
            await asyncio.sleep(30)

    good = _StubJudge("sol", [({"clue_preservation": 3.0}, [])])
    panel = PanelRubricScorer(
        _panel_rubric(),
        members=[good, _HangingJudge()],
        passes=1,
        transient_retries=0,
        timeout=0.05,
    )

    with pytest.raises(RuntimeError, match="returned nothing"):
        asyncio.run(panel.score(_panel_item()))
