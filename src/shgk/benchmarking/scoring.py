from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Protocol

from agents import Agent, AgentOutputSchema, ModelSettings, RunConfig, Runner
from pydantic import BaseModel, Field

from ..providers import ProviderModelFactory, parse_json_payload
from ..translation import is_transient_error
from .models import (
    CategorySpec,
    RubricConfig,
    ScoringInput,
    ScoringResult,
    load_jsonl,
    scoring_input_from_raw,
)


class BenchmarkScorer(Protocol):
    name: str
    version: str

    async def score(self, item: ScoringInput) -> ScoringResult: ...


class CategoryScore(BaseModel):
    """One scored category.

    A list of fixed-key objects instead of an open-ended ``dict[str, float]``:
    dynamic-key objects are expressed as ``additionalProperties`` in JSON Schema,
    which several providers (notably Anthropic through OpenRouter) silently
    return empty rather than populate.
    """

    category: str = Field(description="Rubric category key being scored")
    score: float = Field(description="Numeric score inside the category range")
    rationale: str = Field(description="One concise sentence justifying the score")


class RubricJudgement(BaseModel):
    scores: list[CategoryScore] = Field(
        description="One entry per applicable category; omit categories that cannot apply"
    )
    hard_failures: list[str] = Field(default_factory=list)

    def score_map(self) -> dict[str, float]:
        return {entry.category: entry.score for entry in self.scores}

    def rationale_map(self) -> dict[str, str]:
        return {entry.category: entry.rationale for entry in self.scores}


class DeterministicScorer:
    name = "deterministic"
    version = "1"
    category_specs = [
        CategorySpec(
            name="workflow_complete",
            label="Complete",
            description="The complete agent workflow returned a result.",
            minimum=0,
            maximum=1,
            weight=0,
        ),
        CategorySpec(
            name="output_shape_valid",
            label="Shape",
            description="Required output fields are present for the terminal status.",
            minimum=0,
            maximum=1,
            weight=0,
        ),
        CategorySpec(
            name="status_consistent",
            label="Status",
            description="The terminal status agrees with the populated fields.",
            minimum=0,
            maximum=1,
            weight=0,
        ),
        CategorySpec(
            name="expected_status_match",
            label="Expected",
            description="The terminal status matches an optional labelled expectation.",
            minimum=0,
            maximum=1,
            weight=0,
        ),
    ]

    async def score(self, item: ScoringInput) -> ScoringResult:
        complete = not item.generation_error
        valid_status = item.actual_status in {
            "translated",
            "adapted",
            "untranslatable",
        }
        if item.actual_status == "untranslatable":
            shape_valid = bool(item.untranslatable_reason.strip())
            consistent = shape_valid and not (
                item.translated_question.strip() or item.translated_answer.strip()
            )
        elif item.actual_status in {"translated", "adapted"}:
            shape_valid = all(
                value.strip()
                for value in (
                    item.translated_question,
                    item.translated_answer,
                    item.translated_explanation,
                )
            )
            consistent = shape_valid and not item.untranslatable_reason.strip()
        else:
            shape_valid = False
            consistent = False

        scores = {
            "workflow_complete": float(complete),
            "output_shape_valid": float(shape_valid),
            "status_consistent": float(valid_status and consistent),
        }
        if item.expected_status is not None:
            scores["expected_status_match"] = float(
                item.actual_status == item.expected_status
            )
        failures: list[str] = []
        if not complete:
            failures.append("generation_error")
        if not shape_valid:
            failures.append("invalid_output_shape")
        if not valid_status or not consistent:
            failures.append("invalid_status_shape")
        if (
            item.expected_status is not None
            and item.actual_status != item.expected_status
        ):
            failures.append("unexpected_status")
        return ScoringResult(
            scorer=self.name,
            version=self.version,
            scores=scores,
            rationales={},
            hard_failures=failures,
            category_specs=self.category_specs,
        )


def build_judge_instructions(rubric: RubricConfig) -> str:
    """The judge system prompt. Shared by every provider's scorer."""

    category_text = "\n".join(
        f"- {category.name} ({category.minimum} to {category.maximum}): "
        f"{category.description}"
        for category in rubric.categories
    )
    failure_text = "\n".join(f"- {failure}" for failure in rubric.hard_failures)
    return f"""
You are the independent judge for Russian-to-English What? Where? When? quiz
translations. Compare the Russian source with the final English output. Judge the
result, not the writer's self-description. Use only the category keys and numeric
ranges below. Omit a category only when it genuinely cannot apply, such as most
English-output categories for a correctly untranslatable item. Do not turn a
missing category into zero.

Categories:
{category_text}

Return `scores` as a list with one entry per applicable category, each giving
the category key, its numeric score, and a one-sentence rationale. Never return
an empty list.

Score each category independently: judge only what that category names and
ignore weaknesses that belong to another category, so a single flaw is not
punished repeatedly across the sheet. Follow any anchors given in a category's
description. Return concise rationales for every score. Report only applicable
hard failures from this list:
{failure_text or '- none'}

A hard failure is a concrete integrity violation, not a minor style weakness.
Natural English must sound originally written in English, but do not punish one
natural phrasing merely because another is preferable. For `untranslatable`,
judge whether that feasibility decision and its reason are correct rather than
demanding English text.
""".strip()


class AgentsRubricScorer:
    def __init__(
        self,
        rubric: RubricConfig,
        *,
        provider: str,
        model: str,
        reasoning_effort: str = "low",
    ):
        self.rubric = rubric
        self.name = rubric.name
        self.version = rubric.version
        self.provider = provider
        self.model = model
        factory = ProviderModelFactory(provider)
        factory.require_structured_outputs(model)
        self.native_structured_outputs = True
        schema_instruction = ""
        output_type: type[BaseModel] | AgentOutputSchema | None = AgentOutputSchema(
            RubricJudgement, strict_json_schema=False
        )
        instructions = build_judge_instructions(rubric)

        self.agent = Agent(
            name="ChGK benchmark judge",
            instructions=instructions,
            model=factory.model(model),
            model_settings=ModelSettings(
                max_tokens=1800,
                include_usage=True,
                store=False,
                reasoning={"effort": reasoning_effort},
                extra_body=factory.extra_body(),
                # Stable key so the rubric prefix is cached across cases; the
                # SDK's per-run key would defeat it (see translation.py).
                extra_args=(
                    {"prompt_cache_key": f"shgk-judge-{rubric.name}-v{rubric.version}"}
                    if provider == "openai"
                    else None
                ),
            ),
            output_type=output_type,
        )
        self.run_config = RunConfig(
            tracing_disabled=True,
            workflow_name="ChGK translation benchmark scoring",
        )

    def _validate(self, judgement: RubricJudgement) -> None:
        specs = {category.name: category for category in self.rubric.categories}
        scores = judgement.score_map()
        if not scores:
            raise ValueError(
                f"{self.provider}:{self.model} returned no category scores; "
                "the judgement carries no signal"
            )
        if len(scores) != len(judgement.scores):
            raise ValueError("Judge returned duplicate categories")
        unknown = set(scores) - set(specs)
        if unknown:
            raise ValueError(f"Judge returned unknown categories: {sorted(unknown)}")
        for name, score in scores.items():
            spec = specs[name]
            if not spec.minimum <= score <= spec.maximum:
                raise ValueError(
                    f"Judge score {name}={score} is outside "
                    f"{spec.minimum}..{spec.maximum}"
                )
        unknown_failures = set(judgement.hard_failures) - set(
            self.rubric.hard_failures
        )
        if unknown_failures:
            raise ValueError(
                f"Judge returned unknown hard failures: {sorted(unknown_failures)}"
            )

    async def score(self, item: ScoringInput) -> ScoringResult:
        if item.generation_error:
            return ScoringResult(
                scorer=self.name,
                version=self.version,
                scores={},
                rationales={},
                hard_failures=[],
                category_specs=self.rubric.categories,
                metadata={
                    "provider": self.provider,
                    "model": self.model,
                    "skipped": "generation_error",
                },
            )
        result = await Runner.run(
            self.agent,
            json.dumps(item.model_dump(), ensure_ascii=False, indent=2),
            max_turns=1,
            run_config=self.run_config,
        )
        if self.native_structured_outputs:
            judgement = result.final_output_as(
                RubricJudgement, raise_if_incorrect_type=True
            )
        else:
            judgement = RubricJudgement.model_validate(
                parse_json_payload(result.final_output)
            )
        self._validate(judgement)
        usage = result.context_wrapper.usage
        return ScoringResult(
            scorer=self.name,
            version=self.version,
            scores=judgement.score_map(),
            rationales=judgement.rationale_map(),
            hard_failures=judgement.hard_failures,
            category_specs=self.rubric.categories,
            metadata={
                "provider": self.provider,
                "model": self.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )


class PanelRubricScorer:
    """Aggregate several rubric judges (and repeated passes) by per-category median.

    Cross-family panels reduce self-preference bias; repeated passes reduce
    within-judge variance. A category counts only when at least half of the
    successful judgements report it; hard failures need a majority vote.
    """

    def __init__(
        self,
        rubric: RubricConfig,
        *,
        members: list[BenchmarkScorer],
        passes: int = 1,
        transient_retries: int = 3,
        timeout: float = 240.0,
    ):
        if not members:
            raise ValueError("panel needs at least one member scorer")
        if passes < 1:
            raise ValueError("passes must be at least 1")
        self.rubric = rubric
        self.name = rubric.name
        self.version = rubric.version
        self.members = members
        self.passes = passes
        self.transient_retries = transient_retries
        # A judge call that never returns would otherwise stall the whole run:
        # asyncio.gather waits for every member, and one hung request blocks
        # the case, the file, and everything queued behind it.
        self.timeout = timeout

    async def _score_member(
        self, member: BenchmarkScorer, item: ScoringInput
    ) -> ScoringResult:
        for retry in range(self.transient_retries + 1):
            try:
                return await asyncio.wait_for(member.score(item), timeout=self.timeout)
            except TimeoutError:
                if retry >= self.transient_retries:
                    raise
            except Exception as error:
                if not is_transient_error(error) or retry >= self.transient_retries:
                    raise
            await asyncio.sleep(min(30.0, 2.0 * (2**retry)))
        raise AssertionError("unreachable")

    async def score(self, item: ScoringInput) -> ScoringResult:
        if item.generation_error:
            return ScoringResult(
                scorer=self.name,
                version=self.version,
                scores={},
                rationales={},
                hard_failures=[],
                category_specs=self.rubric.categories,
                metadata={"skipped": "generation_error"},
            )
        outcomes = await asyncio.gather(
            *(
                self._score_member(member, item)
                for member in self.members
                for _ in range(self.passes)
            ),
            return_exceptions=True,
        )
        results = [r for r in outcomes if isinstance(r, ScoringResult)]
        failures = [r for r in outcomes if not isinstance(r, ScoringResult)]
        if len(results) * 2 < len(outcomes):
            raise RuntimeError(
                f"Panel quorum failed: {len(results)}/{len(outcomes)} judgements; "
                f"first error: {failures[0]!r}"
            )
        # Every member must land at least one judgement. Without this, a panel
        # whose second family is down (expired key, no credits) silently
        # collapses into a single-family judge and reintroduces the bias the
        # panel exists to remove.
        heard = {str(r.metadata.get("model")) for r in results}
        silent = [
            getattr(member, "model", None)
            for member in self.members
            if str(getattr(member, "model", None)) not in heard
        ]
        if silent:
            raise RuntimeError(
                f"Panel members returned nothing: {silent}; "
                f"first error: {failures[0]!r}" if failures else
                f"Panel members returned nothing: {silent}"
            )

        by_category: dict[str, list[tuple[float, str, object]]] = defaultdict(list)
        for result in results:
            for name, value in result.scores.items():
                by_category[name].append(
                    (
                        value,
                        result.rationales.get(name, ""),
                        result.metadata.get("model"),
                    )
                )
        scores: dict[str, float] = {}
        rationales: dict[str, str] = {}
        for name, values in by_category.items():
            if len(values) * 2 < len(results):
                continue
            scores[name] = median(value for value, _, _ in values)
            _, rationale, model = min(
                values, key=lambda entry: abs(entry[0] - scores[name])
            )
            rationales[name] = f"[{model}] {rationale}" if rationale else ""
        failure_votes = Counter(
            failure for result in results for failure in set(result.hard_failures)
        )
        hard_failures = sorted(
            failure
            for failure, votes in failure_votes.items()
            if votes * 2 >= len(results)
        )
        return ScoringResult(
            scorer=self.name,
            version=self.version,
            scores=scores,
            rationales=rationales,
            hard_failures=hard_failures,
            category_specs=self.rubric.categories,
            metadata={
                "panel": [
                    {
                        "provider": getattr(member, "provider", None),
                        "model": getattr(member, "model", None),
                    }
                    for member in self.members
                ],
                "passes": self.passes,
                "judgements": len(results),
                "failed_judgements": len(failures),
                "members": [
                    result.model_dump(exclude={"category_specs"}) for result in results
                ],
            },
        )


async def score_translation(
    initial_question: str,
    initial_answer: str,
    translated_question: str,
    translated_answer: str,
    *,
    scorer: BenchmarkScorer,
) -> dict[str, float]:
    """Small public scoring contract with a fully dynamic category dictionary."""

    result = await scorer.score(
        ScoringInput(
            case_id="ad-hoc",
            initial_question=initial_question,
            initial_answer=initial_answer,
            translated_question=translated_question,
            translated_answer=translated_answer,
            actual_status="translated",
        )
    )
    return result.scores


async def score_raw_file(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    scorers: list[BenchmarkScorer],
    overwrite: bool = False,
    resume: bool = True,
    concurrency: int = 1,
    progress=print,
) -> dict[str, int]:
    output_path = Path(output_path)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(raw_path)
    reuse_existing = output_path.exists() and not overwrite
    existing_records = load_jsonl(output_path) if reuse_existing else []
    existing = {str(record.get("case_id")): record for record in existing_records}
    counts = {
        "records": len(records),
        "scored": sum(1 for record in existing_records if not record.get("scoring_errors")),
        "scoring_errors": sum(
            1 for record in existing_records if record.get("scoring_errors")
        ),
    }
    mode = "a" if reuse_existing else "w"
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def score_record(index: int, record: dict[str, object]) -> None:
        case_id = str(record.get("case_id"))
        if case_id in existing:
            if progress:
                progress(f"[{index}/{len(records)}] {case_id}: already scored")
            return
        async with semaphore:
            item = scoring_input_from_raw(record)
            results: list[dict[str, object]] = []
            errors: list[str] = []
            for scorer in scorers:
                try:
                    result = await scorer.score(item)
                    results.append(result.model_dump())
                except Exception as error:
                    errors.append(
                        f"{scorer.name}@{scorer.version}: "
                        f"{type(error).__name__}: {error}"
                    )
            scored_record = {
                **record,
                "scoring": results,
                "scoring_errors": errors,
            }
            async with write_lock:
                stream.write(json.dumps(scored_record, ensure_ascii=False) + "\n")
                stream.flush()
                if errors:
                    counts["scoring_errors"] += 1
                else:
                    counts["scored"] += 1
            if progress:
                progress(
                    f"[{index}/{len(records)}] {item.case_id}: "
                    f"{'scored' if not errors else '; '.join(errors)}"
                )

    with output_path.open(mode, encoding="utf-8") as stream:
        async with asyncio.TaskGroup() as group:
            for index, record in enumerate(records, start=1):
                group.create_task(score_record(index, record))
    return counts
