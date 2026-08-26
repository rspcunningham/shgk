"""Native Anthropic API clients for translation and benchmark judging.

The rest of the pipeline speaks to models through the OpenAI Agents SDK, which
only fits OpenAI-compatible endpoints. Anthropic models are reached through
their own SDK instead of an OpenAI-compatible shim, so this module implements
the same ``TranslationClient`` and ``BenchmarkScorer`` protocols directly.

Two API details differ from the OpenAI path and drive the shape here:

* Reasoning depth is ``output_config.effort`` (GA), not ``reasoning.effort``.
  Current models take ``thinking={"type": "adaptive"}``; ``budget_tokens`` is
  rejected. Older models (Haiku 4.5) are the reverse: they reject ``effort``
  and need an explicit thinking budget.
* Structured output is ``messages.parse(output_format=Model)``, which validates
  the response against the Pydantic schema and returns ``parsed_output``.
"""

from __future__ import annotations

import asyncio
import json
import time

import anthropic
from pydantic import BaseModel

from .translation import (
    _CRITIC_INSTRUCTIONS,
    _EDITOR_INSTRUCTIONS,
    _TRANSLATOR_INSTRUCTIONS,
    AgentCall,
    EnglishEdit,
    TranslationCandidate,
    TranslationCritique,
    TranslationInput,
    UsageTotals,
    is_transient_error,
)

PROVIDER = "anthropic"
API_KEY_ENV = "ANTHROPIC_API_KEY"

# Models taking output_config.effort + adaptive thinking. Everything else falls
# back to an explicit thinking budget, which is what pre-4.6 models require.
_EFFORT_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    }
)

# Thinking budgets for models without effort support; each must stay below the
# request's max_tokens.
_THINKING_BUDGETS = {
    "low": 2048,
    "medium": 6000,
    "high": 12000,
    "xhigh": 20000,
    "max": 28000,
}

# Thinking tokens share this budget with the answer, so deeper efforts need
# more room; a truncated response fails JSON validation and wastes the call.
_MAX_TOKENS_BY_EFFORT = {
    "none": 32000,
    "low": 32000,
    "medium": 32000,
    "high": 48000,
    "xhigh": 64000,
    "max": 64000,
}


def supports_effort(model: str) -> bool:
    return model in _EFFORT_MODELS


def request_options(model: str, reasoning_effort: str) -> dict[str, object]:
    """Per-model thinking/effort parameters for one request."""

    options: dict[str, object] = {
        "max_tokens": _MAX_TOKENS_BY_EFFORT.get(reasoning_effort, 32000)
    }
    if supports_effort(model):
        if reasoning_effort == "none":
            # Accepted below xhigh; pairing disabled thinking with a high effort
            # is a 400, so send no effort at all here.
            options["thinking"] = {"type": "disabled"}
        else:
            options["thinking"] = {"type": "adaptive"}
            options["output_config"] = {"effort": reasoning_effort}
        return options

    if reasoning_effort != "none":
        budget = _THINKING_BUDGETS[reasoning_effort]
        options["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return options


class _AnthropicCaller:
    """Shared retry/usage plumbing for one Anthropic client."""

    def __init__(self, *, transient_retries: int = 6, min_request_interval: float = 0.0):
        # Deep-effort runs are legitimately slow, but an unbounded request
        # stalls its whole config: the runner appends results per case, so one
        # hung call blocks every case queued behind it.
        self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=240.0)
        self.transient_retries = transient_retries
        self.min_request_interval = min_request_interval
        self._last_request_at = 0.0
        self._request_lock = asyncio.Lock()

    async def call(
        self,
        *,
        model: str,
        reasoning_effort: str,
        system: str,
        prompt: str,
        output_type: type[BaseModel],
    ) -> tuple[BaseModel, UsageTotals]:
        options = request_options(model, reasoning_effort)
        for retry in range(self.transient_retries + 1):
            if self.min_request_interval > 0:
                async with self._request_lock:
                    delay = self.min_request_interval - (
                        time.monotonic() - self._last_request_at
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    self._last_request_at = time.monotonic()
            try:
                # Streaming is mandatory at this max_tokens (the SDK rejects a
                # blocking call that could exceed its 10-minute ceiling), and
                # output_format still yields a validated parsed_output.
                async with self._client.messages.stream(
                    model=model,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=output_type,
                    **options,
                ) as stream:
                    response = await stream.get_final_message()
                break
            except Exception as error:
                if not is_transient_error(error) or retry >= self.transient_retries:
                    raise
                await asyncio.sleep(min(60.0, 5.0 * (2**retry)))
        else:  # pragma: no cover - loop always breaks or raises
            raise AssertionError("unreachable")

        parsed = response.parsed_output
        if parsed is None:
            raise ValueError(f"{model} returned no parsable structured output")
        usage = response.usage
        return parsed, UsageTotals(
            requests=1,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            # Thinking tokens are billed inside output_tokens and are not
            # reported separately, so this stays 0 rather than guessing.
            reasoning_output_tokens=0,
        )


class AnthropicTranslationClient:
    """Writer/critic/editor over the native Anthropic API."""

    def __init__(
        self,
        translator_model: str,
        critic_model: str,
        editor_model: str,
        reasoning_effort: str = "low",
        *,
        transient_retries: int = 6,
        min_request_interval: float = 0.0,
    ):
        self.translator_model = translator_model
        self.critic_model = critic_model
        self.editor_model = editor_model
        self.reasoning_effort = reasoning_effort
        self.provider = PROVIDER
        self._caller = _AnthropicCaller(
            transient_retries=transient_retries,
            min_request_interval=min_request_interval,
        )

    async def _run(self, model: str, system: str, task: dict, output_type) -> AgentCall:
        output, usage = await self._caller.call(
            model=model,
            reasoning_effort=self.reasoning_effort,
            system=system,
            prompt=json.dumps(task, ensure_ascii=False, indent=2),
            output_type=output_type,
        )
        return AgentCall(output=output, usage=usage)

    async def propose(
        self,
        source: TranslationInput,
        *,
        previous: TranslationCandidate | None = None,
        feedback: TranslationCritique | None = None,
    ) -> AgentCall:
        task: dict[str, object] = {"original": source.prompt_dict()}
        if previous is not None and feedback is not None:
            task["previous_candidate"] = previous.model_dump()
            task["critic_feedback"] = feedback.model_dump()
            task["instruction"] = "Revise the candidate in response to the critique."
        else:
            task["instruction"] = "Produce the first English candidate."
        return await self._run(
            self.translator_model, _TRANSLATOR_INSTRUCTIONS, task, TranslationCandidate
        )

    async def critique(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall:
        task = {
            "original": source.prompt_dict(),
            "candidate": candidate.model_dump(),
            "instruction": "Judge this candidate independently under the constitution.",
        }
        return await self._run(
            self.critic_model, _CRITIC_INSTRUCTIONS, task, TranslationCritique
        )

    async def edit(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall:
        task = {
            "original": source.prompt_dict(),
            "critic_approved_candidate": candidate.model_dump(),
            "instruction": "Copy-edit this candidate under the editor contract.",
        }
        return await self._run(
            self.editor_model, _EDITOR_INSTRUCTIONS, task, EnglishEdit
        )


class AnthropicRubricScorer:
    """Benchmark judge over the native Anthropic API.

    Mirrors ``AgentsRubricScorer``: same instructions, same validation, same
    ``ScoringResult`` shape, so a panel can mix providers freely.
    """

    def __init__(
        self,
        rubric,
        *,
        model: str,
        reasoning_effort: str = "low",
        transient_retries: int = 4,
    ):
        from .benchmarking.scoring import build_judge_instructions

        self.rubric = rubric
        self.name = rubric.name
        self.version = rubric.version
        self.provider = PROVIDER
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.instructions = build_judge_instructions(rubric)
        self._caller = _AnthropicCaller(transient_retries=transient_retries)

    async def score(self, item):
        from .benchmarking.models import ScoringResult
        from .benchmarking.scoring import RubricJudgement

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

        judgement, usage = await self._caller.call(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            system=self.instructions,
            prompt=json.dumps(item.model_dump(), ensure_ascii=False, indent=2),
            output_type=RubricJudgement,
        )
        self._validate(judgement)
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

    def _validate(self, judgement) -> None:
        specs = {category.name: category for category in self.rubric.categories}
        scores = judgement.score_map()
        if not scores:
            raise ValueError(
                f"{self.provider}:{self.model} returned no category scores; "
                "the judgement carries no signal"
            )
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
        unknown_failures = set(judgement.hard_failures) - set(self.rubric.hard_failures)
        if unknown_failures:
            raise ValueError(
                f"Judge returned unknown hard failures: {sorted(unknown_failures)}"
            )
