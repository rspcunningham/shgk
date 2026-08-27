from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_client
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, Field

from .db import DEFAULT_PATH as DEFAULT_DB_PATH, connect
from .translation_policy import TRANSLATION_CONSTITUTION, TRANSLATION_POLICY_VERSION

# Chosen from the 36-config benchmark matrix (benchmarks/results/matrix):
# luna at max effort matches the best hard-failure rate in the matrix (3 in 40,
# tied with sol@low and terra@max) at $0.019/question versus sol@low's $0.081 —
# roughly $5.6k rather than $23.6k to translate the eligible corpus. Its score
# gap to the leaders sits inside the measured noise band.
# Not defaults: these are the pipeline. Changing one means rerunning the stage.
TRANSLATOR_MODEL = "gpt-5.6-luna"
CRITIC_MODEL = "gpt-5.6-luna"
EDITOR_MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
TRANSLATION_WORKFLOW_VERSION = 13

_TRANSIENT_API_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    TimeoutError,
)


def is_transient_error(error: BaseException | str) -> bool:
    if isinstance(error, _TRANSIENT_API_ERRORS):
        return True
    text = str(error)
    return any(
        marker in text
        for marker in (
            "RateLimitError:",
            "APIConnectionError:",
            "APITimeoutError:",
            "InternalServerError:",
            "Error code: 408",
            "Error code: 429",
            "Error code: 500",
            "Error code: 502",
            "Error code: 503",
            "Error code: 504",
            "'code': 429",
            "'code': 500",
            "'code': 502",
            "'code': 503",
            "'code': 504",
            "Error code: 402",
            "'code': 402",
            "requires more credits",
            "Upstream idle timeout exceeded",
            "temporarily overloaded",
            "Service temporarily overloaded",
            # Anthropic 529 shape: {"type": "overloaded_error", ...}
            "overloaded_error",
            "'message': 'Overloaded'",
            "Error code: 529",
            "no choices (possible provider error payload)",
        )
    )

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


_TRANSLATOR_INSTRUCTIONS = f"""
You are the writer responsible for adapting Russian What? Where? When? quiz
questions into playable English. Return only the requested structured result.

{TRANSLATION_CONSTITUTION}

Translate the question, answer, explanation, accepted-answer criteria, and any
textual handout together. The explanation is reference material, not a source of
new clues. First identify the source's clue-to-answer route and check that every
essential clue is present in the supplied text and still functions in English.
Ordinary knowledge about Russia may remain required knowledge; a mechanism that
works only because of Russian wording or terminology does not survive translation.
Do not mistake presenter remarks, false starts, or transcript corrections in the
explanation for clues that the English puzzle must reproduce.

Use status `translated` for ordinary translation, `adapted` only when a permitted
local repair changes a language-dependent detail, and `untranslatable` when no
fair, self-contained English version is possible. Keep answer_en to the expected
answer; put supporting reasoning in explanation_en. Preserve displayed clue text
exactly only when its exact form is part of the clue; otherwise use standard
English transliteration and typography. Write every final field as polished,
natural English, not as a literal translation. In particular, translate Russian
host cues and stage formulas by what they do in context: do not use “Attention!”
as a routine introduction to a displayed object, and do not replace it with “Here
is...” or “Take a look...” unless the referenced item is actually supplied. Audit
all deictic references such as “this,” “these,” and “which one.” If a generic prop
is not evidential, make the question self-contained; if missing features, choices,
layout, text, sound, or imagery are needed, mark it untranslatable. Remove
non-informative presenter feedback from the final explanation.

When revising, address only the critic's feedback. If the critic concludes that
the mechanism cannot survive or an essential artifact is absent, mark it
untranslatable directly rather than inventing a workaround. Do not churn between
synonymous phrasings once the English is natural and accurate. Describe changes
briefly and specifically.
""".strip()

_CRITIC_INSTRUCTIONS = f"""
You are an independent, skeptical editor of English adaptations of Russian What?
Where? When? quiz questions. Return only the requested structured critique.

{TRANSLATION_CONSTITUTION}

Judge the candidate rather than rubber-stamping it. Apply these gates in order:

1. Decide feasibility once. Try to solve using only question_en and
   handout_text_en, then compare with the Russian source and explanation. Perform
   an explicit referent audit: every “this,” “these,” “here is,” “take a look,” or
   “which one” must refer to supplied text or to a fully described, non-evidential
   generic prop. A source cue such as «Внимание, ...» may be the only surviving
   sign that an image, object, list, or set of choices was shown; an empty handout
   does not make that artifact available. If the route is language-bound, the
   source answer is corrupted, or solving requires absent visual, audible,
   spatial, textual, or choice information, require `untranslatable` immediately.
   Do not first propose a workaround that adds explanation material or replaces
   the original route.
2. If feasible, check that exact letters, scripts, numbers, quotations, and named
   entities were preserved; the answer is unchanged; no explanation-only clue was
   inserted; and every wordplay, terminology, count, or grammar mechanism works.
   Required cultural knowledge is allowed; Russian-only linguistic machinery is
   not. Treat presenter feedback, stage directions, and transcript corrections in
   the explanation as incidental unless the question depends on them.
3. Read question_en, answer_en, explanation_en, acceptance_criteria_en, and
   handout_text_en as an English-only quiz editor. They must sound as though they
   were originally written in English, not translated from Russian. Flag literal
   discourse markers, presenter formulas, Russian syntax, calques, awkward
   collocations, and non-informative transcript chatter. Routine «Внимание» before
   an object should not become the exclamation “Attention!” Exact displayed form
   must be preserved only when that form is itself part of the clue; otherwise
   require standard English transliteration and typography. Also check that every
   field is semantically accurate to the source and does not add, strengthen, or
   silently alter its claims. A clear natural rendering of a proverb, maxim,
   title, or quotation is acceptable even if it is not already a familiar English
   expression.

Request revision only for a material defect affecting correctness, fairness, or
player-facing English. Unnatural translationese that a competent English editor
would immediately rewrite is material; a preference between two equally natural
phrasings is not. Do not cycle through synonyms. Do not request revision solely to change `translated`
versus `adapted`; accept and set accepted_status to the right category. Accept an
untranslatable result only when its reason is specific and no permitted local
repair is plausible. Keep a consistent feasibility judgment across revisions.
On every response, accepted_status is also your feasibility judgment: set it to
`untranslatable` only when no fair English version exists; otherwise set it to
`translated` or `adapted`, even when decision is `revise` for a remaining defect.
""".strip()

_EDITOR_INSTRUCTIONS = f"""
You are the final English-language copy editor for Russian What? Where? When?
quiz questions. You receive the Russian source and a playable English candidate
that has already passed a separate puzzle-integrity review. Return only the
requested structured result.

{TRANSLATION_CONSTITUTION}

Your only job is to make every English field sound as though it was originally
written by a professional English-language quiz editor. Remove translationese,
literal Russian syntax, awkward collocations, redundant transcript language,
and presenter formulas that do not sound natural in English. Prefer clear,
economical sentences that work when read aloud.

Edit conservatively. `unchanged` is a successful result, not a failure to act.
Make a change only when it clearly improves the English. Keep the candidate when
an alternative is merely different, longer, more abstract, or more explanatory.
Do not replace concrete historical, technical, or cultural terms with loose
near-synonyms for style. Do not turn a concise riddle into meta-language that
explains its wordplay.

This is copy editing, not puzzle rewriting. Preserve the answer, every clue and
fact, all qualifications and uncertainty, the reasoning route, intended
ambiguity, and approximate difficulty. Do not add a hint from the explanation to
the question, make an inference explicit, resolve ambiguity, correct source facts,
or substitute a new mechanism. Exact displayed letters or wording must remain
exact when their form is part of the clue. Accuracy outranks elegance.

Treat the critic-approved candidate as authoritative for substantive content,
including any factual or terminology correction already made upstream. Consult
the Russian source to prevent semantic drift, never to reverse the candidate back
to a source error. Do not independently change or restore names, dates, numbers,
scientific terms, historical labels, causal claims, or other factual content. If
such a discrepancy appears to require intervention, leave the candidate unchanged
or return `needs_rework`; do not adjudicate it in this copy-editing stage.

Use `unchanged` only when no competent English editor would materially improve
the candidate. Use `edited` when you can safely improve the prose. Use
`needs_rework` when the existing English is awkward but making it natural would
risk changing clue information, meaning, or difficulty. For `needs_rework`, copy
all five English fields unchanged and explain the conflict. Do not force a
smooth-sounding rewrite when the safe choice is to flag it.

A mechanically literal phrase such as “the population grows only through
unnatural means” is not fixed by swapping “grows” for “increases” or changing a
preposition. A rendering such as “Which state can increase its population only
artificially?” demonstrates the required degree of recasting when it preserves
the source's intended contrast and difficulty; otherwise flag `needs_rework`.
Likewise, keep a natural concise question such as “What square thing do we call a
ring?” instead of expanding it into an explanation of the wordplay.
""".strip()


# Reasoning tokens count against max_tokens; higher efforts need headroom or the
# model runs out of budget before emitting its final structured output.
_MAX_TOKENS_BY_EFFORT = {
    "none": 2500,
    "low": 4000,
    "medium": 12000,
    "high": 24000,
    "xhigh": 48000,
    "max": 60000,
}


class AgentsTranslationClient:
    """Three independent SDK agents with application-owned orchestration."""

    def __init__(
        self,
        *,
        min_request_interval: float = 0.0,
        transient_retries: int = 6,
    ):
        self.translator_model = TRANSLATOR_MODEL
        self.critic_model = CRITIC_MODEL
        self.editor_model = EDITOR_MODEL
        self.reasoning_effort = REASONING_EFFORT
        self.min_request_interval = min_request_interval
        self.transient_retries = transient_retries
        self._last_request_at = 0.0
        self._request_lock = asyncio.Lock()

        # The Agents SDK stamps a fresh prompt_cache_key on every run, which
        # routes each request to a different cache partition: the shared
        # constitution prefix is written every time and never read back (we
        # measured cache_write on 100% of calls and zero hits). A stable
        # per-agent key restores prefix caching; the version suffix keeps a
        # prompt change from colliding with a stale partition.
        def settings_for(role: str) -> ModelSettings:
            return ModelSettings(
                max_tokens=_MAX_TOKENS_BY_EFFORT[REASONING_EFFORT],
                include_usage=True,
                store=False,
                reasoning={"effort": REASONING_EFFORT},
                extra_args={
                    "prompt_cache_key": (
                        f"shgk-translate-p{TRANSLATION_POLICY_VERSION}"
                        f"-w{TRANSLATION_WORKFLOW_VERSION}-{role}"
                    )
                },
            )

        self.translator = Agent(
            name="ChGK English translator",
            instructions=_TRANSLATOR_INSTRUCTIONS,
            model=translator_model,
            model_settings=settings_for("translator"),
            output_type=TranslationCandidate,
        )
        self.critic = Agent(
            name="ChGK translation critic",
            instructions=_CRITIC_INSTRUCTIONS,
            model=critic_model,
            model_settings=settings_for("critic"),
            output_type=TranslationCritique,
        )
        self.editor = Agent(
            name="ChGK English copy editor",
            instructions=_EDITOR_INSTRUCTIONS,
            model=editor_model,
            model_settings=settings_for("editor"),
            output_type=EnglishEdit,
        )
        self.run_config = RunConfig(
            tracing_disabled=True,
            workflow_name="ChGK translation and critique",
        )


    async def _run(
        self,
        agent: Agent,
        prompt: str,
        output_type: type[TranslationCandidate | TranslationCritique | EnglishEdit],
    ) -> AgentCall:
        result = None
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
                result = await Runner.run(
                    agent,
                    prompt,
                    max_turns=1,
                    run_config=self.run_config,
                )
                break
            except Exception as error:
                if not is_transient_error(error) or retry >= self.transient_retries:
                    raise
                await asyncio.sleep(min(60.0, 5.0 * (2**retry)))
        if result is None:
            raise AssertionError("unreachable")
        output = result.final_output_as(output_type, raise_if_incorrect_type=True)
        usage = result.context_wrapper.usage
        return AgentCall(
            output=output,
            usage=UsageTotals(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_input_tokens=usage.input_tokens_details.cache_write_tokens,
                reasoning_output_tokens=usage.output_tokens_details.reasoning_tokens,
            ),
        )

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
            self.translator,
            json.dumps(task, ensure_ascii=False, indent=2),
            TranslationCandidate,
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
            self.critic,
            json.dumps(task, ensure_ascii=False, indent=2),
            TranslationCritique,
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
            self.editor,
            json.dumps(task, ensure_ascii=False, indent=2),
            EnglishEdit,
        )


_POOLED_CLIENT_INSTALLED = False


def install_pooled_openai_client(max_connections: int = 2048) -> None:
    """Widen the shared OpenAI connection pool for high-concurrency runs.

    The SDK's default pool caps at 1000 connections, which silently becomes the
    throughput ceiling long before the account's rate limits do. Token usage is
    what actually binds here (~25k tokens/question against 10M tokens/min), so
    the pool must not be the limiting factor.
    """

    global _POOLED_CLIENT_INSTALLED
    if _POOLED_CLIENT_INSTALLED:
        return
    set_default_openai_client(
        AsyncOpenAI(
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections // 2,
                ),
                timeout=httpx.Timeout(600.0, connect=15.0),
            )
        )
    )
    _POOLED_CLIENT_INSTALLED = True



@dataclass(slots=True)
class WorkflowResult:
    candidate: TranslationCandidate
    translation_attempts: int
    critic_attempts: int
    editor_attempts: int
    usage: UsageTotals
    history: list[dict[str, object]]
    pre_editor_candidate: TranslationCandidate | None
    editor_result: EnglishEdit | None
    editor_usage: UsageTotals
    editor_status: Literal["unchanged", "edited", "needs_rework", "skipped"]


def workflow_result_dict(
    source: TranslationInput,
    result: WorkflowResult,
) -> dict[str, object]:
    return {
        "source": {
            **source.prompt_dict(),
            "source": source.source,
            "source_question_id": source.source_question_id,
            "source_content_hash": source.source_content_hash,
        },
        "output": result.candidate.model_dump(),
        "workflow": {
            "translation_attempts": result.translation_attempts,
            "critic_attempts": result.critic_attempts,
            "editor_attempts": result.editor_attempts,
            "editor_status": result.editor_status,
            "usage": asdict(result.usage),
            "pre_editor_candidate": (
                result.pre_editor_candidate.model_dump()
                if result.pre_editor_candidate
                else None
            ),
            "editor_result": (
                result.editor_result.model_dump() if result.editor_result else None
            ),
            "editor_usage": asdict(result.editor_usage),
            "history": result.history,
        },
    }


def _local_issues(candidate: TranslationCandidate) -> list[str]:
    if candidate.status == "untranslatable":
        return (
            []
            if candidate.untranslatable_reason.strip()
            else ["The untranslatable result has no concrete reason."]
        )
    missing = [
        name
        for name, value in (
            ("question_en", candidate.question_en),
            ("answer_en", candidate.answer_en),
            ("explanation_en", candidate.explanation_en),
            ("changes_description", candidate.changes_description),
        )
        if not value.strip()
    ]
    return (
        [f"Required translated fields are empty: {', '.join(missing)}."]
        if missing
        else []
    )


def _english_fields(candidate: TranslationCandidate) -> dict[str, str]:
    return {
        "question_en": candidate.question_en,
        "answer_en": candidate.answer_en,
        "explanation_en": candidate.explanation_en,
        "acceptance_criteria_en": candidate.acceptance_criteria_en,
        "handout_text_en": candidate.handout_text_en,
    }


async def _finalize_with_editor(
    client: TranslationClient,
    source: TranslationInput,
    candidate: TranslationCandidate,
    *,
    translation_attempts: int,
    critic_attempts: int,
    usage: UsageTotals,
    history: list[dict[str, object]],
) -> WorkflowResult:
    if candidate.status == "untranslatable":
        return WorkflowResult(
            candidate=candidate,
            translation_attempts=translation_attempts,
            critic_attempts=critic_attempts,
            editor_attempts=0,
            usage=usage,
            history=history,
            pre_editor_candidate=None,
            editor_result=None,
            editor_usage=UsageTotals(),
            editor_status="skipped",
        )

    pre_editor = candidate.model_copy(deep=True)
    edited_call = await client.edit(source, pre_editor)
    usage.add(edited_call.usage)
    edit = edited_call.output
    if not isinstance(edit, EnglishEdit):
        raise TypeError("editor returned the wrong structured output")

    original_fields = _english_fields(pre_editor)
    edited_fields = {
        name: getattr(edit, name)
        for name in (
            "question_en",
            "answer_en",
            "explanation_en",
            "acceptance_criteria_en",
            "handout_text_en",
        )
    }
    missing = [
        name
        for name in ("question_en", "answer_en", "explanation_en")
        if not edited_fields[name].strip()
    ]
    if edit.decision == "needs_rework" or missing:
        reason = edit.needs_rework_reason.strip()
        if missing:
            reason = f"Editor returned empty required fields: {', '.join(missing)}."
        edit = edit.model_copy(
            update={
                "decision": "needs_rework",
                **original_fields,
                "needs_rework_reason": reason
                or "Safe English copy editing requires substantive puzzle changes.",
            }
        )
        final = pre_editor
        editor_status = "needs_rework"
    else:
        changed = edited_fields != original_fields
        editor_status = "edited" if changed else "unchanged"
        edit = edit.model_copy(
            update={
                "decision": editor_status,
                "needs_rework_reason": "",
            }
        )
        if changed:
            summary = edit.edit_summary.strip() or "Polished the English prose."
            existing = pre_editor.changes_description.rstrip()
            changes_description = f"{existing} English copy edit: {summary}".strip()
            final = pre_editor.model_copy(
                update={**edited_fields, "changes_description": changes_description}
            )
        else:
            final = pre_editor

    history.append({"editor": edit.model_dump()})
    return WorkflowResult(
        candidate=final,
        translation_attempts=translation_attempts,
        critic_attempts=critic_attempts,
        editor_attempts=1,
        usage=usage,
        history=history,
        pre_editor_candidate=pre_editor,
        editor_result=edit,
        editor_usage=edited_call.usage,
        editor_status=editor_status,
    )


async def run_translation_workflow(
    client: TranslationClient,
    source: TranslationInput,
    *,
    max_revisions: int = 2,
) -> WorkflowResult:
    usage = UsageTotals()
    history: list[dict[str, object]] = []
    previous: TranslationCandidate | None = None
    feedback: TranslationCritique | None = None
    last_playable: TranslationCandidate | None = None

    for attempt in range(max_revisions + 1):
        proposed = await client.propose(source, previous=previous, feedback=feedback)
        usage.add(proposed.usage)
        candidate = proposed.output
        if not isinstance(candidate, TranslationCandidate):
            raise TypeError("translator returned the wrong structured output")

        reviewed = await client.critique(source, candidate)
        usage.add(reviewed.usage)
        critique = reviewed.output
        if not isinstance(critique, TranslationCritique):
            raise TypeError("critic returned the wrong structured output")

        local_issues = _local_issues(candidate)
        if candidate.status != "untranslatable" and not local_issues:
            last_playable = candidate
        if critique.decision == "accept" and (candidate.status == "untranslatable") != (
            critique.accepted_status == "untranslatable"
        ):
            local_issues.append(
                "The critic and writer disagree on whether the question is translatable."
            )
        if local_issues:
            critique = critique.model_copy(
                update={
                    "decision": "revise",
                    "issues": [*critique.issues, *local_issues],
                    "revision_instructions": " ".join(
                        filter(None, [critique.revision_instructions, *local_issues])
                    ),
                }
            )

        history.append(
            {
                "attempt": attempt + 1,
                "candidate": candidate.model_dump(),
                "critique": critique.model_dump(),
            }
        )
        if critique.decision == "accept":
            candidate = candidate.model_copy(
                update={"status": critique.accepted_status}
            )
            return await _finalize_with_editor(
                client,
                source,
                candidate,
                translation_attempts=attempt + 1,
                critic_attempts=attempt + 1,
                usage=usage,
                history=history,
            )

        if attempt == max_revisions:
            if (
                critique.accepted_status != "untranslatable"
                and last_playable is not None
            ):
                salvaged = last_playable.model_copy(
                    update={"status": critique.accepted_status}
                )
                return await _finalize_with_editor(
                    client,
                    source,
                    salvaged,
                    translation_attempts=attempt + 1,
                    critic_attempts=attempt + 1,
                    usage=usage,
                    history=history,
                )
            reason = critique.summary.strip() or critique.revision_instructions.strip()
            exhausted = TranslationCandidate(
                status="untranslatable",
                question_en="",
                answer_en="",
                explanation_en="",
                acceptance_criteria_en="",
                handout_text_en="",
                changes_description="No candidate passed independent review within the revision limit.",
                untranslatable_reason=reason
                or "The translation did not pass independent review.",
            )
            return await _finalize_with_editor(
                client,
                source,
                exhausted,
                translation_attempts=attempt + 1,
                critic_attempts=attempt + 1,
                usage=usage,
                history=history,
            )

        previous = candidate
        feedback = critique

    raise AssertionError("unreachable")


class TranslationPipeline:
    """Stage 4: translate canonical questions that have no current translation."""

    def __init__(self, database: str | Path = DEFAULT_DB_PATH):
        self.database = Path(database)

    def _pending_inputs(
        self,
        *,
        limit: int,
        offset: int,
        refresh: bool,
    ) -> list[TranslationInput]:
        # A translation is current when it was produced from the question text
        # that is in the database now; anything else is missing or stale.
        freshness = (
            ""
            if refresh
            else """
            AND NOT EXISTS (
                SELECT 1 FROM translations AS t
                WHERE t.question_id = q.id AND t.content_hash = q.content_hash
            )
            """
        )
        with connect(self.database, read_only=True) as connection:
            rows = connection.execute(
                f"""
                SELECT q.id, q.content_hash, q.question, q.answer, q.explanation,
                       q.acceptance_criteria, q.handout_text, p.title AS package_title
                FROM questions_canonical AS q
                JOIN packages AS p ON p.id = q.package_id
                WHERE 1 {freshness}
                ORDER BY q.id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [
            TranslationInput(
                question_id=row["id"],
                content_hash=row["content_hash"],
                question=row["question"],
                answer=row["answer"],
                explanation=row["explanation"],
                acceptance_criteria=row["acceptance_criteria"],
                handout_text=row["handout_text"],
                package_title=row["package_title"],
            )
            for row in rows
        ]

    def _save(self, source: TranslationInput, result: WorkflowResult) -> None:
        candidate = result.candidate
        with connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO translations (
                    question_id, content_hash, status, question_en, answer_en,
                    explanation_en, acceptance_criteria_en, handout_text_en,
                    changes_description, untranslatable_reason, editor_status,
                    translation_attempts, critic_attempts, editor_attempts,
                    api_requests, input_tokens, cached_input_tokens,
                    cache_write_input_tokens, output_tokens,
                    reasoning_output_tokens, completed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(question_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    question_en = excluded.question_en,
                    answer_en = excluded.answer_en,
                    explanation_en = excluded.explanation_en,
                    acceptance_criteria_en = excluded.acceptance_criteria_en,
                    handout_text_en = excluded.handout_text_en,
                    changes_description = excluded.changes_description,
                    untranslatable_reason = excluded.untranslatable_reason,
                    editor_status = excluded.editor_status,
                    translation_attempts = excluded.translation_attempts,
                    critic_attempts = excluded.critic_attempts,
                    editor_attempts = excluded.editor_attempts,
                    api_requests = excluded.api_requests,
                    input_tokens = excluded.input_tokens,
                    cached_input_tokens = excluded.cached_input_tokens,
                    cache_write_input_tokens = excluded.cache_write_input_tokens,
                    output_tokens = excluded.output_tokens,
                    reasoning_output_tokens = excluded.reasoning_output_tokens,
                    completed_at = excluded.completed_at
                """,
                (
                    source.question_id,
                    source.content_hash,
                    candidate.status,
                    candidate.question_en,
                    candidate.answer_en,
                    candidate.explanation_en,
                    candidate.acceptance_criteria_en,
                    candidate.handout_text_en,
                    candidate.changes_description,
                    candidate.untranslatable_reason,
                    result.editor_status,
                    result.translation_attempts,
                    result.critic_attempts,
                    result.editor_attempts,
                    result.usage.requests,
                    result.usage.input_tokens,
                    result.usage.cached_input_tokens,
                    result.usage.cache_write_input_tokens,
                    result.usage.output_tokens,
                    result.usage.reasoning_output_tokens,
                    datetime.now(UTC).isoformat(),
                ),
            )

    async def run(
        self,
        client: TranslationClient,
        *,
        limit: int = 10,
        offset: int = 0,
        max_revisions: int = 2,
        refresh: bool = False,
        fail_fast: bool = False,
        workers: int = 1,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        inputs = self._pending_inputs(limit=limit, offset=offset, refresh=refresh)
        counts = {"selected": len(inputs), "completed": 0, "errors": 0}
        semaphore = asyncio.Semaphore(max(1, workers))
        finished = 0

        async def translate_one(source: TranslationInput) -> None:
            nonlocal finished
            async with semaphore:
                try:
                    result = await run_translation_workflow(
                        client, source, max_revisions=max_revisions
                    )
                except Exception as error:
                    counts["errors"] += 1
                    finished += 1
                    if progress:
                        progress(
                            f"[{finished}/{len(inputs)}] {source.question_id}: "
                            f"ERROR: {error}"
                        )
                    if fail_fast:
                        raise
                    return
                self._save(source, result)
                counts["completed"] += 1
                finished += 1
                if progress:
                    progress(
                        f"[{finished}/{len(inputs)}] {source.question_id}: "
                        f"{result.candidate.status} "
                        f"({result.translation_attempts} attempt(s))"
                    )

        async with asyncio.TaskGroup() as group:
            for source in inputs:
                group.create_task(translate_one(source))
        return counts

    def stats(self) -> dict[str, object]:
        with connect(self.database, read_only=True) as connection:
            by_status = {
                row["status"]: row["questions"]
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS questions
                    FROM translations GROUP BY status
                    """
                )
            }
            totals = connection.execute(
                """
                SELECT COUNT(*) AS translations,
                       SUM(api_requests) AS api_requests,
                       SUM(input_tokens) AS input_tokens,
                       SUM(cached_input_tokens) AS cached_input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(reasoning_output_tokens) AS reasoning_output_tokens
                FROM translations
                """
            ).fetchone()
            pending = connection.execute(
                """
                SELECT COUNT(*) FROM questions_canonical AS q
                WHERE NOT EXISTS (
                    SELECT 1 FROM translations AS t
                    WHERE t.question_id = q.id AND t.content_hash = q.content_hash
                )
                """
            ).fetchone()[0]
        return {"by_status": by_status, "pending": pending, **dict(totals)}
